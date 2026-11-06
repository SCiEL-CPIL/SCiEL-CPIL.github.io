import time
from itertools import combinations

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit

    # Precompute base cycle edges (the 40-cycle)
    base_edges = [(i, (i + 1) % N) for i in range(N)]
    base_edge_set = set()
    for u, v in base_edges:
        if u > v:
            u, v = v, u
        base_edge_set.add((u, v))

    # We add chords from vertex 0 to some subset S of {2, ..., N-1}
    # A chord (0, s) enables:
    #   - cycle of length s+1: [0, 1, ..., s]  (using base edges 0-1, 1-2, ..., (s-1)-s, and chord s-0)
    #   - cycle of length N-s+1: [0, N-1, N-2, ..., s] (using base edges (N-1)-0, (N-2)-(N-1), ..., s-(s+1), and chord s-0)
    #   - for two chords a < b: cycle of length b-a+2: [0, b, b-1, ..., a] (using chord 0-b, base edges b-(b-1), ..., (a+1)-a, chord a-0)
    
    # Target: cover all lengths 3..N
    required = set(range(3, N + 1))
    
    # Length N is always covered by the base cycle
    covered = {N}
    
    # For a given set S of chord endpoints, compute which lengths are covered
    def compute_covered(S):
        cov = {N}
        S_sorted = sorted(S)
        for s in S_sorted:
            l1 = s + 1
            if 3 <= l1 <= N:
                cov.add(l1)
            l2 = N - s + 1
            if 3 <= l2 <= N:
                cov.add(l2)
        for i in range(len(S_sorted)):
            for j in range(i + 1, len(S_sorted)):
                a = S_sorted[i]
                b = S_sorted[j]
                l = b - a + 2
                if 3 <= l <= N:
                    cov.add(l)
        return cov

    # Greedy search for minimal S
    # Candidates: 2..N-1 (chord endpoints; 1 and N-1 are already base edges, so exclude)
    # Actually, chord (0,1) and (0, N-1) are base edges, so we only consider 2..N-2
    candidates = list(range(2, N))  # 2..39
    
    S = set()
    covered = {N}
    
    while not required.issubset(covered):
        if time.time() > deadline - 5.0:
            break  # safety: if we can't finish, use what we have (shouldn't happen with greedy)
        
        best_s = None
        best_new_count = -1
        for s in candidates:
            if s in S:
                continue
            new_S = S | {s}
            new_cov = compute_covered(new_S)
            new_count = len(new_cov & required)
            if new_count > best_new_count:
                best_new_count = new_count
                best_s = s
        if best_s is None:
            break
        S.add(best_s)
        covered = compute_covered(S)
    
    # Now we have S. Build the edge list.
    edges = list(base_edges)
    for s in S:
        edges.append((0, s))
    
    # Remove duplicate edges (e.g., if s=1 or s=N-1, but we excluded those)
    unique_edges = []
    seen = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        if (u, v) not in seen:
            seen.add((u, v))
            unique_edges.append((u, v))
    edges = unique_edges
    
    # Build cycles for each length 3..N
    cycles_by_len = {}
    
    # Hamiltonian cycle
    cycles_by_len[N] = list(range(N))
    
    S_sorted = sorted(S)
    
    # From single chords
    for s in S_sorted:
        l1 = s + 1
        if 3 <= l1 <= N and l1 not in cycles_by_len:
            cycles_by_len[l1] = list(range(s + 1))
        
        l2 = N - s + 1
        if 3 <= l2 <= N and l2 not in cycles_by_len:
            # Cycle: 0, N-1, N-2, ..., s
            cycles_by_len[l2] = [0] + list(range(N - 1, s - 1, -1))
    
    # From pairs of chords
    for i in range(len(S_sorted)):
        for j in range(i + 1, len(S_sorted)):
            a = S_sorted[i]
            b = S_sorted[j]
            l = b - a + 2
            if 3 <= l <= N and l not in cycles_by_len:
                # Cycle: 0, b, b-1, ..., a
                cycles_by_len[l] = [0] + list(range(b, a - 1, -1))
    
    # Check if all lengths are covered
    for L in range(3, N + 1):
        if L not in cycles_by_len:
            # Fallback: this shouldn't happen with a correct greedy, but just in case
            # Use a simple construction: if L <= 21, use [0,1,...,L-1] (requires chord to L-1)
            # If L > 21, use [0, 39, ..., 41-L] (requires chord to 41-L)
            if L <= 21:
                cycles_by_len[L] = list(range(L))
            elif L <= 39:
                k = 41 - L
                cycles_by_len[L] = [0] + list(range(39, k - 1, -1))
            else:
                cycles_by_len[L] = list(range(N))
    
    final_cycles = [cycles_by_len[L] for L in range(3, N + 1)]
    
    # Verify all cycles are valid (edges exist)
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    for L, cyc in zip(range(3, N + 1), final_cycles):
        for i in range(L):
            a = cyc[i]
            b = cyc[(i + 1) % L]
            if a > b:
                a, b = b, a
            if (a, b) not in edge_set:
                # This is a bug in our construction logic. Fallback to seed.
                return {
                    'edges': base_edges + [(0, k) for k in range(2, 21)],
                    'cycles': [
                        list(range(L)) if L <= 21 else ([0] + list(range(39, 41 - L - 1, -1)) if L <= 39 else list(range(N)))
                        for L in range(3, N + 1)
                    ]
                }
    
    return {'edges': edges, 'cycles': final_cycles}
