import time
import sys

N = 40

def construct():
    """
    Constructs a sparse pancyclic graph on N=40 vertices.
    Uses a Hamiltonian cycle plus 19 chords from vertex 0 to cover all cycle lengths 3..40.
    Total edges: 40 (Hamiltonian) + 19 (chords) = 59.
    Score: 59 - 40 = 19.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # Keep a 10s margin for 60s limit
    
    edges = []
    edge_set = set()
    
    # 1. Hamiltonian Cycle edges
    for i in range(N):
        u = i
        v = (i + 1) % N
        if u > v:
            u, v = v, u
        edge = (u, v)
        if edge not in edge_set:
            edges.append(edge)
            edge_set.add(edge)
    
    # 2. Chords from vertex 0 to vertices 2..20
    # Chord (0, j) creates cycles of length j+1 and N-j+1
    # We need to cover lengths 3..39.
    # j=2 -> 3, 39
    # j=3 -> 4, 38
    # ...
    # j=20 -> 21, 21
    for j in range(2, 21):
        u = 0
        v = j
        if u > v:
            u, v = v, u
        edge = (u, v)
        if edge not in edge_set:
            edges.append(edge)
            edge_set.add(edge)
            
    # Check time
    if time.time() > deadline:
        # Return best effort, though construction is fast
        pass

    # 3. Generate Cycles
    cycles = []
    
    # Helper to get path from 0 to k clockwise (0, 1, ..., k)
    def get_path_cw(k):
        return list(range(k + 1))
        
    # Helper to get path from 0 to k counter-clockwise (0, N-1, ..., k)
    def get_path_ccw(k):
        if k == 0:
            return [0]
        # 0, N-1, N-2, ..., k
        return [0] + list(range(N - 1, k - 1, -1))
        
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle
            # 0, 1, ..., N-1
            cycle = list(range(N))
        else:
            # L is between 3 and N-1
            # Check if L <= N/2 + 1 ?
            # Chord j = L-1 covers L if L-1 <= N/2 ?
            # Actually, we defined chords for j in 2..20.
            # If L-1 is in [2, 20], use chord (0, L-1) and path 0..L-1.
            # If 41-L is in [2, 20], use chord (0, 41-L) and path 0..41-L (CCW).
            
            j1 = L - 1
            j2 = N - L + 1
            
            if 2 <= j1 <= 20:
                # Use chord (0, j1) and CW path
                # Path: 0, 1, ..., j1
                # Cycle: 0, 1, ..., j1, 0
                # Vertices: 0, 1, ..., L-1
                cycle = list(range(L))
            elif 2 <= j2 <= 20:
                # Use chord (0, j2) and CCW path
                # Path: 0, N-1, ..., j2
                # Cycle: 0, N-1, ..., j2, 0
                # Vertices: 0, N-1, ..., j2
                # Count: 1 (0) + (N-1 - j2 + 1) = N - j2 + 1 = L
                # Order: 0, N-1, N-2, ..., j2
                cycle = [0] + list(range(N - 1, j2 - 1, -1))
            else:
                # This should not happen for N=40 and L in 3..39
                # Fallback: just use range(L) which might be invalid if edges missing, 
                # but our logic covers all cases.
                cycle = list(range(L))
                
        cycles.append(cycle)
        
    return {'edges': edges, 'cycles': cycles}

if __name__ == "__main__":
    res = construct()
    # Basic validation for debugging
    edge_set = set(tuple(sorted(e)) for e in res['edges'])
    for i, cyc in enumerate(res['cycles']):
        L = i + 3
        assert len(cyc) == L, f"Cycle length mismatch for L={L}, got {len(cyc)}"
        assert len(set(cyc)) == L, f"Duplicate vertices in cycle L={L}"
        for j in range(L):
            u = cyc[j]
            v = cyc[(j+1)%L]
            e = tuple(sorted((u, v)))
            assert e in edge_set, f"Edge {e} missing for cycle L={L}"
    print(f"Edges: {len(res['edges'])}, Score: {len(res['edges']) - 40}")
