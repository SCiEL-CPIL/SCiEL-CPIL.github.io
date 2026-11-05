import time
import random

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin

    # Target differences to cover: 1 to 38
    # Corresponds to cycle lengths 3 to 40
    # L = v - u + 2 => v - u = L - 2
    # If L=3, diff=1. If L=40, diff=38.
    target_diffs = set(range(1, 39))
    
    best_S = None
    best_size = 999
    
    # We want to minimize |S|.
    # Try sizes from 10 upwards.
    # If we find a valid set of size 10, we stop.
    # If not, we try 11, etc.
    
    def get_coverage(S):
        S_sorted = sorted(S)
        diffs = set()
        for i in range(len(S_sorted)):
            for j in range(i+1, len(S_sorted)):
                d = S_sorted[j] - S_sorted[i]
                if d in target_diffs:
                    diffs.add(d)
        return len(diffs)
    
    def is_valid(S):
        S_sorted = sorted(S)
        diffs = set()
        for i in range(len(S_sorted)):
            for j in range(i+1, len(S_sorted)):
                d = S_sorted[j] - S_sorted[i]
                if d in target_diffs:
                    diffs.add(d)
        return diffs == target_diffs

    # Local Search to find a valid S of a specific size
    def find_valid_S(target_size, time_budget):
        if time.time() > time_budget:
            return None
            
        # Initialize S
        # Use a heuristic init: spread out
        S = [1, 39]
        candidates = list(range(2, 39))
        random.shuffle(candidates)
        S.extend(candidates[:target_size-2])
        
        current_score = get_coverage(S)
        
        # If we already have full coverage, return
        if current_score == len(target_diffs):
            return S

        for iter in range(20000):
            if time.time() > time_budget:
                return None
            
            if current_score == len(target_diffs):
                return S
            
            # Pick a random index to change
            idx = random.randint(0, len(S)-1)
            old_val = S[idx]
            
            # Pick a new value
            # Bias towards values that might help fill gaps? 
            # For now, random is simple.
            new_val = random.randint(1, 39)
            
            # Ensure distinctness in S
            if new_val in S and new_val != old_val:
                continue
                
            S[idx] = new_val
            
            new_score = get_coverage(S)
            
            if new_score >= current_score:
                current_score = new_score
            else:
                # Revert
                S[idx] = old_val
                
        return None

    # Try to find minimal S
    # Start with size 10, then 11, etc.
    # We allocate time budget to each size.
    # If we find a solution of size 10, it's optimal (likely).
    
    # Allocate 30s for size 10, 10s for 11, 5s for 12.
    # Or just iterate and check time.
    
    for size in [10, 11, 12, 13]:
        if time.time() > deadline - 5:
            break
            
        # Try multiple restarts for each size
        restarts = 10 if size == 10 else 5
        
        for r in range(restarts):
            if time.time() > deadline - 5:
                break
                
            S = find_valid_S(size, deadline - 2)
            if S is not None:
                if len(S) < best_size:
                    best_S = S
                    best_size = len(S)
        
        # If we found a solution of size 10, we can stop searching for larger sizes
        if best_size == 10:
            break

    if best_S is None:
        # Fallback to a known valid construction if search fails
        # Use the 19-chord hub solution from current program as fallback
        edges = set()
        for i in range(N):
            u, v = i, (i + 1) % N
            if u > v: u, v = v, u
            edges.add((u, v))
        for d in range(2, 21):
            edges.add((0, d))
            
        cycles = []
        for L in range(3, N + 1):
            if L == N:
                cycles.append(list(range(N)))
            elif L <= 21:
                cycles.append(list(range(L)))
            else:
                m = N - L
                # Cycle 0-m-(m+1)-...-39-0
                # Length: 1 + (N-m) = N-m+1? 
                # Vertices: 0, m, ..., 39.
                # Count: 1 + (39-m+1) = 40-m.
                # We want length L. 40-m = L => m = 40-L.
                # If L=39, m=1. Cycle 0-1-...-39-0 is length 40? No.
                # The cycle 0-m-...-39-0 uses edge (0,39) and (0,m).
                # Path m to 39 has length 39-m.
                # Total length 1 (0-m) + (39-m) + 1 (39-0) = 40-m.
                # So if m = 40-L, length is L.
                # If L=39, m=1. But (0,1) is in Hamiltonian cycle.
                # Is (0,1) a "chord"? In this fallback, yes.
                # Cycle: 0, 1, 2, ..., 39. Length 40? No, 0-1-2-...-39-0 is 40 edges.
                # Wait, 0-1-2-...-39 is 39 edges. Close with 39-0 is 40th edge.
                # So 0-1-...-39-0 is length 40.
                # To get length 39, we remove one vertex from the Hamiltonian cycle?
                # No, we use chord (0,2).
                # Cycle: 0-2-3-...-39-0.
                # Vertices: 0, 2, ..., 39.
                # Edges: (0,2), (2,3), ..., (38,39), (39,0).
                # Count: 1 + 37 + 1 = 39.
                # So for L=39, m=2.
                # Formula: m = 40 - L + 1?
                # If L=39, m=2. 40-39+1 = 2.
                # If L=21, m=20. 40-21+1 = 20.
                # If L=22, m=19. 40-22+1 = 19.
                # So m = 41 - L.
                m = 41 - L
                cycle_verts = [0] + list(range(m, N))
                cycles.append(cycle_verts)
                
        return {'edges': list(edges), 'cycles': cycles}

    # Construct graph with best_S
    S_sorted = sorted(best_S)
    S_set = set(S_sorted)
    
    # Path edges: 1-2, ..., 38-39
    path_edges = [(i, i+1) for i in range(1, N-1)]
    # Spoke edges: 0-s for s in S
    spoke_edges = [(0, s) for s in S_sorted]
    
    edges = path_edges + spoke_edges
    
    # Build mapping from diff to pair (u, v)
    # We need for each L in 3..40, a pair u, v in S such that v-u = L-2
    cycles = []
    for L in range(3, N+1):
        diff_needed = L - 2
        # Find u, v in best_S such that v - u == diff_needed
        found = False
        for u in S_sorted:
            v = u + diff_needed
            if v in S_set:
                # Cycle: 0 -> u -> ... -> v -> 0
                # Vertices: 0, u, u+1, ..., v
                cycle_verts = [0] + list(range(u, v+1))
                cycles.append(cycle_verts)
                found = True
                break
        if not found:
            # This should not happen if is_valid passed
            # Fallback: raise error or use a dummy
            # To be safe, we'll try to find any cycle of length L using BFS?
            # But this path is guaranteed to exist if coverage is correct.
            # If it fails, the search was flawed.
            raise Exception("Cycle not found for length " + str(L))
            
    return {'edges': edges, 'cycles': cycles}

if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res['edges'])}, Score: {len(res['edges'])-40}")
