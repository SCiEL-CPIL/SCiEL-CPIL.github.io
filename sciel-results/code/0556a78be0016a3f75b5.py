import random
import time

N = 40

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 80.0  # 80 seconds, well within 90s limit
    
    # Target differences to cover: 1 to 38
    target_diffs = set(range(1, 39))
    n_diffs = len(target_diffs)
    
    best_S = None
    best_size = 99
    
    def get_diffs(S):
        S_sorted = sorted(S)
        diffs = set()
        for i in range(len(S_sorted)):
            for j in range(i+1, len(S_sorted)):
                d = S_sorted[j] - S_sorted[i]
                if d in target_diffs:
                    diffs.add(d)
        return diffs
    
    def coverage(S):
        return len(get_diffs(S))
    
    def is_valid(S):
        return get_diffs(S) == target_diffs
    
    # Try sizes from 9 upward
    for size in [9, 10, 11, 12]:
        if time.time() - start_time > timeout:
            break
        
        found = False
        # Multiple random restarts
        num_restarts = 50 if size <= 10 else 20
        for restart in range(num_restarts):
            if time.time() - start_time > timeout:
                break
            
            # Initialize S
            if size == 2:
                S = [1, 39]
            else:
                S = [1, 39]
                candidates = list(range(2, 39))
                random.shuffle(candidates)
                S.extend(candidates[:size-2])
            
            current_cov = coverage(S)
            
            # Local search with simulated annealing style
            for it in range(50000):
                if time.time() - start_time > timeout:
                    break
                
                if current_cov == n_diffs:
                    best_S = S
                    best_size = size
                    found = True
                    break
                
                # Pick random index
                idx = random.randint(0, len(S)-1)
                old_val = S[idx]
                new_val = random.randint(1, 39)
                
                if new_val in S and new_val != old_val:
                    continue
                    
                S[idx] = new_val
                new_cov = coverage(S)
                
                if new_cov >= current_cov:
                    current_cov = new_cov
                else:
                    S[idx] = old_val
        
        if found:
            break
    
    if best_S is None:
        # Fallback: use a known good construction
        # Path + spokes with a larger S
        S = list(range(1, 40))  # All vertices as spokes - this is the star+path
        best_S = S
        best_size = 39
    
    # Construct graph
    path_edges = [(i, i+1) for i in range(1, N-1)]
    spoke_edges = [(0, s) for s in best_S]
    edges = path_edges + spoke_edges
    
    S_set = set(best_S)
    cycles = []
    for L in range(3, N+1):
        diff_needed = L - 2
        found_cycle = False
        for u in best_S:
            v = u + diff_needed
            if v in S_set:
                cycle_verts = [0] + list(range(u, v+1))
                cycles.append(cycle_verts)
                found_cycle = True
                break
        if not found_cycle:
            raise Exception("Cycle not found for length " + str(L))
    
    return {'edges': edges, 'cycles': cycles}
