import random
import time

N = 40

def construct():
    start_time = time.time()
    timeout = 85.0  # Leave 5s margin for 90s limit
    
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

    # We know size 9 is impossible (max 36 diffs).
    # We suspect size 10 might be the bottleneck or the current best.
    # We will aggressively search for size 10, then 11.
    
    for size in [10, 11]:
        if time.time() - start_time > timeout:
            break
        
        found = False
        # Increase restarts for size 10
        num_restarts = 200 if size == 10 else 50
        
        for restart in range(num_restarts):
            if time.time() - start_time > timeout:
                break
            
            # Heuristic Init: Start with a spread-out set
            # Try to include 1 and 39 to maximize range coverage potential
            S = [1, 39]
            candidates = list(range(2, 39))
            random.shuffle(candidates)
            S.extend(candidates[:size-2])
            S = list(set(S))
            # Ensure size is correct if set() reduced it (unlikely with unique range)
            while len(S) < size:
                S.append(random.randint(1, 39))
            S = list(set(S))
            if len(S) != size:
                # Rare case, just re-sample
                S = random.sample(range(1, 40), size)
            
            current_cov = coverage(S)
            
            # Simulated Annealing
            T = 1.0
            cooling = 0.99
            min_T = 0.01
            
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
                
                delta = new_cov - current_cov
                
                if delta >= 0:
                    current_cov = new_cov
                else:
                    # Accept with probability exp(delta / T)
                    if random.random() < pow(2.718281828, delta / T):
                        current_cov = new_cov
                    else:
                        S[idx] = old_val
                
                T *= cooling
                if T < min_T:
                    T = min_T
        
        if found:
            break
    
    if best_S is None:
        # Fallback: size 12 or just a valid larger set
        # This should rarely happen
        S = random.sample(range(1, 40), 12)
        # Quick local search to make it valid
        for _ in range(10000):
            if is_valid(S):
                break
            idx = random.randint(0, 11)
            old_val = S[idx]
            new_val = random.randint(1, 39)
            if new_val in S and new_val != old_val:
                continue
            S[idx] = new_val
            if coverage(S) >= coverage(S): # placeholder, just accept if not worse? 
                 # Actually just accept if it improves or equal
                 pass
        # Ensure validity for fallback
        if not is_valid(S):
            # Use a known valid structure if possible, or just raise
            # For now, let's assume we found something or use a dense fallback
            # Dense fallback: all spokes
            S = list(range(1, 40))
        
        best_S = S
        best_size = len(S)

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
