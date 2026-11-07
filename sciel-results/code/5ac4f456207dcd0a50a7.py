import random
import time

def construct():
    n = 128
    
    # 1. Precompute bad sets and indices
    # A bad set is a tuple of 6 values (a, b, c, a+b, a+c, b+c)
    # We need to ensure no such tuple is fully contained in S.
    bad_sets = []
    bad_sets_containing = [[] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                s_ab = a + b
                s_ac = a + c
                s_bc = b + c
                if s_bc > n:
                    break
                
                vals = (a, b, c, s_ab, s_ac, s_bc)
                bad_sets.append(vals)
                for v in vals:
                    bad_sets_containing[v].append(vals)

    # 2. Helper function to check if adding 'val' to current set is valid
    def is_valid_add(in_set, val):
        # Check if any bad set containing 'val' becomes fully contained in S
        # by adding 'val'.
        # This happens if all OTHER members of that bad set are already in S.
        for bset in bad_sets_containing[val]:
            all_others_present = True
            for v in bset:
                if v == val:
                    continue
                if not in_set[v]:
                    all_others_present = False
                    break
            if all_others_present:
                return False
        return True

    # 3. Local Search Routine
    def run_local_search(in_set, S_size):
        improved = True
        while improved:
            improved = False
            
            # Phase A: Try to add any single element
            for v in range(1, n + 1):
                if not in_set[v]:
                    if is_valid_add(in_set, v):
                        in_set[v] = True
                        S_size += 1
                        improved = True
            
            # Phase B: Try to remove 1 and add 2 (net gain +1)
            found_move = False
            remove_candidates = [u for u in range(1, n + 1) if in_set[u]]
            random.shuffle(remove_candidates)
            
            for u in remove_candidates:
                if not in_set[u]:
                    continue
                
                # Tentatively remove u
                in_set[u] = False
                S_size -= 1
                
                added = []
                candidates = [v for v in range(1, n + 1) if not in_set[v]]
                random.shuffle(candidates)
                
                for v in candidates:
                    if is_valid_add(in_set, v):
                        in_set[v] = True
                        S_size += 1
                        added.append(v)
                        if len(added) == 2:
                            found_move = True
                            improved = True
                            break
                
                if not found_move:
                    # Revert changes
                    in_set[u] = True
                    S_size += 1
                    for v in added:
                        in_set[v] = False
                        S_size -= 1
                else:
                    # Move accepted, break out of u-loop
                    break
        
        return S_size

    # 4. Main Loop with Restarts
    best_size = 0
    best_set = []
    
    start_time = time.time()
    time_limit = 55.0 # Leave buffer for return and overhead
    
    while time.time() - start_time < time_limit:
        in_set = [False] * (n + 1)
        S_size = 0
        
        # Greedy Initialization with random order
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        for v in order:
            if is_valid_add(in_set, v):
                in_set[v] = True
                S_size += 1
        
        # Local Search
        S_size = run_local_search(in_set, S_size)
        
        if S_size > best_size:
            best_size = S_size
            best_set = [i for i in range(1, n + 1) if in_set[i]]
            
    return best_set
