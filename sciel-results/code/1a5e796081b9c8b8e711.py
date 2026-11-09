import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s kill
    
    n = 512
    
    # Precompute all 4-APs
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(ap_list)
    
    # Index: element -> list of AP indices
    element_to_aps = [[] for _ in range(n + 1)]
    for idx, ap in enumerate(ap_list):
        for elem in ap:
            element_to_aps[elem].append(idx)
    
    # Helper: Base-4 seed
    def base4_seed():
        result = set()
        for x in range(1, n + 1):
            v = x
            valid = True
            for _ in range(5):  # 4^5 > 512
                if v % 4 > 1:
                    valid = False
                    break
                v //= 4
            if valid:
                result.add(x)
        return result

    best_size = 0
    best_set = set()
    
    # State variables
    in_set = np.zeros(n + 1, dtype=bool)
    ap_counts = np.zeros(num_aps, dtype=int)
    active_aps = 0
    
    def add_elem(x):
        nonlocal active_aps
        if in_set[x]:
            return
        in_set[x] = True
        for ap_idx in element_to_aps[x]:
            ap_counts[ap_idx] += 1
            if ap_counts[ap_idx] == 4:
                active_aps += 1

    def remove_elem(x):
        nonlocal active_aps
        if not in_set[x]:
            return
        in_set[x] = False
        for ap_idx in element_to_aps[x]:
            ap_counts[ap_idx] -= 1
            if ap_counts[ap_idx] == 3:
                active_aps -= 1

    def is_valid(x):
        # Check if adding x creates any 4-AP
        for ap_idx in element_to_aps[x]:
            if ap_counts[ap_idx] == 3:
                return False
        return True

    def get_conflicts(x):
        conflicts = []
        for ap_idx in element_to_aps[x]:
            if ap_counts[ap_idx] == 3:
                conflicts.append(ap_idx)
        return conflicts

    # Number of restarts based on time
    # Each restart takes roughly 1-5 seconds
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Reset state
        in_set[:] = False
        ap_counts[:] = 0
        active_aps = 0
        
        # Initialize with seed
        seed = base4_seed()
        for x in seed:
            add_elem(x)
            
        # Greedy Add
        candidates = list(set(range(1, n + 1)) - seed)
        random.shuffle(candidates)
        for x in candidates:
            if is_valid(x):
                add_elem(x)
                
        # Local Search
        # We try to improve size by swaps or additions
        # Since we did greedy add, there are no free additions.
        # We must use swaps.
        
        # To speed up, we can limit the number of swap attempts per restart
        swap_attempts = 0
        max_swap_attempts = 5000
        
        while swap_attempts < max_swap_attempts:
            if time.time() > deadline - 5.0:
                break
                
            swap_attempts += 1
            
            # Pick random x not in set
            # Use rejection sampling
            x = random.randint(1, n)
            if in_set[x]:
                continue
                
            conflicts = get_conflicts(x)
            
            if not conflicts:
                # Can add directly
                add_elem(x)
                continue
            
            # Find common elements in conflicts
            # conflicts is list of ap_indices
            # We need y such that y is in all APs in conflicts (except x)
            
            # Optimization: If number of conflicts is large, intersection is likely empty
            if len(conflicts) > 5:
                continue
                
            common = None
            for ap_idx in conflicts:
                elems = set(ap_list[ap_idx])
                elems.discard(x)
                if common is None:
                    common = elems
                else:
                    common &= elems
                    
            if common:
                y = random.choice(list(common))
                
                # Perform swap: remove y, add x
                remove_elem(y)
                
                # Check if x is now valid
                if is_valid(x):
                    add_elem(x)
                else:
                    # Revert
                    add_elem(y)
        
        # Update best
        current_size = int(np.sum(in_set))
        if current_size > best_size:
            best_size = current_size
            best_set = set(np.where(in_set)[0])
            best_set.discard(0)
            
    return sorted(best_set)
