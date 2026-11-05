import time
import random

def construct():
    N = 512
    # Precompute all 4-APs
    aps = []
    aps_by_elem = [[] for _ in range(N + 1)]
    
    ap_id = 0
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            aps.append((a, b, c, e))
            aps_by_elem[a].append(ap_id)
            aps_by_elem[b].append(ap_id)
            aps_by_elem[c].append(ap_id)
            aps_by_elem[e].append(ap_id)
            ap_id += 1
            
    num_aps = len(aps)
    
    # State
    in_set = [False] * (N + 1)
    
    def count_aps_if_add(x):
        """
        Returns the number of 4-APs that would be formed if x were added to the current set.
        Assumes x is NOT in the set.
        """
        count = 0
        for ap_idx in aps_by_elem[x]:
            a, b, c, e = aps[ap_idx]
            if x == a:
                if in_set[b] and in_set[c] and in_set[e]:
                    count += 1
            elif x == b:
                if in_set[a] and in_set[c] and in_set[e]:
                    count += 1
            elif x == c:
                if in_set[a] and in_set[b] and in_set[e]:
                    count += 1
            else: # x == e
                if in_set[a] and in_set[b] and in_set[c]:
                    count += 1
        return count

    def count_aps_if_remove(x):
        """
        Returns the number of 4-APs that would be destroyed if x were removed from the current set.
        Assumes x IS in the set.
        """
        count = 0
        for ap_idx in aps_by_elem[x]:
            a, b, c, e = aps[ap_idx]
            if x == a:
                if in_set[b] and in_set[c] and in_set[e]:
                    count += 1
            elif x == b:
                if in_set[a] and in_set[c] and in_set[e]:
                    count += 1
            elif x == c:
                if in_set[a] and in_set[b] and in_set[e]:
                    count += 1
            else: # x == e
                if in_set[a] and in_set[b] and in_set[c]:
                    count += 1
        return count

    # Greedy Initialization
    # Add elements in order 1..N if valid
    for x in range(1, N + 1):
        if count_aps_if_add(x) == 0:
            in_set[x] = True
            
    current_size = sum(in_set)
    n_aps_active = 0 # Guaranteed 0 by greedy
    
    best_size = current_size
    best_in_set = in_set[:]
    
    start_time = time.time()
    deadline = 85.0
    
    # Simulated Annealing Parameters
    K = 1000
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.95
    
    # We will run until deadline
    # Each iteration:
    # 1. If valid, try to add any valid element (Greedy Polish)
    # 2. If not valid or no add possible, do SA step
    
    while time.time() - start_time < deadline:
        
        # Polish Phase: If we are in a valid state, try to add more elements
        if n_aps_active == 0:
            added_any = False
            # Try adding in random order to break ties/symmetries
            # To avoid O(N^2) overhead of shuffling every time, we can just pick random candidates
            # But N=512 is small.
            candidates = [x for x in range(1, N + 1) if not in_set[x]]
            random.shuffle(candidates)
            
            for x in candidates:
                if count_aps_if_add(x) == 0:
                    in_set[x] = True
                    current_size += 1
                    added_any = True
                    # n_aps_active remains 0
                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set[:]
            
            if added_any:
                continue # Keep polishing
        
        # SA Phase
        # If we are here, we either have APs or we are stuck in a local optimum of valid sets
        
        # Cool down
        T = max(T_min, T * cooling_rate)
        
        # Do a few SA steps
        # Number of steps can be adaptive, but fixed is fine
        steps = 100
        for _ in range(steps):
            if time.time() - start_time > deadline:
                break
            
            x = random.randint(1, N)
            
            if in_set[x]:
                # Remove x
                k = count_aps_if_remove(x)
                delta_n_aps = -k
                delta_size = -1
                delta_cost = K * delta_n_aps - delta_size
            else:
                # Add x
                k = count_aps_if_add(x)
                delta_n_aps = k
                delta_size = 1
                delta_cost = K * delta_n_aps - delta_size
            
            # Acceptance Criteria
            if delta_cost < 0:
                accept = True
            else:
                if T > 0:
                    # P = exp(-delta_cost / T)
                    ratio = delta_cost / T
                    if ratio > 50:
                        accept = False
                    else:
                        accept = random.random() < (2.71828 ** (-ratio))
                else:
                    accept = False
            
            if accept:
                in_set[x] = not in_set[x]
                n_aps_active += delta_n_aps
                current_size += delta_size
                
                # Update best if valid
                if n_aps_active == 0:
                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set[:]
                        
        # If we are stuck in a valid local optimum for a long time, 
        # we might want to reset T to escape?
        # The cooling rate handles this naturally by lowering T.
        # But if T is too low, we can't accept the +1 cost of removal.
        # So we should ensure T is high enough to allow removals when stuck.
        # If we are in a valid state and no adds are possible, we are stuck.
        # We need to accept a removal (Cost +1).
        # So T should be > 1.
        # If T < 1, we might never escape.
        # So if n_aps_active == 0 and no adds possible, force T to be at least 1.0?
        if n_aps_active == 0:
            if T < 1.0:
                T = 1.0

    return sorted([i for i in range(1, N + 1) if best_in_set[i]])
