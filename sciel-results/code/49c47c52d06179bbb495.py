import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    N = 1000
    max_sum = 2 * N
    
    # Helper to compute counts and conflicts
    def compute_counts_and_conflicts(S):
        counts = [0] * (max_sum + 1)
        for i in range(len(S)):
            x = S[i]
            counts[2 * x] += 1
            for j in range(i + 1, len(S)):
                y = S[j]
                counts[x + y] += 1
        
        conflicts = 0
        for s in range(2, max_sum + 1):
            if counts[s] > 2:
                conflicts += counts[s] - 2
        return counts, conflicts

    # 1. Greedy Initial Solution
    # We use a randomized greedy to potentially find different local minima
    def greedy_init():
        chosen = []
        counts = [0] * (max_sum + 1)
        
        # Randomize order to break symmetry? 
        # Actually, standard greedy 1..N is often good. 
        # Let's stick to 1..N for determinism in init, but SA will randomize.
        for x in range(1, N + 1):
            valid = True
            # Check if adding x violates constraint
            # x + x
            if counts[2 * x] == 2:
                valid = False
            else:
                for y in chosen:
                    s = x + y
                    if counts[s] == 2:
                        valid = False
                        break
            
            if valid:
                chosen.append(x)
                counts[2 * x] += 1
                for y in chosen[:-1]:
                    counts[x + y] += 1
        return chosen

    # Initial solution
    current_set = greedy_init()
    current_counts, current_conflicts = compute_counts_and_conflicts(current_set)
    
    best_set = list(current_set)
    best_size = len(current_set)
    
    # If initial is valid, it's our baseline
    if current_conflicts == 0:
        pass # best_set is already set

    # Local Search (Simulated Annealing)
    # We will run ILS: short SA runs with perturbations
    
    def sa_run(S, counts, conflicts, time_limit):
        S = list(S)
        counts = list(counts)
        conflicts = conflicts
        
        in_set = [False] * (N + 1)
        for x in S:
            in_set[x] = True
            
        available = [x for x in range(1, N + 1) if not in_set[x]]
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        # Pre-compute neighbors for speed? No, dynamic is fine for N=1000
        
        def calc_delta_add(x):
            delta = 0
            # x + x
            if counts[2 * x] == 2:
                delta += 1
            # x + y
            for y in S:
                s = x + y
                if counts[s] == 2:
                    delta += 1
            return delta

        def calc_delta_remove(x):
            delta = 0
            # Removing x decreases counts.
            # This might fix conflicts.
            # Conflict reduction is the goal.
            # delta_conflicts = sum (max(0, C[s]-2) - max(0, C[s]-1-2)) for affected s
            # If C[s] > 2, removing 1 reduces conflict by 1.
            # If C[s] <= 2, removing 1 does not change conflict (stays 0).
            
            # So delta is - (number of affected sums with count > 2)
            
            affected = set()
            affected.add(2 * x)
            for y in S:
                if y != x:
                    affected.add(x + y)
            
            for s in affected:
                if counts[s] > 2:
                    delta -= 1
            return delta

        def calc_delta_swap(x, y):
            # Remove x, Add y
            # 1. Remove x
            # 2. Add y
            # We need to calculate net change in conflicts.
            
            # Sums affected by removing x:
            rem_sums = set()
            rem_sums.add(2 * x)
            for z in S:
                if z != x:
                    rem_sums.add(x + z)
            
            # Sums affected by adding y (to S \ {x}):
            add_sums = set()
            add_sums.add(2 * y)
            for z in S:
                if z != x:
                    add_sums.add(y + z)
            
            all_sums = rem_sums.union(add_sums)
            
            delta = 0
            for s in all_sums:
                old_c = counts[s]
                new_c = old_c
                if s in rem_sums:
                    new_c -= 1
                if s in add_sums:
                    new_c += 1
                
                old_conf = max(0, old_c - 2)
                new_conf = max(0, new_c - 2)
                delta += new_conf - old_conf
                
            return delta

        while time.time() < time_limit:
            move_type = random.random()
            
            if move_type < 0.3:
                # Add
                if not available:
                    continue
                idx = random.randint(0, len(available) - 1)
                x = available[idx]
                
                delta = calc_delta_add(x)
                
                if delta == 0 or (delta > 0 and random.random() < math.exp(-delta / T)):
                    # Accept
                    in_set[x] = True
                    S.append(x)
                    available.pop(idx)
                    
                    # Update counts
                    counts[2 * x] += 1
                    for y in S[:-1]:
                        counts[x + y] += 1
                        
                    conflicts += delta
                    
            elif move_type < 0.6:
                # Remove
                if not S:
                    continue
                idx = random.randint(0, len(S) - 1)
                x = S[idx]
                
                delta = calc_delta_remove(x)
                
                if delta <= 0 or (delta > 0 and random.random() < math.exp(-delta / T)):
                    # Accept
                    in_set[x] = False
                    S.pop(idx)
                    available.append(x)
                    
                    # Update counts
                    counts[2 * x] -= 1
                    for y in S:
                        counts[x + y] -= 1
                        
                    conflicts += delta
                    
            else:
                # Swap
                if not S or not available:
                    continue
                x_idx = random.randint(0, len(S) - 1)
                x = S[x_idx]
                y_idx = random.randint(0, len(available) - 1)
                y = available[y_idx]
                
                delta = calc_delta_swap(x, y)
                
                if delta <= 0 or (delta > 0 and random.random() < math.exp(-delta / T)):
                    # Accept
                    in_set[x] = False
                    in_set[y] = True
                    S[x_idx] = y
                    available[y_idx] = x
                    
                    # Update counts: Remove x, Add y
                    # Remove x
                    counts[2 * x] -= 1
                    for z in S:
                        if z != y: # z is in S after swap, so z != y is always true for old S elements? 
                                   # Wait, S[x_idx] is now y. 
                                   # We are iterating over S which includes y. 
                                   # We need to subtract x+z for z in S_old \ {x}.
                                   # S_old \ {x} is S_new \ {y}.
                                   # So we subtract x+z for z in S_new where z != y.
                                   # And subtract 2x.
                                   # Then add y+z for z in S_new where z != y.
                                   # And add 2y.
                                   
                                   # Let's just re-calculate affected sums to be safe and simple, 
                                   # or update carefully.
                                   
                                   # Careful update:
                                   # Old S: {x, z1, ..., zk}
                                   # New S: {y, z1, ..., zk}
                                   
                                   # Decrement:
                                   # 2x
                                   # x + z_i for all i
                                   
                                   # Increment:
                                   # 2y
                                   # y + z_i for all i
                                   
                                   # Note: z_i are elements in S_new \ {y}.
                                   
                                   pass # Logic handled below

                    # Let's do the count update explicitly and safely
                    # We already popped x and pushed y into S? No, we replaced in place.
                    # S is now the new set.
                    # We need to subtract contributions of x (from old set)
                    # and add contributions of y (to new set).
                    
                    # Subtract x
                    counts[2 * x] -= 1
                    for z in S:
                        if z != y: # z is in new set. x was in old set. 
                                   # The pairs removed were (x, x) and (x, z) for z in old_set\{x}.
                                   # old_set\{x} == new_set\{y}.
                                   # So we subtract x+z for z in new_set where z != y.
                                   counts[x + z] -= 1
                                   
                    # Add y
                    counts[2 * y] += 1
                    for z in S:
                        if z != y:
                            counts[y + z] += 1
                            
                    conflicts += delta

            T *= cooling
            if T < min_T:
                T = min_T

        return S, counts, conflicts

    # ILS Loop
    while time.time() < deadline:
        # Run SA for 5 seconds
        sa_end = time.time() + 5.0
        if sa_end > deadline:
            sa_end = deadline
            
        current_set, current_counts, current_conflicts = sa_run(
            current_set, current_counts, current_conflicts, sa_end
        )
        
        if current_conflicts == 0:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
                
        # Perturb
        if time.time() >= deadline:
            break
            
        # Perturb: Swap 10% of elements
        num_swaps = max(1, len(current_set) // 10)
        for _ in range(num_swaps):
            if not current_set:
                break
            x_idx = random.randint(0, len(current_set) - 1)
            x = current_set[x_idx]
            
            # Find a y not in current_set
            # To avoid O(N) search, we can just pick random y in 1..N and check
            # But we need to maintain the set structure.
            # For simplicity, we will just remove x and add a random y.
            # This changes size. 
            # Better: Swap x with a random y not in set.
            
            # Find random y not in set
            # Use a set for O(1) lookup
            in_set = set(current_set)
            y = random.randint(1, N)
            while y in in_set:
                y = random.randint(1, N)
                
            current_set[x_idx] = y
            
        # Recompute counts and conflicts for perturbed set
        current_counts, current_conflicts = compute_counts_and_conflicts(current_set)

    return best_set
