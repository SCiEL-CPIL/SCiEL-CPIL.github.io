import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    N = 1000
    max_sum = 2 * N
    
    # 1. Randomized Greedy Initialization
    best_initial = []
    best_size = 0
    
    # Run multiple randomized greedy attempts
    for _ in range(20):
        if time.time() > deadline - 60: # Leave 60s for SA
            break
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        chosen = []
        counts = [0] * (max_sum + 1)
        
        for x in candidates:
            valid = True
            # Check if x can be added
            # We need to check all y in chosen
            # Optimization: If counts[2*x] == 2, invalid.
            if counts[2 * x] == 2:
                continue
            
            # Check sums with existing elements
            # We can break early if we find a conflict
            can_add = True
            for y in chosen:
                s = x + y
                if counts[s] == 2:
                    can_add = False
                    break
            
            if can_add:
                chosen.append(x)
                counts[2 * x] += 1
                for y in chosen[:-1]:
                    counts[x + y] += 1
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_initial = chosen.copy()
            
    # 2. Local Search (Simulated Annealing)
    current = best_initial
    in_set = [False] * (N + 1)
    for x in current:
        in_set[x] = True
        
    available = [x for x in range(1, N + 1) if not in_set[x]]
    
    # Re-initialize counts
    counts = [0] * (max_sum + 1)
    for i in range(len(current)):
        x = current[i]
        counts[2 * x] += 1
        for j in range(i + 1, len(current)):
            y = current[j]
            counts[x + y] += 1
            
    def get_conflicts(counts):
        c = 0
        for s in range(2, max_sum + 1):
            if counts[s] > 2:
                c += counts[s] - 2
        return c

    current_conflicts = get_conflicts(counts)
    best_conflicts = current_conflicts
    best_solution = list(current)
    best_size_found = len(current)
    
    T = 5.0  # Start with higher temperature
    cooling = 0.999
    min_T = 0.01
    
    # Helper to calculate delta conflicts for a move
    def calc_delta_add(x, current, counts):
        delta = 0
        sums_to_add = []
        # 2x
        s = 2 * x
        if counts[s] == 2:
            delta += 1
        sums_to_add.append(s)
        # x + y
        for y in current:
            s = x + y
            if counts[s] == 2:
                delta += 1
            sums_to_add.append(s)
        return delta, sums_to_add

    def calc_delta_swap(x, y, current, counts):
        # x is in current, y is not
        removed_sums = set()
        removed_sums.add(2 * x)
        for z in current:
            if z != x:
                removed_sums.add(x + z)
                
        added_sums = set()
        added_sums.add(2 * y)
        for z in current:
            if z != x:
                added_sums.add(y + z)
                
        all_sums = removed_sums.union(added_sums)
        
        delta = 0
        for s in all_sums:
            old_c = counts[s]
            new_c = old_c
            if s in removed_sums:
                new_c -= 1
            if s in added_sums:
                new_c += 1
            
            old_conf = max(0, old_c - 2)
            new_conf = max(0, new_c - 2)
            delta += new_conf - old_conf
            
        sums_to_dec = list(removed_sums)
        sums_to_inc = list(added_sums)
        
        return delta, sums_to_dec, sums_to_inc

    while time.time() < deadline:
        if random.random() < 0.4:
            # Add Move
            if not available:
                continue
            idx = random.randint(0, len(available) - 1)
            x = available[idx]
            
            delta, sums_to_add = calc_delta_add(x, current, counts)
            delta_size = 1
            
            # Acceptance
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if delta_size > 0:
                    accept = True
                else:
                    accept = random.random() < 0.5
            else:
                accept = random.random() < math.exp(-delta / T)
                
            if accept:
                in_set[x] = True
                current.append(x)
                available.pop(idx)
                
                for s in sums_to_add:
                    counts[s] += 1
                current_conflicts += delta
                
                if current_conflicts < best_conflicts or (current_conflicts == best_conflicts and len(current) > best_size_found):
                    best_conflicts = current_conflicts
                    if current_conflicts == 0:
                        best_solution = list(current)
                        best_size_found = len(current)
                        
        elif random.random() < 0.3:
            # Remove Move (to escape local minima)
            if not current:
                continue
            x_idx = random.randint(0, len(current) - 1)
            x = current[x_idx]
            
            # Calculate delta for removal
            delta = 0
            sums_to_dec = [2 * x]
            for y in current:
                if y != x:
                    sums_to_dec.append(x + y)
                    
            for s in sums_to_dec:
                old_c = counts[s]
                new_c = old_c - 1
                old_conf = max(0, old_c - 2)
                new_conf = max(0, new_c - 2)
                delta += new_conf - old_conf
                
            delta_size = -1
            
            # Acceptance
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.1 # Rarely remove neutral
            else:
                accept = random.random() < math.exp(-delta / T)
                
            if accept:
                in_set[x] = False
                current.pop(x_idx)
                available.append(x)
                
                for s in sums_to_dec:
                    counts[s] -= 1
                current_conflicts += delta
                
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    if current_conflicts == 0:
                        best_solution = list(current)
                        best_size_found = len(current)
                        
        else:
            # Swap Move
            if not current or not available:
                continue
            x_idx = random.randint(0, len(current) - 1)
            x = current[x_idx]
            y_idx = random.randint(0, len(available) - 1)
            y = available[y_idx]
            
            delta, sums_to_dec, sums_to_inc = calc_delta_swap(x, y, current, counts)
            delta_size = 0
            
            # Acceptance
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.5
            else:
                accept = random.random() < math.exp(-delta / T)
                
            if accept:
                in_set[x] = False
                in_set[y] = True
                current[x_idx] = y
                available[y_idx] = x
                
                for s in sums_to_dec:
                    counts[s] -= 1
                for s in sums_to_inc:
                    counts[s] += 1
                    
                current_conflicts += delta
                
                if current_conflicts < best_conflicts or (current_conflicts == best_conflicts and len(current) > best_size_found):
                    best_conflicts = current_conflicts
                    if current_conflicts == 0:
                        best_solution = list(current)
                        best_size_found = len(current)
                            
        T *= cooling
        if T < min_T:
            T = min_T

    # Final Polish: Greedy Addition
    if best_conflicts == 0:
        # Try to add more elements greedily
        current = best_solution
        in_set = [False] * (N + 1)
        for x in current:
            in_set[x] = True
            
        counts = [0] * (max_sum + 1)
        for i in range(len(current)):
            x = current[i]
            counts[2 * x] += 1
            for j in range(i + 1, len(current)):
                y = current[j]
                counts[x + y] += 1
                
        added = True
        while added and time.time() < deadline - 2:
            added = False
            for x in range(1, N + 1):
                if in_set[x]:
                    continue
                # Check if x can be added
                if counts[2 * x] == 2:
                    continue
                can_add = True
                for y in current:
                    s = x + y
                    if counts[s] == 2:
                        can_add = False
                        break
                if can_add:
                    current.append(x)
                    in_set[x] = True
                    counts[2 * x] += 1
                    for y in current[:-1]:
                        counts[x + y] += 1
                    added = True
                    if time.time() > deadline - 2:
                        break
                    
        best_solution = current
        best_size_found = len(current)

    return best_solution

