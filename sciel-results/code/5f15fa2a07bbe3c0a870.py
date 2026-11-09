import random
import time
import math

def construct():
    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    
    # Initialize state
    current_set = list(prescribed)
    current_set_set = set(current_set)
    
    # sum_counts: dict mapping sum -> count
    sum_counts = {}
    
    def update_sum_counts(s, delta):
        if delta == 1:
            sum_counts[s] = sum_counts.get(s, 0) + 1
        else:
            cnt = sum_counts.get(s, 0)
            if cnt == 1:
                del sum_counts[s]
            else:
                sum_counts[s] = cnt - 1

    # Initialize sum_counts
    for i in range(len(current_set)):
        for j in range(i, len(current_set)):
            s = current_set[i] + current_set[j]
            update_sum_counts(s, 1)
            
    def get_duplicates():
        # D = sum(count * (count-1) / 2)
        d = 0
        for c in sum_counts.values():
            if c > 1:
                d += c * (c - 1) // 2
        return d

    current_dupes = get_duplicates()
    best_set = list(current_set)
    best_size = len(current_set)
    
    start_time = time.time()
    time_limit = 115.0
    
    # Simulated Annealing parameters
    T = 1.0
    cooling_rate = 0.9999
    min_T = 1e-5
    
    # We want to maximize size, so we bias towards adding elements
    # But we must keep dupes low.
    
    candidates_pool = list(range(1, 285))
    
    while time.time() - start_time < time_limit:
        # Generate move
        # Probabilistic choice: Add vs Remove
        # If current set is small, add more. If large, maybe remove to fix conflicts.
        # Let's say 80% add, 20% remove if size > prescribed size.
        
        if len(current_set) < 20:
            prob_add = 0.9
        else:
            prob_add = 0.5
            
        is_add = random.random() < prob_add
        
        if is_add:
            # Pick random x not in current_set
            # Optimization: pre-compute list of candidates not in set? 
            # Set size is small (<20), so checking membership is O(1).
            # Random choice from 1..284.
            x = random.randint(1, 284)
            if x in current_set_set:
                continue # Retry or skip? Let's just skip this iteration to save time? 
                         # Better to pick again.
            # Actually, picking from range is efficient enough.
            
            # Calculate delta_dupes for adding x
            delta_d = 0
            new_sums = []
            for y in current_set:
                s = x + y
                c = sum_counts.get(s, 0)
                delta_d += c
                new_sums.append(s)
            s_2x = 2 * x
            c_2x = sum_counts.get(s_2x, 0)
            delta_d += c_2x
            new_sums.append(s_2x)
            
            # Acceptance
            if delta_d <= 0 or random.random() < math.exp(-delta_d / T):
                # Apply move
                current_set.append(x)
                current_set_set.add(x)
                for s in new_sums:
                    update_sum_counts(s, 1)
                current_dupes += delta_d
                
                if current_dupes == 0:
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = list(current_set)
                        if best_size == 20:
                            return best_set
        else:
            # Remove move
            if len(current_set) == len(prescribed):
                continue # Can't remove prescribed
            
            # Pick random non-prescribed element
            non_pres = [x for x in current_set if x not in prescribed_set]
            if not non_pres:
                continue
            x = random.choice(non_pres)
            
            # Calculate delta_dupes for removing x
            delta_d = 0
            old_sums = []
            for y in current_set:
                if y == x:
                    continue
                s = x + y
                c = sum_counts.get(s, 0)
                # Change is 1 - c
                delta_d += 1 - c
                old_sums.append(s)
            s_2x = 2 * x
            c_2x = sum_counts.get(s_2x, 0)
            delta_d += 1 - c_2x
            old_sums.append(s_2x)
            
            # Acceptance
            # Note: delta_d can be negative (improvement) or positive (worsening)
            if delta_d <= 0 or random.random() < math.exp(-delta_d / T):
                # Apply move
                current_set.remove(x)
                current_set_set.remove(x)
                for s in old_sums:
                    update_sum_counts(s, -1)
                current_dupes += delta_d
                
                # If we removed, we might have made it valid? 
                # But we are losing size. Only update best if valid AND size > best.
                # Since we removed, size decreased, so it won't beat best_size unless best_size was invalid?
                # No, best_set is always valid. So removing won't improve best_size.
                # However, it might allow future additions to reach a higher size.
        
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
    return best_set
