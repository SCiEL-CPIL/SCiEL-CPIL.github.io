import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    
    # Bitmask: sum s represented as bit (s-2). Sums range from 2 to 568.
    def compute_sums_bitmask(arr):
        mask = 0
        n = len(arr)
        for i in range(n):
            a = arr[i]
            for j in range(i, n):
                s = a + arr[j]
                mask |= (1 << (s - 2))
        return mask

    def can_add_bitmask(candidate, arr, mask):
        for y in arr:
            s = candidate + y
            if (mask >> (s - 2)) & 1:
                return False
        s = 2 * candidate
        if (mask >> (s - 2)) & 1:
            return False
        return True

    def add_to_bitmask(candidate, arr, mask):
        new_mask = mask
        for y in arr:
            s = candidate + y
            new_mask |= (1 << (s - 2))
        s = 2 * candidate
        new_mask |= (1 << (s - 2))
        return new_mask

    base_arr = sorted(prescribed)
    base_mask = compute_sums_bitmask(base_arr)
    
    best_set = base_arr[:]
    best_mask = base_mask
    best_size = len(best_set)

    rng = random.Random(42)
    
    def greedy_fill(arr, mask, time_limit):
        local_arr = arr[:]
        local_mask = mask
        candidates = list(range(1, 285))
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > time_limit:
                break
            if x in local_arr:
                continue
            if can_add_bitmask(x, local_arr, local_mask):
                local_arr.append(x)
                local_mask = add_to_bitmask(x, local_arr[:-1], local_mask)
        return local_arr, local_mask

    # Initial greedy fill
    greedy_arr, greedy_mask = greedy_fill(base_arr, base_mask, deadline)
    if len(greedy_arr) > best_size:
        best_size = len(greedy_arr)
        best_set = greedy_arr[:]
        best_mask = greedy_mask

    # Hybrid Search: Local Search + Kicks
    current_set = best_set[:]
    current_mask = best_mask
    
    T = 1.0
    T_min = 0.01
    
    # We will run this loop until deadline
    while time.time() < deadline:
        if best_size >= 20:
            break
            
        # Local Search Phase (Simulated Annealing)
        # Run for a fixed number of steps or short time
        steps = 1000
        for _ in range(steps):
            if time.time() > deadline:
                break
                
            if len(current_set) >= 20:
                break
                
            move_type = rng.random()
            
            if move_type < 0.4:
                # Add
                x = rng.randint(1, 284)
                if x not in current_set:
                    if can_add_bitmask(x, current_set, current_mask):
                        new_set = current_set + [x]
                        new_mask = add_to_bitmask(x, current_set, current_mask)
                        current_set = new_set
                        current_mask = new_mask
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = current_set[:]
                            best_mask = current_mask
            elif move_type < 0.8:
                # Remove
                removable = [x for x in current_set if x not in prescribed_set]
                if removable:
                    x = rng.choice(removable)
                    new_set = [y for y in current_set if y != x]
                    new_mask = compute_sums_bitmask(new_set)
                    # Accept with prob exp(-1/T)
                    if rng.random() < math.exp(-1.0 / T):
                        current_set = new_set
                        current_mask = new_mask
            else:
                # Swap
                removable = [x for x in current_set if x not in prescribed_set]
                if removable:
                    x_remove = rng.choice(removable)
                    temp_set = [y for y in current_set if y != x_remove]
                    temp_mask = compute_sums_bitmask(temp_set)
                    x_add = rng.randint(1, 284)
                    if x_add not in temp_set:
                        if can_add_bitmask(x_add, temp_set, temp_mask):
                            new_set = temp_set + [x_add]
                            new_mask = add_to_bitmask(x_add, temp_set, temp_mask)
                            # Accept same size swaps to explore
                            current_set = new_set
                            current_mask = new_mask
                            if len(current_set) > best_size:
                                best_size = len(current_set)
                                best_set = current_set[:]
                                best_mask = current_mask
            
            T *= 0.99
            if T < T_min:
                T = 1.0

        # If we are stuck (no improvement in last local search phase), Kick
        # Simple check: if current_size == best_size and we haven't improved in a while
        # For simplicity, just kick every few iterations of the outer loop?
        # Let's kick if we haven't improved best_size in the last local search phase.
        # We don't track that easily. Let's just kick periodically.
        
        # Kick: Remove 2 random non-prescribed elements and greedy fill
        removable = [x for x in best_set if x not in prescribed_set]
        if removable:
            num_remove = min(2, len(removable))
            to_remove = rng.sample(removable, num_remove)
            start_set = [x for x in best_set if x not in to_remove]
            start_mask = compute_sums_bitmask(start_set)
            
            new_arr, new_mask = greedy_fill(start_set, start_mask, deadline)
            
            if len(new_arr) > best_size:
                best_size = len(new_arr)
                best_set = new_arr[:]
                best_mask = new_mask
                current_set = best_set[:]
                current_mask = best_mask
                if best_size >= 20:
                    break

    # Final validation
    final_sums = set()
    lst = sorted(best_set)
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            final_sums.add(lst[i] + lst[j])
    
    expected = len(lst) * (len(lst) + 1) // 2
    if len(final_sums) != expected:
        return sorted(prescribed)
        
    return lst
