import itertools
import random
import time

def count_sidon_4_subsets(S):
    """
    Count the number of 4-element subsets of S that ARE Sidon.
    We want to minimize this count.
    """
    count = 0
    # Pre-compute list for slight speedup
    S_list = list(S)
    for quad in itertools.combinations(S_list, 4):
        a, b, c, d = quad
        # Check if all 10 sums are unique
        # Sums: 2a, 2b, 2c, 2d, a+b, a+c, a+d, b+c, b+d, c+d
        # Using a set is fast enough for 10 items
        sums = {2*a, 2*b, 2*c, 2*d, a+b, a+c, a+d, b+c, b+d, c+d}
        if len(sums) == 10:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    best_set = None
    best_count = float('inf')
    
    # Initial candidates
    candidates = []
    # Arithmetic Progressions tend to have many non-Sidon 4-subsets (many APs)
    # But wait, an AP of 4 elements {a, a+d, a+2d, a+3d} has sums:
    # 2a, 2a+2d, 2a+4d, 2a+6d, 2a+d, 2a+3d, 2a+5d, 2a+2d, 2a+4d, 2a+5d
    # Duplicates: 2a+2d (from 2b and a+c) and 2a+4d (from 2c and a+d).
    # So APs of 4 are NOT Sidon.
    # So a full AP of 12 will have many non-Sidon 4-subsets.
    # We want to MINIMIZE Sidon 4-subsets. So AP is a good start.
    candidates.append(list(range(12)))
    candidates.append(list(range(0, 128, 10)))
    
    # Random restarts
    for _ in range(20):
        candidates.append(random.sample(range(129), 12))
        
    for c in candidates:
        if len(c) != 12 or len(set(c)) != 12:
            continue
        if min(c) < 0 or max(c) > 128:
            continue
        
        cnt = count_sidon_4_subsets(c)
        if cnt < best_count:
            best_count = cnt
            best_set = c[:]
            
        if best_count == 0:
            return best_set
            
        if time.time() > deadline:
            return best_set

    current = best_set[:]
    current_count = best_count
    
    no_improvement_count = 0
    
    while time.time() < deadline:
        # Pick a random index
        idx = random.randint(0, 11)
        old_val = current[idx]
        
        # Generate a new value
        # Bias towards values close to existing ones to maintain structure?
        # Or just random.
        new_val = random.randint(0, 128)
        
        if new_val in current:
            continue
            
        current[idx] = new_val
        new_count = count_sidon_4_subsets(current)
        
        if new_count < current_count:
            current_count = new_count
            if current_count < best_count:
                best_count = current_count
                best_set = current[:]
            no_improvement_count = 0
            if best_count == 0:
                return best_set
        else:
            # Revert
            current[idx] = old_val
            no_improvement_count += 1
            
            if no_improvement_count > 500:
                # Random restart or swap
                # Try a swap
                i = random.randint(0, 11)
                j = random.randint(0, 11)
                if i != j:
                    current[i], current[j] = current[j], current[i]
                    swap_count = count_sidon_4_subsets(current)
                    if swap_count < current_count:
                        current_count = swap_count
                        if current_count < best_count:
                            best_count = current_count
                            best_set = current[:]
                        no_improvement_count = 0
                    else:
                        current[i], current[j] = current[j], current[i]
                no_improvement_count = 0

    return best_set
