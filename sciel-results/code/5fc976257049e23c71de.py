import time
import random
import math

N = 256

def construct():
    # Precomputation of 3-APs and 4-APs involving each element
    
    # three_aps[x] = list of (y, z) such that {x, y, z} is a 3-AP
    three_aps = [[] for _ in range(N + 1)]
    
    # four_aps[x] = list of (y, z, w) such that {x, y, z, w} is a 4-AP
    four_aps = [[] for _ in range(N + 1)]
    
    # Generate all 3-APs and 4-APs
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            three_aps[a].append((b, c))
            three_aps[b].append((a, c))
            three_aps[c].append((a, b))
            
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            four_aps[a].append((b, c, e))
            four_aps[b].append((a, c, e))
            four_aps[c].append((a, b, e))
            four_aps[e].append((a, b, c))

    def count_3aps_involving(x, A_mask):
        count = 0
        for y, z in three_aps[x]:
            if ((A_mask >> (y - 1)) & 1) and ((A_mask >> (z - 1)) & 1):
                count += 1
        return count

    def can_add(A_mask, x):
        idx = x - 1
        if (A_mask >> idx) & 1:
            return False
        for y, z, w in four_aps[x]:
            if ((A_mask >> (y - 1)) & 1) and ((A_mask >> (z - 1)) & 1) and ((A_mask >> (w - 1)) & 1):
                return False
        return True

    def total_3aps(A_mask):
        total = 0
        for x in range(1, N + 1):
            if (A_mask >> (x - 1)) & 1:
                total += count_3aps_involving(x, A_mask)
        return total // 3

    # Greedy Initialization
    A_mask = 0
    for x in range(1, N + 1):
        if can_add(A_mask, x):
            A_mask |= (1 << (x - 1))
            
    current_score = total_3aps(A_mask)
    best_mask = A_mask
    best_score = current_score
    
    start_time = time.time()
    timeout = 55.0
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.9995
    min_T = 0.01
    
    all_elements = list(range(1, N + 1))
    
    while T > min_T and (time.time() - start_time) < timeout:
        in_set = []
        not_in_set = []
        for x in all_elements:
            if (A_mask >> (x - 1)) & 1:
                in_set.append(x)
            else:
                not_in_set.append(x)
                
        if not in_set or not not_in_set:
            break
            
        x_remove = random.choice(in_set)
        x_add = random.choice(not_in_set)
        
        temp_mask = A_mask & ~(1 << (x_remove - 1))
        
        if not can_add(temp_mask, x_add):
            T *= cooling
            continue
            
        new_mask = temp_mask | (1 << (x_add - 1))
        
        removed_count = count_3aps_involving(x_remove, A_mask)
        added_count = count_3aps_involving(x_add, temp_mask)
        
        new_score = current_score - removed_count + added_count
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            A_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = A_mask
        
        T *= cooling

    # Final Greedy Repair
    for x in range(1, N + 1):
        if not (best_mask >> (x - 1)) & 1:
            if can_add(best_mask, x):
                best_mask |= (1 << (x - 1))
                
    return sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
