import time
import random
import math

def construct():
    start_time = time.time()
    limit = 55.0  # Safety margin for 60s limit
    
    n = 256
    
    # Precompute conflict masks
    # conflict_mask[i] is a bitmask of all j that conflict with i
    conflict_mask = [0] * (n + 1)
    
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            # Check 3-AP conditions
            # 1. w = 2u - v
            w = 2 * u - v
            if 1 <= w <= n and w != u and w != v:
                conflict_mask[u] |= (1 << (v - 1))
                conflict_mask[v] |= (1 << (u - 1))
                continue
            # 2. w = 2v - u
            w = 2 * v - u
            if 1 <= w <= n and w != u and w != v:
                conflict_mask[u] |= (1 << (v - 1))
                conflict_mask[v] |= (1 << (u - 1))
                continue
            # 3. w = (u + v) / 2
            if (u + v) % 2 == 0:
                w = (u + v) // 2
                if 1 <= w <= n and w != u and w != v:
                    conflict_mask[u] |= (1 << (v - 1))
                    conflict_mask[v] |= (1 << (u - 1))

    weights = [0.0] + [1.0 / i for i in range(1, n + 1)]
    
    # Initial Solution: Greedy by weight (smallest indices first)
    current_mask = 0
    current_score = 0.0
    for i in range(1, n + 1):
        bit = 1 << (i - 1)
        if (conflict_mask[i] & current_mask) == 0:
            current_mask |= bit
            current_score += weights[i]
            
    best_mask = current_mask
    best_score = current_score
    
    # Simulated Annealing
    T = 0.1
    T_min = 0.0001
    cooling_rate = 0.9999
    
    all_indices = list(range(1, n + 1))
    
    while time.time() - start_time < limit:
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add a random non-member
            v = random.choice(all_indices)
            bit_v = 1 << (v - 1)
            if not (current_mask & bit_v):
                if (conflict_mask[v] & current_mask) == 0:
                    new_mask = current_mask | bit_v
                    delta = weights[v]
                    current_mask = new_mask
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                        
        elif move_type < 0.7:
            # Try to remove a random member
            found = False
            while not found:
                u = random.choice(all_indices)
                bit_u = 1 << (u - 1)
                if current_mask & bit_u:
                    found = True
                    break
            
            if found:
                new_mask = current_mask ^ bit_u
                delta = -weights[u]
                if random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                        
        else:
            # Try to swap a random member u with a random non-member v
            found_u = False
            while not found_u:
                u = random.choice(all_indices)
                bit_u = 1 << (u - 1)
                if current_mask & bit_u:
                    found_u = True
                    break
            
            found_v = False
            while not found_v:
                v = random.choice(all_indices)
                bit_v = 1 << (v - 1)
                if not (current_mask & bit_v):
                    found_v = True
                    break
            
            if found_u and found_v:
                mask_without_u = current_mask ^ bit_u
                if (conflict_mask[v] & mask_without_u) == 0:
                    new_mask = mask_without_u | bit_v
                    delta = weights[v] - weights[u]
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current_mask = new_mask
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    # Final Polish: Greedy add any possible
    for i in range(1, n + 1):
        bit = 1 << (i - 1)
        if not (best_mask & bit):
            if (conflict_mask[i] & best_mask) == 0:
                best_mask |= bit
                
    # Convert mask to list
    result = []
    for i in range(1, n + 1):
        if best_mask & (1 << (i - 1)):
            result.append(i)
            
    return result

if __name__ == "__main__":
    construct()
