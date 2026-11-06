import time
import random
import math

N = 256
DEADLINE = 55.0

def construct():
    start = time.time()
    random.seed(42)
    
    # Precompute 4-AP conflicts
    # conflicts[x] is a list of bitmasks. If mask & cm == cm, adding x creates a 4-AP.
    conflicts = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        seen = set()
        # x as 1st
        for d in range(1, (N - x) // 3 + 1):
            a, b, c = x + d, x + 2*d, x + 3*d
            cm = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            seen.add(cm)
        # x as 2nd
        for d in range(1, min(x - 1, (N - x) // 2) + 1):
            a, b, c = x - d, x + d, x + 2*d
            cm = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            seen.add(cm)
        # x as 3rd
        for d in range(1, min((x - 1) // 2, N - x) + 1):
            a, b, c = x - 2*d, x - d, x + d
            cm = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            seen.add(cm)
        # x as 4th
        for d in range(1, (x - 1) // 3 + 1):
            a, b, c = x - 3*d, x - 2*d, x - d
            cm = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            seen.add(cm)
        conflicts[x] = list(seen)

    def has_4ap(mask, x):
        for cm in conflicts[x]:
            if (mask & cm) == cm:
                return True
        return False

    def delta_add(mask, x):
        # Returns number of 3-APs created by adding x
        cnt = 0
        # x as 1st
        for d in range(1, (N - x) // 2 + 1):
            if ((mask >> (x + d - 1)) & 1) and ((mask >> (x + 2*d - 1)) & 1):
                cnt += 1
        # x as 2nd
        for d in range(1, min(x - 1, N - x) + 1):
            if ((mask >> (x - d - 1)) & 1) and ((mask >> (x + d - 1)) & 1):
                cnt += 1
        # x as 3rd
        for d in range(1, (x - 1) // 2 + 1):
            if ((mask >> (x - 2*d - 1)) & 1) and ((mask >> (x - d - 1)) & 1):
                cnt += 1
        return cnt

    def delta_remove(mask, x):
        # Returns number of 3-APs destroyed by removing x
        # This is the same as delta_add on mask without x? No.
        # It is the count of 3-APs containing x.
        return delta_add(mask & ~(1 << (x - 1)), x)

    def count_3aps(mask):
        total = 0
        for x in range(1, N + 1):
            if (mask >> (x - 1)) & 1:
                total += delta_remove(mask, x)
        # Each 3-AP is counted 3 times (once for each element)
        return total // 3

    # Greedy Seed
    def greedy_seed():
        mask = 0
        # Random order for diversity
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if not has_4ap(mask, x):
                mask |= (1 << (x - 1))
        return mask

    best_mask = 0
    best_score = -1
    
    # Multiple Restarts
    for restart in range(20):
        if time.time() - start > DEADLINE - 5:
            break
            
        mask = greedy_seed()
        current_score = count_3aps(mask)
        
        if current_score > best_score:
            best_score = current_score
            best_mask = mask
            
        # Simulated Annealing
        T = 5.0
        T_min = 0.1
        cooling_rate = 0.99
        
        # We want to run SA for a fixed amount of time or steps
        # Let's use time-based
        sa_start = time.time()
        sa_duration = min(2.0, (DEADLINE - (time.time() - start)) / 20) # 2s per restart max
        
        while time.time() - sa_start < sa_duration:
            # Choose move type
            r = random.random()
            if r < 0.3:
                # Add
                # Find candidates
                candidates = [x for x in range(1, N + 1) if not ((mask >> (x - 1)) & 1)]
                if not candidates: continue
                x = random.choice(candidates)
                if has_4ap(mask, x): continue
                
                new_mask = mask | (1 << (x - 1))
                delta = delta_add(mask, x)
                new_score = current_score + delta
                
                if new_score > current_score or random.random() < math.exp((new_score - current_score) / T):
                    mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = mask
            elif r < 0.6:
                # Remove
                candidates = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
                if not candidates: continue
                x = random.choice(candidates)
                
                new_mask = mask & ~(1 << (x - 1))
                delta = -delta_remove(mask, x)
                new_score = current_score + delta
                
                if new_score > current_score or random.random() < math.exp((new_score - current_score) / T):
                    mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = mask
            else:
                # Swap
                in_candidates = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
                out_candidates = [x for x in range(1, N + 1) if not ((mask >> (x - 1)) & 1)]
                if not in_candidates or not out_candidates: continue
                
                x = random.choice(in_candidates)
                y = random.choice(out_candidates)
                
                # Check if y can be added after removing x
                temp_mask = mask & ~(1 << (x - 1))
                if has_4ap(temp_mask, y): continue
                
                new_mask = temp_mask | (1 << (y - 1))
                # Delta = (3APs in new_mask) - (3APs in mask)
                # = (3APs in temp_mask + delta_add(temp_mask, y)) - (3APs in mask)
                # 3APs in mask = 3APs in temp_mask + delta_remove(mask, x)
                # So Delta = delta_add(temp_mask, y) - delta_remove(mask, x)
                delta = delta_add(temp_mask, y) - delta_remove(mask, x)
                new_score = current_score + delta
                
                if new_score > current_score or random.random() < math.exp((new_score - current_score) / T):
                    mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = mask
                        
            T *= cooling_rate
            if T < T_min:
                T = T_min

    result = [x for x in range(1, N + 1) if (best_mask >> (x - 1)) & 1]
    return sorted(result)
