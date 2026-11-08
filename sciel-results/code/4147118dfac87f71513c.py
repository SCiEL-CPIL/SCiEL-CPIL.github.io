import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def score_set(U):
        if not U or 0 not in U:
            return 0.0
        U = sorted(set(U))
        n = len(U)
        max_u = U[-1]
        if max_u == 0:
            return 0.0
        
        # Fast sumset and diffset calculation using bitsets
        # Bit i represents presence of i in the set
        # We need to handle negative differences.
        # Strategy:
        # Sumset: 
        # mask = sum(1 << u for u in U)
        # sum_mask = 0
        # for u in U: sum_mask |= (mask << u)
        # P = sum_mask.bit_count()
        
        # Diffset:
        # We compute non-negative differences.
        # diff_pos_mask = 0
        # for i, a in enumerate(U):
        #    for j, b in enumerate(U):
        #        if a >= b:
        #             diff_pos_mask |= (1 << (a-b))
        # D = 2 * diff_pos_mask.bit_count() - 1
            
        # This is O(N^2) shifts/ORs.
        # For N=20, 400 ops. Very fast.
        
        mask = 0
        for u in U:
            mask |= (1 << u)
            
        sum_mask = 0
        for u in U:
            sum_mask |= (mask << u)
        P = sum_mask.bit_count()
        
        if P == 0:
            return 0.0
            
        diff_pos_mask = 0
        # Optimization: only need a >= b
        for i in range(n):
            a = U[i]
            for j in range(i + 1):
                b = U[j]
                diff_pos_mask |= (1 << (a - b))
                
        D_pos = diff_pos_mask.bit_count()
        D = 2 * D_pos - 1
        
        ratio = D / P
        if ratio <= 1:
            return 0.0
            
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 0.0
            
        return 1 + math.log(ratio) / denom

    def local_search(U, max_iters=20000):
        U = list(set(U))
        U.sort()
        if 0 not in U:
            U.append(0)
            U.sort()
            
        current_score = score_set(U)
        best_score = current_score
        best_U = list(U)
        
        # Temperature for simulated annealing
        T = 0.05
        T_min = 0.001
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Cooldown
            T = max(T_min, T * 0.99)
            
            new_U = list(U)
            move_type = random.random()
            
            if move_type < 0.3 and len(U) < 25:
                # Add
                max_val = max(U)
                # Try adding near existing elements or extending the range
                if random.random() < 0.5:
                    base = random.choice(U)
                    new_val = base + random.randint(1, max(1, max_val // 10))
                else:
                    new_val = random.randint(1, max_val + max(1, max_val // 5))
                
                if new_val not in new_U:
                    new_U.append(new_val)
            elif move_type < 0.5 and len(U) > 2:
                # Remove
                removable = [x for x in new_U if x != 0]
                if removable:
                    idx = random.choice(removable)
                    new_U.remove(idx)
            else:
                # Replace
                if len(U) > 1:
                    removable = [x for x in new_U if x != 0]
                    if removable:
                        old_val = random.choice(removable)
                        max_val = max(U)
                        # Perturb old value
                        if random.random() < 0.5:
                            new_val = old_val + random.randint(-5, 5)
                        else:
                            new_val = random.randint(1, max_val + 10)
                        
                        if new_val >= 0 and new_val not in new_U:
                            new_U.remove(old_val)
                            new_U.append(new_val)
            
            new_U = sorted(set(new_U))
            if 0 not in new_U:
                new_U.append(0)
                new_U.sort()
                
            new_score = score_set(new_U)
            
            if new_score > current_score:
                current_score = new_score
                U = new_U
                if current_score > best_score:
                    best_score = current_score
                    best_U = list(U)
            else:
                # Accept with probability
                if current_score > 0 and T > 0:
                    delta = new_score - current_score
                    if delta < 0:
                        # Avoid overflow/underflow
                        if delta > -10:
                            accept_prob = math.exp(delta / T)
                        else:
                            accept_prob = 0
                        if random.random() < accept_prob:
                            current_score = new_score
                            U = new_U

        return best_U, best_score

    # Initial candidates
    candidates = []
    
    # Base 3 geometric progression
    for k in range(4, 15):
        U = [3**i for i in range(k)]
        candidates.append(U)
        
    # Base 2 geometric progression
    for k in range(4, 15):
        U = [2**i for i in range(k)]
        candidates.append(U)
        
    # Base 4
    for k in range(4, 12):
        U = [4**i for i in range(k)]
        candidates.append(U)
        
    # Perturbed Base 3
    base = [3**i for i in range(6)]
    for _ in range(10):
        U = list(base)
        # Randomly add or modify
        if random.random() < 0.5 and len(U) < 10:
            U.append(random.randint(1, max(U) + 10))
        else:
            idx = random.randint(1, len(U)-1)
            U[idx] = U[idx] + random.randint(-2, 2)
        U = sorted(set(x for x in U if x >= 0))
        if 0 in U:
            candidates.append(U)
            
    # Random sets with increasing gaps
    for _ in range(20):
        size = random.randint(5, 15)
        U = [0]
        current = 1
        for i in range(size - 1):
            gap = random.randint(1, 5)
            current += gap
            U.append(current)
        candidates.append(U)

    best_U = None
    best_score = 0
    
    for cand in candidates:
        if time.time() > deadline:
            break
        refined_U, refined_score = local_search(cand, max_iters=5000)
        if refined_score > best_score:
            best_score = refined_score
            best_U = refined_U
    
    # Final polish on best
    if best_U is not None:
        final_U, final_score = local_search(best_U, max_iters=10000)
        if final_score > best_score:
            best_score = final_score
            best_U = final_U
    
    if best_U is None:
        best_U = [0, 1, 3, 9]
        
    return sorted(set(best_U))

# EVOLVE-BLOCK-END
