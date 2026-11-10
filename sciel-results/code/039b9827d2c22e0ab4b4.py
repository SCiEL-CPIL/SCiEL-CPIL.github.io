import time
import random
import math
from collections import Counter

def construct():
    n = 20
    max_val = 10000
    start_time = time.time()
    deadline = start_time + 85.0
    
    def local_search(A_init, time_limit):
        A = A_init[:]
        A_set = set(A)
        
        sum_counts = Counter()
        prod_counts = Counter()
        for i in range(n):
            for j in range(i, n):
                sum_counts[A[i] + A[j]] += 1
                prod_counts[A[i] * A[j]] += 1
                
        curr_s = len(sum_counts)
        curr_p = len(prod_counts)
        curr_score = max(curr_s, curr_p)
        
        best_score = curr_score
        best_A = A[:]
        
        T = 5.0
        cooling = 0.995
        min_T = 0.01
        
        while time.time() < time_limit:
            i = random.randint(0, n - 1)
            old_val = A[i]
            
            # Generate new val
            # 90% perturb, 10% jump
            if random.random() < 0.1:
                new_val = random.randint(1, max_val)
            else:
                # Step size distribution
                # Smaller steps more likely
                step = random.randint(-5, 5)
                new_val = old_val + step
                if new_val < 1: new_val = 1
                if new_val > max_val: new_val = max_val
                
            if new_val == old_val:
                continue
                
            if new_val in A_set:
                continue
                
            # Save state
            old_sc = sum_counts.copy()
            old_pc = prod_counts.copy()
            
            # Apply move
            for j in range(n):
                s = old_val + A[j]
                p = old_val * A[j]
                sum_counts[s] -= 1
                if sum_counts[s] == 0: del sum_counts[s]
                prod_counts[p] -= 1
                if prod_counts[p] == 0: del prod_counts[p]
                
            A[i] = new_val
            A_set.remove(old_val)
            A_set.add(new_val)
            
            for j in range(n):
                s = new_val + A[j]
                p = new_val * A[j]
                sum_counts[s] += 1
                prod_counts[p] += 1
                
            new_s = len(sum_counts)
            new_p = len(prod_counts)
            new_score = max(new_s, new_p)
            
            if new_score < curr_score:
                curr_score = new_score
                if curr_score < best_score:
                    best_score = curr_score
                    best_A = A[:]
            else:
                delta = new_score - curr_score
                # Avoid overflow/underflow issues, though delta is small
                if delta < 50: 
                    prob = math.exp(-delta / T)
                else:
                    prob = 0.0
                    
                if random.random() < prob:
                    curr_score = new_score
                else:
                    A[i] = old_val
                    A_set.remove(new_val)
                    A_set.add(old_val)
                    sum_counts = old_sc
                    prod_counts = old_pc
                    
            T *= cooling
            if T < min_T:
                T = min_T
                
        return best_A, best_score

    inits = []
    # 1. AP
    inits.append(list(range(1, n + 1)))
    
    # 2. Multiplicative (2^a 3^b)
    nums = set()
    for a in range(15):
        for b in range(15):
            v = (2**a) * (3**b)
            if v <= max_val:
                nums.add(v)
    inits.append(sorted(list(nums))[:n])
    
    # 3. Random
    random.seed(42)
    inits.append(sorted(random.sample(range(1, max_val + 1), n)))
    
    best_overall = None
    best_score_overall = float('inf')
    
    time_per_start = 80.0 / len(inits)
    
    for init in inits:
        t_start = time.time()
        t_limit = t_start + time_per_start
        if time.time() > deadline - 5: break
        
        res_A, res_score = local_search(init, t_limit)
        
        if res_score < best_score_overall:
            best_score_overall = res_score
            best_overall = res_A
            
    return best_overall
