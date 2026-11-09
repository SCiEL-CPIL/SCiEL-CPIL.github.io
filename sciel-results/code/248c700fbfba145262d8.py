import random
import time
import math

def compute_objective(A):
    A_set = set(A)
    if len(A_set) != len(A):
        return float('inf')
    if any(x < 1 or x > 10000 for x in A):
        return float('inf')
    
    sum_set = set()
    prod_set = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            sum_set.add(A[i] + A[j])
            prod_set.add(A[i] * A[j])
            
    return max(len(sum_set), len(prod_set))

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin
    
    # Initial Candidates
    # 1. Fibonacci-like
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765]
    fib_distinct = list(set(fib))
    fib_distinct.sort()
    # We need 20. fib_distinct has 19 (1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181,6765)
    # Add a candidate to make 20. Let's add 4 to make it 1,2,3,4,5...
    A_fib = fib_distinct.copy()
    A_fib.append(4)
    A_fib.sort()
    
    # 2. Powers of 2 + small numbers
    # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 (14 elements)
    # Add 6 small numbers? 3, 5, 6, 7, 9, 10?
    A_pow2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    A_pow2.extend([3, 5, 6, 7, 9, 10])
    A_pow2.sort()
    
    # 3. Arithmetic Progression 1..20
    A_ap = list(range(1, 21))
    
    # 4. Random
    A_rand = random.sample(range(1, 10001), 20)
    
    candidates = [A_fib, A_pow2, A_ap, A_rand]
    
    best_A = None
    best_score = float('inf')
    
    for A in candidates:
        s = compute_objective(A)
        if s < best_score:
            best_score = s
            best_A = A[:]
            
    # Simulated Annealing
    current_A = best_A[:]
    current_score = best_score
    best_A_sa = current_A[:]
    best_score_sa = current_score
    
    T = 100.0
    T_min = 0.1
    cooling = 0.99
    
    # Candidate pool for swaps: small numbers and numbers in current set
    # We will generate random neighbors
    
    while time.time() < deadline:
        # Generate neighbor
        A_new = current_A[:]
        
        # Strategy: Swap one element with another
        idx = random.randint(0, 19)
        old_val = A_new[idx]
        
        # Propose new value
        # Bias towards small numbers and existing values
        if random.random() < 0.5:
            # Small random number
            new_val = random.randint(1, 100)
        else:
            # Perturb current
            new_val = old_val + random.randint(-50, 50)
        
        if new_val < 1: new_val = 1
        if new_val > 10000: new_val = 10000
        
        if new_val in A_new:
            continue # Skip duplicates
        
        A_new[idx] = new_val
        A_new.sort()
        
        new_score = compute_objective(A_new)
        
        if new_score < current_score:
            current_A = A_new
            current_score = new_score
            if new_score < best_score_sa:
                best_score_sa = new_score
                best_A_sa = current_A[:]
        else:
            # Accept with probability exp(-delta/T)
            delta = new_score - current_score
            if delta < 0:
                prob = 1.0
            else:
                prob = math.exp(-delta / T)
            
            if random.random() < prob:
                current_A = A_new
                current_score = new_score
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Restart with best found?
            if random.random() < 0.1:
                current_A = best_A_sa[:]
                current_score = best_score_sa

    return best_A_sa
