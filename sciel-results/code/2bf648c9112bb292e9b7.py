import time
import random
import math

def compute_metrics(A):
    """Compute max(|A+A|, |A*A|) for a set A."""
    n = len(A)
    sum_set = set()
    prod_set = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sum_set.add(ai + aj)
            prod_set.add(ai * aj)
    return max(len(sum_set), len(prod_set))

def generate_pool(limit=10000):
    """Generate a pool of smooth numbers and powers of 2."""
    pool = set()
    # Powers of 2
    p = 1
    while p <= limit:
        pool.add(p)
        p *= 2
    # 2,3-smooth
    p2 = 1
    while p2 <= limit:
        p3 = p2
        while p3 <= limit:
            pool.add(p3)
            p3 *= 3
        p2 *= 2
    # 2,3,5-smooth
    p2 = 1
    while p2 <= limit:
        p3 = p2
        while p3 <= limit:
            p5 = p3
            while p5 <= limit:
                pool.add(p5)
                p5 *= 5
            p3 *= 3
        p2 *= 2
    return sorted(list(pool))

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    pool = generate_pool(10000)
    pool_size = len(pool)
    
    # Initial set: Greedy
    # Start with 1, then add best candidates
    # To speed up, start with a reasonable guess: Powers of 2 + small smooths
    init_A = []
    p = 1
    while len(init_A) < 14 and p <= 10000:
        init_A.append(p)
        p *= 2
    # Add smallest remaining smooths
    for x in pool:
        if len(init_A) >= 20:
            break
        if x not in init_A:
            init_A.append(x)
    
    init_A = list(set(init_A))
    if len(init_A) < 20:
        # Fill with random small numbers if needed (should not happen)
        for x in range(1, 10001):
            if len(init_A) >= 20:
                break
            if x not in init_A:
                init_A.append(x)
    
    current_A = sorted(init_A)
    current_score = compute_metrics(current_A)
    best_A = current_A[:]
    best_score = current_score
    
    random.seed(42)
    
    # Local Search
    # We will try to swap each element with a random subset of the pool
    # To ensure diversity, we'll do multiple passes
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        improved = False
        # Shuffle pool to avoid bias
        random.shuffle(pool)
        
        for i in range(20):
            if time.time() > deadline:
                break
            old_val = current_A[i]
            # Try swapping with a sample of the pool
            # Use a fixed sample size to control time
            sample_size = min(50, pool_size)
            candidates = random.sample(pool, sample_size)
            
            for new_val in candidates:
                if new_val in current_A:
                    continue
                
                trial_A = current_A[:]
                trial_A[i] = new_val
                trial_A.sort()
                
                # Validate
                if len(set(trial_A)) != 20 or any(x < 1 or x > 10000 for x in trial_A):
                    continue
                
                trial_score = compute_metrics(trial_A)
                
                if trial_score < current_score:
                    current_A = trial_A
                    current_score = trial_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_A = current_A[:]
                    break # Restart search for this i with new set
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Random restart or perturbation
                # Perturb best_A
                if random.random() < 0.5:
                    # Swap 2 random elements
                    i, j = random.sample(range(20), 2)
                    new_val = random.choice(pool)
                    if new_val not in best_A:
                        trial_A = best_A[:]
                        trial_A[i] = new_val
                        trial_A.sort()
                        if len(set(trial_A)) == 20 and all(1 <= x <= 10000 for x in trial_A):
                            trial_score = compute_metrics(trial_A)
                            if trial_score < best_score * 1.1: # Allow slight worsening to escape
                                current_A = trial_A
                                current_score = trial_score
                else:
                    # Reset to best and try different random seed for pool shuffle
                    current_A = best_A[:]
                    current_score = best_score
                no_improve_count = 0
        else:
            no_improve_count = 0
            
    return best_A
