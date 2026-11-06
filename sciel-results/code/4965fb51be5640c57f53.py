import time
import random
import math

def compute_metrics(A):
    n = len(A)
    sum_set = set()
    prod_set = set()
    for i in range(n):
        ai = A[i]
        sum_set.add(2 * ai)
        for j in range(i + 1, n):
            aj = A[j]
            sum_set.add(ai + aj)
            prod_set.add(ai * aj)
    # Add diagonal products
    for i in range(n):
        prod_set.add(A[i] * A[i])
    return max(len(sum_set), len(prod_set))

def generate_smooth_numbers(primes, limit):
    nums = {1}
    for p in primes:
        new_nums = set()
        for x in nums:
            v = x
            while v * p <= limit:
                v *= p
                new_nums.add(v)
        nums.update(new_nums)
    return sorted([x for x in nums if x <= limit])

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Generate candidate pool
    # 1. Small integers
    pool = set(range(1, 101))
    # 2. 5-smooth
    pool.update(generate_smooth_numbers([2, 3, 5], 10000))
    # 3. 7-smooth
    pool.update(generate_smooth_numbers([2, 3, 5, 7], 10000))
    # 4. Powers of 2, 3, 5
    for p in [2, 3, 5, 7, 11, 13]:
        v = 1
        while v <= 10000:
            pool.add(v)
            v *= p
    
    pool = sorted([x for x in pool if 1 <= x <= 10000])
    pool_size = len(pool)
    
    # Initial sets
    smooth_23 = generate_smooth_numbers([2, 3], 10000)
    smooth_235 = generate_smooth_numbers([2, 3, 5], 10000)
    smooth_2357 = generate_smooth_numbers([2, 3, 5, 7], 10000)
    initial_sets = [
        smooth_23[:20],
        smooth_235[:20],
        smooth_2357[:20],
        [2**i for i in range(20)],
        list(range(1, 21)),
        sorted(random.sample(pool, 20))
    ]
    
    best_score = float('inf')
    best_A = initial_sets[0]
    
    # Evaluate initial sets
    for s in initial_sets:
        s = sorted(set(s))
        if len(s) == 20:
            score = compute_metrics(s)
            if score < best_score:
                best_score = score
                best_A = s[:]
                
    # Simulated Annealing
    current_A = best_A[:]
    current_score = best_score
    
    random.seed(42)
    T = 50.0
    T_min = 0.1
    cooling = 0.99995
    
    steps_since_improvement = 0
    max_steps_no_improve = 3000
    
    while time.time() < deadline:
        if time.time() >= deadline:
            break
            
        # Check for restart
        if steps_since_improvement > max_steps_no_improve:
            # Restart from best known or smooth-based random
            r = random.random()
            if r < 0.4:
                current_A = best_A[:]
                current_score = best_score
            elif r < 0.7:
                base = smooth_23 if random.random() < 0.5 else smooth_235
                if len(base) >= 20:
                    current_A = sorted(random.sample(base, 20))
                else:
                    current_A = sorted(random.sample(pool, 20))
                current_score = compute_metrics(current_A)
            else:
                current_A = sorted(random.sample(pool, 20))
                current_score = compute_metrics(current_A)
            steps_since_improvement = 0
            T = max(T, 15.0)
            
        # Choose move type: replace (60%) or swap (40%)
        if random.random() < 0.6:
            # Replace one element
            idx = random.randint(0, 19)
            new_val = pool[random.randint(0, pool_size - 1)]
            
            if new_val in current_A:
                T = max(T * cooling, T_min)
                continue
                
            trial_A = current_A[:]
            trial_A[idx] = new_val
            trial_A = sorted(set(trial_A))
            
            if len(trial_A) != 20:
                T = max(T * cooling, T_min)
                continue
        else:
            # Swap two elements with pool values
            i, j = random.sample(range(20), 2)
            new_i = pool[random.randint(0, pool_size - 1)]
            new_j = pool[random.randint(0, pool_size - 1)]
            
            # Ensure new values are distinct and not in current set (except the ones being replaced)
            trial_A = current_A[:]
            removed = {trial_A[i], trial_A[j]}
            if new_i == new_j:
                T = max(T * cooling, T_min)
                continue
            if (new_i in current_A and new_i not in removed) or (new_j in current_A and new_j not in removed):
                T = max(T * cooling, T_min)
                continue
                
            trial_A[i] = new_i
            trial_A[j] = new_j
            trial_A = sorted(set(trial_A))
            
            if len(trial_A) != 20:
                T = max(T * cooling, T_min)
                continue
        
        trial_score = compute_metrics(trial_A)
        delta = trial_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / max(T, 0.01)):
            current_A = trial_A[:]
            current_score = trial_score
            
            if current_score < best_score:
                best_score = current_score
                best_A = current_A[:]
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
        else:
            steps_since_improvement += 1
            
        T = max(T * cooling, T_min)
        
    return best_A
