import time
import random
import math

def compute_metrics(A):
    """
    Compute max(|A+A|, |A*A|) for a set A.
    A is a list of 20 distinct positive integers.
    """
    n = len(A)
    sum_set = set()
    prod_set = set()
    
    # Optimization: use local variables for speed
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sum_set.add(ai + aj)
            prod_set.add(ai * aj)
            
    return max(len(sum_set), len(prod_set))

def generate_pool(limit=10000):
    """
    Generate a pool of candidate numbers.
    Focus on smooth numbers (products of small primes) and small integers.
    Smooth numbers tend to have smaller sumsets and product sets.
    """
    pool = set()
    
    # 1. Small integers
    for i in range(1, 151):
        pool.add(i)
        
    # 2. Powers of 2, 3, 5, 7
    # 2-smooth
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
        
    # 2,3,5,7-smooth
    p2 = 1
    while p2 <= limit:
        p3 = p2
        while p3 <= limit:
            p5 = p3
            while p5 <= limit:
                p7 = p5
                while p7 <= limit:
                    pool.add(p7)
                    p7 *= 7
                p5 *= 5
            p3 *= 3
        p2 *= 2
        
    # Filter and sort
    pool = [x for x in pool if 1 <= x <= limit]
    pool.sort()
    return pool

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Generate candidate pool
    pool = generate_pool(10000)
    pool_size = len(pool)
    
    # Initial sets
    # 1. First 20 small ints
    init1 = list(range(1, 21))
    # 2. Powers of 2
    init2 = [2**i for i in range(20)]
    # 3. Mix of small smooths
    init3 = pool[:20]
    # 4. Random sample from pool
    random.seed(42)
    init4 = random.sample(pool, 20)
    # 5. Geometric progression with ratio 1.5
    init5 = []
    x = 1
    while len(init5) < 20 and x <= 10000:
        init5.append(x)
        x = int(x * 1.5) + 1
    if len(init5) < 20:
        # Fill with random
        while len(init5) < 20:
            v = random.choice(pool)
            if v not in init5:
                init5.append(v)
                
    initial_sets = [init1, init2, init3, init4, init5]
    
    # Validate and deduplicate initial sets
    valid_initials = []
    for s in initial_sets:
        s = sorted(list(set(s)))
        if len(s) == 20 and all(1 <= x <= 10000 for x in s):
            valid_initials.append(s)
            
    if not valid_initials:
        valid_initials = [list(range(1, 21))]
        
    # Find best initial
    best_score = float('inf')
    best_A = valid_initials[0][:]
    for s in valid_initials:
        score = compute_metrics(s)
        if score < best_score:
            best_score = score
            best_A = s[:]
            
    # Simulated Annealing
    current_A = best_A[:]
    current_score = best_score
    
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    # Pre-compute indices for faster access if needed, but list access is fine
    # We will iterate as long as we have time
    while time.time() < deadline:
        if time.time() >= deadline:
            break
            
        # Choose a random index to replace
        idx = random.randint(0, 19)
        old_val = current_A[idx]
        
        # Choose a new value from the pool
        new_val = pool[random.randint(0, pool_size - 1)]
        
        if new_val == old_val:
            T = max(T * cooling, T_min)
            continue
            
        # Check distinctness
        # Since A is a list, we can check if new_val is in A
        # Optimization: keep a set of current A for O(1) lookup?
        # A has 20 elements, O(20) check is fast.
        if new_val in current_A:
            T = max(T * cooling, T_min)
            continue
            
        # Create trial set
        trial_A = current_A[:]
        trial_A[idx] = new_val
        
        # Evaluate
        trial_score = compute_metrics(trial_A)
        
        delta = trial_score - current_score
        
        # Acceptance criterion
        if delta < 0:
            # Better
            current_A = trial_A
            current_score = trial_score
        else:
            # Worse, accept with probability
            if random.random() < math.exp(-delta / max(T, 1e-9)):
                current_A = trial_A
                current_score = trial_score
                
        # Update best
        if current_score < best_score:
            best_score = current_score
            best_A = current_A[:]
            
        # Cool down
        T = max(T * cooling, T_min)
        
        # Optional: If stuck for a long time (low T), maybe restart from best?
        # For now, simple SA should suffice.
        
    return best_A
