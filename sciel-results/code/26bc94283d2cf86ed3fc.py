import time
import random
import math

def _compute_metrics(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers.
    Uses i<=j to handle the 'equal summands/factors allowed' (diagonal) pairs
    while keeping the set semantics for distinct results."""
    n = len(A)
    sum_set = set()
    prod_set = set()
    for i in range(n):
        a = A[i]
        for j in range(i, n):
            b = A[j]
            sum_set.add(a + b)
            prod_set.add(a * b)
    return max(len(sum_set), len(prod_set))

def _generate_pool():
    """Generate a candidate pool biased towards smooth numbers (for product collisions)
    and small integers (for sum collisions)."""
    pool = set()
    
    # 1. Small integers
    for i in range(1, 200):
        pool.add(i)
        
    # 2. 5-smooth numbers (primes 2, 3, 5) up to 10000
    def is_smooth(n, primes):
        for p in primes:
            while n % p == 0:
                n //= p
        return n == 1
        
    primes_5 = [2, 3, 5]
    for i in range(1, 10001):
        if is_smooth(i, primes_5):
            pool.add(i)
            
    # 3. 7-smooth numbers
    primes_7 = [2, 3, 5, 7]
    for i in range(1, 10001):
        if is_smooth(i, primes_7):
            pool.add(i)
            
    # 4. Powers of small bases
    for b in [2, 3, 4]:
        val = 1
        while val <= 10000:
            pool.add(val)
            val *= b
            
    # 5. Structured multiples (to help with sum structure)
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 18, 24, 30, 36, 48, 60]:
        for k in range(1, 100):
            v = base * k
            if v <= 10000:
                pool.add(v)
                
    pool_list = list(pool)
    random.shuffle(pool_list)
    return pool_list

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    pool_list = _generate_pool()
    
    # Generate diverse initial sets
    # 1. First 20 smooth numbers (very dense in log space, good for products)
    smooths = sorted([x for x in pool_list if x <= 100]) # Small smooths
    if len(smooths) < 20:
        smooths = sorted(pool_list[:100]) # Fallback to small pool
    
    # 2. Pure small integers
    initial_sets = [
        list(range(1, 21)),
        sorted(smooths[:20]),
        sorted(random.sample(pool_list, 20)) if len(pool_list) >= 20 else list(range(1, 21)),
        sorted(random.sample(pool_list, 20)) if len(pool_list) >= 20 else list(range(1, 21)),
        # A mix of small and some larger smooths
        sorted(list(set(list(range(1, 15)) + [2**i for i in range(4, 12)] + [3**i for i in range(4, 8)])))[:20]
    ]
    
    # Ensure valid initial sets
    valid_initials = []
    for init in initial_sets:
        init_set = set(init)
        if len(init_set) == 20 and max(init_set) <= 10000:
            valid_initials.append(sorted(init_set))
            
    if not valid_initials:
        return list(range(1, 21))
        
    best_A = valid_initials[0]
    best_cost = _compute_metrics(best_A)
    
    # Run Simulated Annealing for each initial set, or just the best one if time is tight
    # Given the time constraint, we will run SA on the best initial candidate,
    # and potentially restart if time permits.
    
    # Sort initial candidates by cost to start from the best
    initial_scores = [( _compute_metrics(s), s) for s in valid_initials]
    initial_scores.sort(key=lambda x: x[0])
    
    best_initial_cost, best_initial_set = initial_scores[0]
    best_A = best_initial_set[:]
    best_cost = best_initial_cost
    current_A = best_initial_set[:]
    current_cost = best_initial_cost
    
    # Simulated Annealing parameters
    T_start = 10.0
    T_end = 0.01
    # We don't know exact steps, so we use time-based loop
    # But we need a cooling schedule. Let's estimate steps.
    # Assume ~5000 steps/sec. 50s -> 250k steps.
    # Let's just run until time limit, adjusting T based on elapsed time.
    
    total_time_budget = 45.0 # Leave 10s for final polish
    
    while time.time() < start_time + total_time_budget:
        # Linear cooling based on time? Or fixed steps?
        # Fixed steps is more stable for SA. Let's say we target 100,000 steps.
        # But we don't want to run too many if we are slow.
        # Let's use a hybrid: check time, break if exceeded.
        
        # Cooling: exponential decay
        # We need a step counter to apply cooling.
        # Let's just use a simple loop with a time check inside.
        
        # Move: replace one element
        idx = random.randint(0, 19)
        old_val = current_A[idx]
        
        # Choose new value from pool
        # Bias towards smooth numbers? The pool is already mixed.
        # To avoid duplicates:
        new_val = None
        # Try a few times
        for _ in range(10):
            candidate = random.choice(pool_list)
            if candidate not in current_A:
                new_val = candidate
                break
        
        if new_val is None:
            continue
            
        current_A[idx] = new_val
        new_cost = _compute_metrics(current_A)
        
        # Acceptance probability
        # We need T. Let's calculate T based on time elapsed.
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / total_time_budget)
        T = T_start * (T_end / T_start) ** progress
        
        delta = new_cost - current_cost
        if delta <= 0:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_A = current_A[:]
        else:
            # Accept with probability exp(-delta/T)
            # If T is very small, exp(-delta/T) is ~0
            if random.random() < math.exp(-delta / T):
                current_cost = new_cost
            else:
                current_A[idx] = old_val
        
        # Occasionally do a 2-swap to escape local minima
        if random.random() < 0.05:
            idx1 = random.randint(0, 19)
            idx2 = random.randint(0, 19)
            if idx1 != idx2:
                v1 = current_A[idx1]
                v2 = current_A[idx2]
                
                new_v1 = None
                new_v2 = None
                for _ in range(10):
                    cand1 = random.choice(pool_list)
                    if cand1 not in current_A:
                        new_v1 = cand1
                        break
                if new_v1 is None: continue
                
                for _ in range(10):
                    cand2 = random.choice(pool_list)
                    if cand2 not in current_A and cand2 != new_v1:
                        new_v2 = cand2
                        break
                if new_v2 is None: continue
                
                current_A[idx1] = new_v1
                current_A[idx2] = new_v2
                new_cost = _compute_metrics(current_A)
                
                delta = new_cost - current_cost
                if delta <= 0:
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_A = current_A[:]
                else:
                    if random.random() < math.exp(-delta / T):
                        current_cost = new_cost
                    else:
                        current_A[idx1] = v1
                        current_A[idx2] = v2

    # Final Polish: Local Search
    # Try to improve best_A by single swaps with any pool element
    if time.time() < deadline - 2.0:
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(20):
                old_val = best_A[i]
                # Try a subset of the pool for speed, or the whole pool if fast enough
                # Pool is ~1000-2000 size. 20 * 2000 = 40k evals.
                # Each eval is O(400) ops. 16M ops. Might be slow in Python.
                # Let's restrict to a smaller "refined" pool or just the best 500 candidates.
                # Actually, let's just iterate through the pool but break early if time is tight.
                for new_val in pool_list:
                    if new_val in best_A:
                        continue
                    best_A[i] = new_val
                    s = _compute_metrics(best_A)
                    if s < best_cost:
                        best_cost = s
                        improved = True
                    else:
                        best_A[i] = old_val
                best_A[i] = old_val
                if time.time() > deadline - 1.0:
                    break
    
    return sorted(best_A)
