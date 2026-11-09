import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    def evaluate(A):
        """
        Compute max(|A+A|, |A*A|) efficiently using NumPy.
        A is a list or array of 20 distinct positive integers.
        """
        A_arr = np.array(A, dtype=np.int64)
        # Compute all pairwise sums
        sums = A_arr[:, None] + A_arr[None, :]
        # Compute all pairwise products
        prods = A_arr[:, None] * A_arr[None, :]
        
        # Count unique values
        # Using np.unique is efficient for these small matrices (20x20)
        sum_size = len(np.unique(sums))
        prod_size = len(np.unique(prods))
        
        return max(sum_size, prod_size)

    def get_divisors(n):
        divs = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
            i += 1
        return sorted(divs)

    best_set = None
    best_score = float('inf')
    
    # 1. Generate initial candidates
    candidates = []
    
    # Strategy 1: Divisors of highly composite numbers
    # These sets have many product collisions (a*b = c*d)
    highly_composite_nums = [
        2520, 5040, 10080, 15120, 20160, 27720, 45360, 50400, 55440, 83160, 
        110880, 166320, 221760, 277200, 332640, 498960, 554400, 665280, 720720, 1081080
    ]
    
    for N in highly_composite_nums:
        divs = [d for d in get_divisors(N) if d <= 10000]
        if len(divs) >= 20:
            # Try the smallest 20
            candidates.append(divs[:20])
            # Try a mix of small and large divisors to keep values spread out but structured
            # Actually, for product collisions, staying within the divisor lattice is key.
            # Let's just try the first 20 for now, and maybe some random subsets later.
            
    # Strategy 2: Arithmetic Progressions
    # Small difference 'd' leads to smaller sumset size (approx 2n-1)
    # Product set size depends on 'a' and 'd'.
    for a in range(1, 50):
        for d in range(1, 10):
            A_ap = [a + i * d for i in range(20)]
            if max(A_ap) <= 10000:
                candidates.append(A_ap)
                
    # Strategy 3: Geometric Progressions (Powers of small bases)
    # Powers of 2: Sumset is large (all sums distinct), Product set is small (powers of 2).
    # Powers of 3: Similar.
    # We want to balance. Maybe a mix?
    # Let's just add pure powers for now.
    for base in [2, 3, 4, 5, 6]:
        A_geo = []
        val = 1
        for i in range(20):
            A_geo.append(val)
            val *= base
        if max(A_geo) <= 10000:
            candidates.append(A_geo)
            
    # Strategy 4: Random sets with small numbers
    random.seed(42)
    for _ in range(50):
        A_rand = sorted(random.sample(range(1, 100), 20))
        candidates.append(A_rand)
        
    # Strategy 5: "Smooth" numbers (5-smooth, 7-smooth)
    # Numbers of the form 2^a 3^b 5^c
    smooth_candidates = set()
    for a in range(0, 15):
        for b in range(0, 10):
            for c in range(0, 8):
                val = (2**a) * (3**b) * (5**c)
                if val <= 10000:
                    smooth_candidates.add(val)
    smooth_list = sorted(smooth_candidates)
    if len(smooth_list) >= 20:
        candidates.append(smooth_list[:20])
        # Also try some random subsets of smooth numbers
        for _ in range(20):
            candidates.append(sorted(random.sample(smooth_list, 20)))

    # Evaluate all initial candidates
    for A in candidates:
        if len(set(A)) != 20:
            continue
        score = evaluate(A)
        if score < best_score:
            best_score = score
            best_set = A[:]
            
    if time.time() > deadline:
        return best_set

    # 2. Local Search (Hill Climbing)
    # We will perform random restarts.
    # In each restart, we take the best set found so far (or a random candidate) and try to improve it.
    
    num_restarts = 10
    steps_per_restart = 0 # Will be determined by time
    
    # Estimate time per step to set steps_per_restart
    # Just do a few dummy steps
    t0 = time.time()
    dummy_A = best_set[:]
    for _ in range(100):
        i = random.randint(0, 19)
        old_val = dummy_A[i]
        dummy_A[i] = random.randint(1, 10000)
        if dummy_A[i] not in dummy_A:
            dummy_A.sort()
            evaluate(dummy_A)
            dummy_A[i] = old_val
            dummy_A.sort()
    t1 = time.time()
    avg_step_time = (t1 - t0) / 100.0
    
    # Target: use 80% of remaining time for search
    time_budget = deadline - time.time()
    search_time = time_budget * 0.8
    
    # We will loop until time runs out, rather than fixed steps, to be safe.
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Start from best found so far, or a random candidate for diversity
        if r == 0:
            current = best_set[:]
        else:
            # Pick a random candidate from our initial list or generate a new one
            if random.random() < 0.5:
                current = best_set[:]
            else:
                current = candidates[random.randint(0, len(candidates)-1)][:]
                # Ensure validity
                if len(set(current)) != 20:
                    current = best_set[:]
                    
        current_score = evaluate(current)
        
        # Local search loop
        # We try to replace one element with a random value.
        # To be smarter, we can bias the replacement towards values that are "close" 
        # to existing values or divisors/multiples, but random is a good baseline.
        
        # We can also try "perturbation": replace 2 elements at once.
        
        while time.time() < deadline - 1.0: # 1s safety margin
            # Choose move type
            # 80% single replace, 20% double replace
            if random.random() < 0.8:
                i = random.randint(0, 19)
                old_val = current[i]
                # Generate new value
                # Bias towards small numbers? Or random?
                # Let's try random 1..10000 first.
                new_val = random.randint(1, 10000)
                
                if new_val in current:
                    continue
                
                current[i] = new_val
                current.sort()
                new_score = evaluate(current)
                
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_set = current[:]
                else:
                    # Revert
                    current[i] = old_val
                    current.sort()
            else:
                # Double replace
                i, j = random.sample(range(20), 2)
                old_i, old_j = current[i], current[j]
                new_i = random.randint(1, 10000)
                new_j = random.randint(1, 10000)
                
                # Check validity
                if new_i in current or new_j in current or new_i == new_j:
                    continue
                    
                current[i] = new_i
                current[j] = new_j
                current.sort()
                new_score = evaluate(current)
                
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_set = current[:]
                else:
                    current[i] = old_i
                    current[j] = old_j
                    current.sort()

    return best_set

# EVOLVE-BLOCK-END
