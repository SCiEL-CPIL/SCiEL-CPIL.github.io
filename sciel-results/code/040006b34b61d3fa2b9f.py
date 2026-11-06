import time
import random
import numpy as np

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 82  # safety margin before 90s kill

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    A_arr = np.array(A, dtype=np.int64)
    # A+A with equal summands allowed
    sums = A_arr[:, None] + A_arr[None, :]
    # A*A with equal factors allowed
    products = A_arr[:, None] * A_arr[None, :]
    
    # np.unique is faster than set(sums.flatten()) for small arrays in Python
    num_sums = len(np.unique(sums))
    num_products = len(np.unique(products))
    
    return max(num_sums, num_products)

def _is_valid(A):
    """Check if A has 20 distinct positive integers <= 10000."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > MAX_VAL:
            return False
    return True

def _rational_geometric_search(deadline):
    """
    Search for sets of the form { round(a * (p/q)^i) } for i=0..19.
    We iterate over small p, q and optimize a.
    Also explores unions of two geometric progressions.
    """
    best_obj = float('inf')
    best_A = None
    rng = random.Random(42)
    
    max_pq = 20 # Search up to 20/1, 19/1, etc
    
    # Phase 1: Single Geometric Progressions
    for p in range(2, max_pq + 1):
        for q in range(1, p):
            if time.time() > deadline:
                return best_A, best_obj
            
            ratio = p / q
            
            # The sequence is a, a*ratio, a*ratio^2, ...
            # a * ratio^19 <= MAX_VAL
            # a <= MAX_VAL / (ratio^19)
            max_a = int(MAX_VAL / (ratio ** 19))
            if max_a < 1:
                continue
            
            candidates_a = set()
            
            # 1. Boundary
            candidates_a.add(max_a)
            
            # 2. Small integers
            for k in range(1, 10):
                if k <= max_a:
                    candidates_a.add(k)
            
            # 3. Multiples of q^19 to ensure integrality of exact geometric progression
            base = q ** 19
            max_k = max_a // base
            if max_k > 0:
                for k in range(1, min(max_k, 20) + 1):
                    candidates_a.add(k * base)
            
            # 4. Random samples in [1, max_a]
            for _ in range(5):
                candidates_a.add(rng.randint(1, max_a))
            
            # Evaluate candidates
            for a in candidates_a:
                if a < 1:
                    continue
                
                A = []
                valid = True
                current = a
                for i in range(20):
                    if current > MAX_VAL:
                        valid = False
                        break
                    if current in A:
                        valid = False
                        break
                    A.append(current)
                    if i < 19:
                        next_val = (current * p) // q
                        current = next_val
                
                if not valid or len(A) != 20:
                    continue
                
                if len(set(A)) != 20:
                    continue
                
                obj = _objective(A)
                if obj < best_obj:
                    best_obj = obj
                    best_A = list(A)
                
                if time.time() > deadline:
                    return best_A, best_obj

    # Phase 2: Union of Two Geometric Progressions
    # A = { a * r1^i for i in 0..k-1 } U { b * r2^j for j in 0..m-1 }
    # We want to maximize collisions in products.
    # If r1 and r2 are small integers (2, 3), products will be of form 2^x 3^y.
    # This structure minimizes the size of the product set.
    
    # Try splitting 20 elements into two groups of sizes k and 20-k
    # We'll focus on k=10, m=10 or similar balanced splits for now, 
    # or maybe 12/8. Let's try a few splits.
    
    splits = [(10, 10), (12, 8), (8, 12), (15, 5), (5, 15)]
    
    # Try small integer ratios for the two progressions
    # Ratios should be > 1.
    # Using 2 and 3 is a classic strategy for small multiplicative semigroups.
    # We can also try 2 and 2 (which is just one GP), or 3 and 3.
    # But distinct elements are required.
    
    ratios_to_try = [2, 3, 4, 5]
    
    for k, m in splits:
        if time.time() > deadline:
            return best_A, best_A if best_A else None # Return best found so far
        
        for r1 in ratios_to_try:
            for r2 in ratios_to_try:
                if time.time() > deadline:
                    return best_A, best_obj
                
                # We need to choose starting values a and b.
                # To maximize collisions, we might want a and b to be powers of 2 and 3.
                # e.g., a = 2^x, b = 3^y.
                # Let's just search a and b in a small range or specific powers.
                
                # Max possible a: a * r1^(k-1) <= MAX_VAL
                max_a = int(MAX_VAL / (r1 ** (k - 1)))
                # Max possible b: b * r2^(m-1) <= MAX_VAL
                max_b = int(MAX_VAL / (r2 ** (m - 1)))
                
                if max_a < 1 or max_b < 1:
                    continue
                
                # Candidate a's: powers of 2 up to max_a, and some small integers
                cand_a = set()
                power = 1
                while power <= max_a:
                    cand_a.add(power)
                    power *= 2
                # Also add 1, 2, 3... if small
                for i in range(1, min(10, max_a + 1)):
                    cand_a.add(i)
                
                cand_b = set()
                power = 1
                while power <= max_b:
                    cand_b.add(power)
                    power *= 3
                for i in range(1, min(10, max_b + 1)):
                    cand_b.add(i)
                
                # Sample a and b
                # To save time, don't test all combinations if too many.
                # But max_a and max_b are likely small for large k,m.
                # For k=10, r1=2, max_a = 10000/512 = 19. So cand_a is small.
                
                for a in cand_a:
                    if time.time() > deadline:
                        return best_A, best_obj
                    
                    A1 = []
                    for i in range(k):
                        val = a * (r1 ** i)
                        if val > MAX_VAL:
                            break
                        A1.append(val)
                    
                    if len(A1) < k:
                        continue
                    
                    for b in cand_b:
                        if time.time() > deadline:
                            return best_A, best_obj
                        
                        A2 = []
                        for j in range(m):
                            val = b * (r2 ** j)
                            if val > MAX_VAL:
                                break
                            A2.append(val)
                        
                        if len(A2) < m:
                            continue
                        
                        A = A1 + A2
                        
                        # Check distinctness
                        if len(set(A)) != 20:
                            continue
                        
                        # Check range (already checked in generation, but good to be safe)
                        if max(A) > MAX_VAL:
                            continue
                        
                        obj = _objective(A)
                        if obj < best_obj:
                            best_obj = obj
                            best_A = list(A)

    return best_A, best_obj

def _local_search(A, time_limit, rng):
    """Local search with simulated annealing to minimize the objective."""
    A = list(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 100.0
    cooling_rate = 0.995
    min_temperature = 0.01
    
    while temperature > min_temperature:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random element to modify
        idx = rng.randint(0, 19)
        old_val = A[idx]
        
        # Try a few random changes
        found_improvement = False
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Random delta
            if rng.random() < 0.3:
                new_val = old_val + rng.randint(-50, 50)
            else:
                new_val = old_val + rng.randint(-200, 200)
            
            if new_val < 1 or new_val > MAX_VAL:
                continue
            if new_val in A:
                continue
            
            # Test the change
            A[idx] = new_val
            new_obj = _objective(A)
            A[idx] = old_val
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                A[idx] = new_val
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
                found_improvement = True
                break
        
        temperature *= cooling_rate
    
    return best_A, best_obj

def _model_and_optimize(time_budget):
    """
    Model-and-optimize approach:
    1. Use structural insight: geometric sequences have small sum/product sets
    2. Parameterize by (start, ratio) and locally optimize
    3. Use multi-start with different initial parameterizations
    """
    rng = random.Random(42)
    best_A = None
    best_obj = float('inf')
    
    start_time = time.time()
    
    # Phase 1: Explore geometric parameter space
    ratios = [1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.5, 3.0]
    starts = [1, 2, 3, 5, 8, 10, 15, 20]
    
    for ratio in ratios:
        if time.time() - start_time > time_budget:
            break
        for start in starts:
            if time.time() - start_time > time_budget:
                break
            A = _geometric_start(ratio, start)
            if not _is_valid(A):
                continue
            A_sorted, obj = _local_search(A, time_budget - (time.time() - start_time), rng)
            if obj < best_obj:
                best_obj = obj
                best_A = A_sorted
    
    # Phase 2: Refine with random perturbations around best
    if best_A is not None and time.time() - start_time < time_budget:
        remaining = time_budget - (time.time() - start_time)
        for _ in range(5):
            if remaining < 2:
                break
            # Perturb the best solution
            A = list(best_A)
            idx = rng.randint(0, 19)
            delta = rng.randint(-100, 100)
            new_val = A[idx] + delta
            if 1 <= new_val <= MAX_VAL and new_val not in A:
                A[idx] = new_val
            A_sorted, obj = _local_search(A, remaining - 2, rng)
            if obj < best_obj:
                best_obj = obj
                best_A = A_sorted
            remaining = time_budget - (time.time() - start_time)
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
    
    return best_A, best_obj

def _geometric_start(ratio, start_val):
    """Generate a geometric-like starting set."""
    A = set()
    current = start_val
    while len(A) < 20 and current <= MAX_VAL:
        A.add(current)
        current = int(current * ratio)
        if current <= start_val:
            break
    if len(A) < 20:
        while len(A) < 20:
            v = random.randint(1, MAX_VAL)
            A.add(v)
    return list(A)

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5  # leave margin
    
    # Phase 1: Deterministic rational geometric search
    # This exploits multiplicative collisions to minimize product set size
    deadline1 = start_time + (time_budget * 0.6)
    best_A, best_obj = _rational_geometric_search(deadline1)
    
    # Phase 2: Local search refinement
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
    
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining > 1:
        try:
            refined_A, refined_obj = _local_search(best_A, remaining, random.Random())
            if refined_obj < best_obj:
                best_A = refined_A
                best_obj = refined_obj
        except Exception:
            pass
    
    # Ensure validity
    if not _is_valid(best_A):
        best_A = list(range(1, 21))
    
    return sorted(best_A)
