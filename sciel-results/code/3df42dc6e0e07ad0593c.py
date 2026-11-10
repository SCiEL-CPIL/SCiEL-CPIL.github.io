import time
import random
import itertools

def calc_score(A):
    """
    Calculate the max of the size of the sumset and product set of A.
    A is a list of distinct positive integers.
    """
    sumset = set(a + b for a in A for b in A)
    prodset = set(a * b for a in A for b in A)
    return max(len(sumset), len(prodset))

def generate_smooth_numbers(limit, primes):
    """Generate all numbers <= limit whose prime factors are in primes."""
    nums = {1}
    for p in primes:
        new_nums = set()
        for n in nums:
            val = n
            while val * p <= limit:
                val *= p
                new_nums.add(val)
        nums.update(new_nums)
    return sorted(nums)

def local_search(A, time_limit, rng, pool):
    """
    Perform local search to minimize the score.
    """
    A = list(A)
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    start_time = time.time()
    
    # Shuffle pool initially
    pool_shuffled = list(pool)
    rng.shuffle(pool_shuffled)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Re-shuffle pool each pass to avoid order bias
        rng.shuffle(pool_shuffled)
        
        for i in range(len(A)):
            if (time.time() - start_time) > time_limit:
                return best_A, best_score
            
            old_val = A[i]
            
            for new_val in pool_shuffled:
                if new_val == old_val or new_val in A:
                    continue
                
                # Create new set
                new_A = A.copy()
                new_A[i] = new_val
                
                # Calculate score
                score = calc_score(new_A)
                
                if score < current_score:
                    current_score = score
                    A = new_A
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_A = list(A)
                    # Break inner loop to move to next i with updated A
                    break
                    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin
    
    rng = random.Random(42)
    
    # Candidate pool: 2,3,5,7-smooth numbers <= 10000
    # Plus small integers 1..100
    pool_smooth = generate_smooth_numbers(10000, [2, 3, 5, 7])
    pool_small = list(range(1, 101))
    pool = list(set(pool_smooth + pool_small))
    
    # Initial candidates
    candidates = []
    
    # 1. 2,3-smooth grid (10, 2) -> 20 elements
    # 2^i * 3^j, i in 0..9, j in 0..1
    A1 = [ (2**i) * (3**j) for i in range(10) for j in range(2) ]
    candidates.append(A1)
    
    # 2. 2,3-smooth grid (5, 4)
    A2 = [ (2**i) * (3**j) for i in range(5) for j in range(4) ]
    candidates.append(A2)
    
    # 3. 2,3-smooth grid (4, 5)
    A3 = [ (2**i) * (3**j) for i in range(4) for j in range(5) ]
    candidates.append(A3)
    
    # 4. 2,3,5-smooth grid?
    # Take 20 smallest 2,3,5-smooth numbers
    smooth_235 = generate_smooth_numbers(10000, [2, 3, 5])
    candidates.append(smooth_235[:20])
    
    # 5. Arithmetic Progression 1..20 (baseline)
    candidates.append(list(range(1, 21)))
    
    # 6. Random samples from pool
    for _ in range(5):
        try:
            candidates.append(rng.sample(pool, 20))
        except ValueError:
            pass
    
    best_A = None
    best_score = float('inf')
    
    # Evaluate initial candidates quickly to find a good starting point
    for A in candidates:
        if len(set(A)) != 20 or any(x > 10000 for x in A):
            continue
        score = calc_score(A)
        if score < best_score:
            best_score = score
            best_A = list(A)
            
    if best_A is None:
        best_A = list(range(1, 21))
        best_score = calc_score(best_A)
        
    # Run local search on the best initial candidate first
    # Then try others or restarts
    
    # We will run local search for a fixed duration, then restart if time remains
    # Or run until no improvement, then restart with perturbation
    
    while time.time() < deadline - 5.0:
        # Pick a starting point
        # 50% chance: best_A
        # 50% chance: random from initial candidates or perturbed best_A
        if rng.random() < 0.5 and best_A:
            start_A = best_A
        else:
            # Pick a random candidate
            start_A = rng.choice(candidates)
            if len(set(start_A)) != 20:
                continue
            
        # Run local search
        # Give it a time slice, e.g., 5 seconds, or until no improvement
        # Since local search returns when no improvement, it might finish fast.
        # If it finishes fast, we can immediately try another.
        
        improved_A, score = local_search(start_A, 5.0, rng, pool)
        
        if score < best_score:
            best_score = score
            best_A = improved_A
            
        # If we have time, we can try to perturb best_A and search again
        # to escape local minima
        if time.time() < deadline - 5.0 and best_A:
            # Perturb: swap 2-3 random elements
            perturbed = list(best_A)
            for _ in range(3):
                idx1 = rng.randint(0, 19)
                idx2 = rng.randint(0, 19)
                # Swap with random pool elements
                new_val1 = rng.choice(pool)
                new_val2 = rng.choice(pool)
                
                # Ensure distinctness and bounds
                # Simple check
                if new_val1 not in perturbed and 1 <= new_val1 <= 10000:
                    perturbed[idx1] = new_val1
                if new_val2 not in perturbed and 1 <= new_val2 <= 10000:
                    perturbed[idx2] = new_val2
                    
            if len(set(perturbed)) == 20 and all(1 <= x <= 10000 for x in perturbed):
                improved_A, score = local_search(perturbed, 3.0, rng, pool)
                if score < best_score:
                    best_score = score
                    best_A = improved_A

    return best_A
