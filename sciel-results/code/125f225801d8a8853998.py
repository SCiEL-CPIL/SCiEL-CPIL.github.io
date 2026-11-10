import random
import time
import math

def _factor_primes(n):
    """
    Factor n and return a set of its prime factors.
    Optimized for small n (sums <= 20000).
    """
    if n <= 1:
        return set()
    primes = set()
    # Check 2
    if n % 2 == 0:
        primes.add(2)
        while n % 2 == 0:
            n //= 2
    # Check odd numbers up to sqrt(n)
    d = 3
    while d * d <= n:
        if n % d == 0:
            primes.add(d)
            while n % d == 0:
                n //= d
        d += 2
    if n > 1:
        primes.add(n)
    return primes

def _count_distinct_primes(nums):
    """
    Calculate the number of distinct prime factors across all pairwise sums of distinct elements.
    """
    all_primes = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            all_primes.update(_factor_primes(s))
    return len(all_primes)

def construct():
    start_time = time.time()
    time_limit = 55.0
    best_score = float('inf')
    best_set = None
    
    # Pre-compute a list of primes up to 20000 for faster factorization if needed,
    # but simple trial division is fine for ~66 numbers per eval.
    # Let's stick to trial division for simplicity and low overhead per call.
    
    def get_initial_set():
        # Strategy: Pick numbers that are multiples of a small smooth number, 
        # but allow variation. 
        # Or just random distinct integers.
        # A good heuristic is to start with numbers that are close to each other 
        # or have a common structure, but random is safer for global search.
        # Let's try a mix: some structured, some random.
        return sorted(random.sample(range(1, 10001), 12))

    def simulated_annealing(initial_set, max_iter=5000):
        nonlocal best_score, best_set
        
        current_set = list(initial_set)
        current_score = _count_distinct_primes(current_set)
        
        if current_score < best_score:
            best_score = current_score
            best_set = list(current_set)
            
        temp = 1.0
        cooling = 0.9995
        
        for i in range(max_iter):
            if time.time() - start_time > time_limit:
                return
            
            # Generate a neighbor
            new_set = list(current_set)
            move_type = random.random()
            
            if move_type < 0.5:
                # Replace one element
                idx = random.randint(0, 11)
                # Try to pick a value close to current or random
                if random.random() < 0.5:
                    # Local mutation
                    delta = random.randint(-100, 100)
                    new_val = current_set[idx] + delta
                else:
                    # Random replacement
                    new_val = random.randint(1, 10000)
                
                if 1 <= new_val <= 10000 and new_val not in new_set:
                    new_set[idx] = new_val
                else:
                    continue
            elif move_type < 0.8:
                # Swap two elements with random values? No, just replace two.
                idx1 = random.randint(0, 11)
                idx2 = random.randint(0, 11)
                if idx1 == idx2:
                    continue
                new_val1 = random.randint(1, 10000)
                new_val2 = random.randint(1, 10000)
                if new_val1 in new_set and new_val1 != current_set[idx1]:
                    continue
                if new_val2 in new_set and new_val2 != current_set[idx2]:
                    continue
                if new_val1 == new_val2:
                    continue
                new_set[idx1] = new_val1
                new_set[idx2] = new_val2
            else:
                # Swap two existing elements with different values from a pool?
                # Simpler: Just swap two elements (doesn't change the set, so score is same).
                # This is useless. Let's do a "perturb" move: add a small random value to a random element.
                idx = random.randint(0, 11)
                delta = random.choice([-10, -5, -2, -1, 1, 2, 5, 10])
                new_val = current_set[idx] + delta
                if 1 <= new_val <= 10000 and new_val not in new_set:
                    new_set[idx] = new_val
                else:
                    continue

            new_score = _count_distinct_primes(new_set)
            
            if new_score < best_score:
                best_score = new_score
                best_set = list(new_set)
                
            delta_score = new_score - current_score
            
            if delta_score < 0:
                current_set = new_set
                current_score = new_score
            else:
                # Accept with probability exp(-delta / temp)
                # Since scores are integers, delta is small.
                prob = math.exp(-delta_score / temp)
                if random.random() < prob:
                    current_set = new_set
                    current_score = new_score
            
            temp *= cooling
            if temp < 0.01:
                break

    # Run multiple independent simulated annealing runs
    num_runs = 20
    for run in range(num_runs):
        if time.time() - start_time > time_limit:
            break
        
        # Different initialization strategies for diversity
        if run % 3 == 0:
            # Random
            init = get_initial_set()
        elif run % 3 == 1:
            # Multiples of a small number
            base = random.choice([2, 3, 5, 7, 11, 13])
            candidates = [i * base for i in range(1, 10001 // base + 1)]
            if len(candidates) >= 12:
                init = sorted(random.sample(candidates, 12))
            else:
                init = get_initial_set()
        else:
            # Powers of 2 scaled
            base = random.choice([1, 2, 3])
            init = []
            for i in range(12):
                val = base * (2 ** i)
                if val <= 10000:
                    init.append(val)
                else:
                    init.append(10000 - (12 - i)) # Fallback
            # Ensure distinct and valid
            if len(set(init)) < 12:
                init = get_initial_set()
            else:
                init = sorted(init)
        
        simulated_annealing(init, max_iter=2000)
        
        # If we found a score of 2 or 3, we can stop early or tighten search
        if best_score <= 3:
             # Try to optimize further with more iterations if time permits
             if time.time() - start_time < time_limit - 5:
                 simulated_annealing(best_set, max_iter=5000)
             else:
                 break

    if best_set is None:
        # Fallback
        return list(range(1, 13))
    
    return best_set
