import random
import time
import math

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin before 60s limit
    
    # 1. Generate a pool of candidate integers
    # We want numbers that are likely to produce overlaps in sums and products.
    # Smooth numbers are good for products. Small numbers are good for sums.
    pool = set()
    
    # Add small integers
    for i in range(1, 200):
        pool.add(i)
        
    # Add 5-smooth numbers (primes 2, 3, 5) up to 10000
    def is_smooth(n, primes):
        for p in primes:
            while n % p == 0:
                n //= p
        return n == 1
        
    primes = [2, 3, 5]
    for i in range(1, 10001):
        if is_smooth(i, primes):
            pool.add(i)
            
    # Add some 7-smooth numbers
    primes7 = [2, 3, 5, 7]
    for i in range(1, 10001):
        if is_smooth(i, primes7):
            pool.add(i)
            
    # Add powers of 2 and 3
    for b in [2, 3, 4]:
        val = 1
        while val <= 10000:
            pool.add(val)
            val *= b
            
    pool_list = list(pool)
    random.shuffle(pool_list)
    
    def evaluate(A):
        # A is a set of integers
        sums = set()
        prods = set()
        for a in A:
            for b in A:
                sums.add(a + b)
                prods.add(a * b)
        return max(len(sums), len(prods))
        
    def local_search(initial_set, max_iters=50000):
        A = set(initial_set)
        current_cost = evaluate(A)
        best_A = A.copy()
        best_cost = current_cost
        
        T = 10.0
        cooling = 0.9999
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            # Pick a random element to replace
            idx = random.randint(0, 19)
            elem_list = list(A)
            old_elem = elem_list[idx]
            
            # Pick a new element from pool
            # Ensure it's not already in A
            new_elem = None
            for _ in range(10): # Try a few times to find a valid new element
                candidate = random.choice(pool_list)
                if candidate not in A:
                    new_elem = candidate
                    break
            
            if new_elem is None:
                continue
                
            # Perform swap
            A.remove(old_elem)
            A.add(new_elem)
            
            new_cost = evaluate(A)
            
            # Accept if better, or with probability exp(-delta/T)
            if new_cost < current_cost:
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = A.copy()
            else:
                delta = new_cost - current_cost
                if delta > 0:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_cost = new_cost
                    else:
                        # Revert
                        A.remove(new_elem)
                        A.add(old_elem)
                else:
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = A.copy()
                        
            T *= cooling
            
        return best_A, best_cost

    # Generate diverse initial sets
    initial_sets = []
    
    # 1. First 20 smooth numbers
    smooths = sorted([x for x in pool if is_smooth(x, [2,3,5])])
    if len(smooths) >= 20:
        initial_sets.append(smooths[:20])
        
    # 2. First 20 integers
    initial_sets.append(list(range(1, 21)))
    
    # 3. Geometric progression (scaled)
    # Try to fit 20 elements in 1..10000
    # r^19 <= 10000 => r <= 10000^(1/19) ~ 1.8
    # Let's try ratio 2 but cap at 10000 and fill gaps with smooths?
    # Or just a dense set of small numbers.
    
    # 4. Random smooth numbers
    random.shuffle(smooths)
    if len(smooths) >= 20:
        initial_sets.append(smooths[:20])
        
    # 5. Mix of small and medium smooths
    mixed = sorted(pool_list[:100])
    initial_sets.append(mixed[:20])
    
    best_overall_A = None
    best_overall_cost = float('inf')
    
    # Run local search on each initial set
    for init in initial_sets:
        if time.time() > deadline:
            break
        # Ensure distinct and valid
        init_set = set(init)
        if len(init_set) != 20:
            continue
        if max(init_set) > 10000:
            continue
            
        A, cost = local_search(init_set)
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_A = A
            
        # If we found a very good solution, we might stop early or continue to refine
        # Theoretical min is around 39-45.
        if best_overall_cost <= 40:
            # Can't do much better than 2n-1=39. 40 is excellent.
            break
            
    # Final safety check
    if best_overall_A is None:
        return list(range(1, 21))
        
    return sorted(list(best_overall_A))

# EVOLVE-BLOCK-END
