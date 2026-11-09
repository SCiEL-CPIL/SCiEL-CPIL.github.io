import numpy as np
import time
import random
from math import exp, gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit

    def evaluate(A):
        """Compute max(|A+A|, |A*A|) efficiently."""
        n = len(A)
        sums = set()
        prods = set()
        # Unroll loops for speed if n is small, but standard loops are fine for n=20
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
        return max(len(sums), len(prods))

    # Candidate pool: Include small integers, powers of 2, 3, 5, 6, etc.
    # We want numbers that have many multiplicative relations.
    pool = set()
    # Add 1..50
    for i in range(1, 51):
        pool.add(i)
    # Add powers of 2
    p = 1
    while p <= 10000:
        pool.add(p)
        p *= 2
    # Add powers of 3
    p = 1
    while p <= 10000:
        pool.add(p)
        p *= 3
    # Add 2^i * 3^j (smooth numbers)
    p2 = 1
    while p2 <= 10000:
        p3 = 1
        while p2 * p3 <= 10000:
            pool.add(p2 * p3)
            p3 *= 3
        p2 *= 2
    # Add some 5-smooth
    p2 = 1
    while p2 <= 10000:
        p3 = 1
        while p2 * p3 <= 10000:
            p5 = 1
            while p2 * p3 * p5 <= 10000:
                pool.add(p2 * p3 * p5)
                p5 *= 5
            p3 *= 3
        p2 *= 2
        
    pool = sorted(pool)
    pool_index = {v: i for i, v in enumerate(pool)}
    n_pool = len(pool)

    # Precompute "neighbors" in pool for faster moves?
    # Instead, we will pick random candidates from a subset of the pool
    # to keep the search space manageable and focused on structured numbers.
    # Let's restrict the search to the first 500 numbers in the pool (mostly small/smooth)
    # and some larger powers.
    search_pool = pool[:300] + [p for p in pool if p > 1000][:50]
    search_pool = sorted(set(search_pool))
    search_pool_index = {v: i for i, v in enumerate(search_pool)}
    n_search = len(search_pool)

    # Initial states
    initial_states = []
    
    # 1. Small integers
    initial_states.append(list(range(1, 21)))
    
    # 2. Powers of 2
    initial_states.append([2**i for i in range(20)])
    
    # 3. Powers of 3
    initial_states.append([3**i for i in range(1, 21) if 3**i <= 10000])
    if len(initial_states[-1]) < 20:
         initial_states.pop()
         
    # 4. 2-smooth numbers (Hamming)
    hamming = [x for x in pool if x <= 10000]
    # Filter to only 2^i 3^j
    h23 = set()
    p2 = 1
    while p2 <= 10000:
        p3 = 1
        while p2 * p3 <= 10000:
            h23.add(p2 * p3)
            p3 *= 3
        p2 *= 2
    h23 = sorted(h23)
    if len(h23) >= 20:
        initial_states.append(h23[:20])
        
    # 5. Random from search pool
    for _ in range(5):
        initial_states.append(random.sample(search_pool, 20))

    best_score = float('inf')
    best_A = None

    def sa_run(init_A, temp_start, cooling_rate, min_temp):
        nonlocal best_score, best_A
        current_A = list(init_A)
        # Ensure distinct
        current_A = sorted(set(current_A))
        while len(current_A) < 20:
             current_A.append(random.choice(search_pool))
        current_A = sorted(current_A[:20])
        
        current_score = evaluate(current_A)
        if current_score < best_score:
            best_score = current_score
            best_A = list(current_A)

        temp = temp_start
        while temp > min_temp:
            if time.time() > deadline:
                break

            # Move: Replace one element
            idx = random.randint(0, 19)
            old_val = current_A[idx]
            
            # Pick a new value from search pool
            # Bias towards values with similar magnitude or structure?
            # For now, uniform from search_pool
            new_val = search_pool[random.randint(0, n_search - 1)]
            
            if new_val == old_val:
                continue
            if new_val in current_A:
                continue
                
            current_A[idx] = new_val
            new_score = evaluate(current_A)
            delta = new_score - current_score

            if delta < 0 or random.random() < exp(-delta / temp):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_A = list(current_A)
            else:
                current_A[idx] = old_val

            # Occasionally do a 2-swap to escape local minima
            if random.random() < 0.05:
                i1, i2 = random.sample(range(20), 2)
                old1, old2 = current_A[i1], current_A[i2]
                new1 = search_pool[random.randint(0, n_search - 1)]
                new2 = search_pool[random.randint(0, n_search - 1)]
                
                # Check validity
                if new1 == new2 or new1 in current_A or new2 in current_A:
                    continue
                if new1 == old1 or new1 == old2 or new2 == old1 or new2 == old2:
                    continue
                
                current_A[i1] = new1
                current_A[i2] = new2
                new_score = evaluate(current_A)
                delta = new_score - current_score

                if delta < 0 or random.random() < exp(-delta / temp):
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_A = list(current_A)
                else:
                    current_A[i1] = old1
                    current_A[i2] = old2

            temp *= cooling_rate

    # Run SA with multiple restarts
    for init_A in initial_states:
        if time.time() > deadline:
            break
        # Adjust parameters for restarts
        sa_run(init_A, 50.0, 0.999, 0.1)

    # Final polish: Run SA on the best found so far with lower temp
    if best_A is not None and time.time() < deadline - 2.0:
        sa_run(best_A, 5.0, 0.998, 0.01)

    return best_A
