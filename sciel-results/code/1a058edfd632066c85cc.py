import time
import random
from math import isqrt

# EVOLVE-BLOCK-START
P = {'n': 40, 'maximum_size': 18}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Generate candidates 2 to 40
    candidates = list(range(2, P['n'] + 1))
    
    # Precompute primes
    def _is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    primes = [a for a in range(2, P['n']+1) if _is_prime(a)]
    composites = [a for a in range(2, P['n']+1) if not _is_prime(a)]
    
    def check_set(set_vals):
        prods = {1}
        for v in set_vals:
            new_prods = list(prods)
            for prod in new_prods:
                if v * prod in prods:
                    return False
                prods.add(v * prod)
        return True

    def try_greedy(order):
        current = []
        prods = {1}
        for v in order:
            # Check if we can add v
            new_prods = []
            valid = True
            for p in prods:
                np = v * p
                if np in prods:
                    valid = False
                    break
                new_prods.append(np)
            if valid:
                current.append(v)
                prods.update(new_prods)
        return current

    def local_search(initial_set, max_swaps=50):
        """Try to improve by swapping elements"""
        current = list(initial_set)
        if not current:
            return current
        current_set = set(current)
        improved = True
        swap_count = 0
        while improved and swap_count < max_swaps and time.time() < deadline:
            improved = False
            # Try removing one element and adding another
            for i in range(len(current)):
                for j in range(P['n'] - 1):
                    candidate = j + 2
                    if candidate in current_set:
                        continue
                    # Try swapping current[i] with candidate
                    test_set = current[:i] + current[i+1:] + [candidate]
                    if check_set(test_set):
                        # Check if this is an improvement (same size, but maybe better for future)
                        # For now, just accept if it's valid and different
                        if test_set != current:
                            current = test_set
                            current_set = set(current)
                            improved = True
                            swap_count += 1
                            break
                if improved:
                    break
        return current

    best_set = []
    best_size = 0
    
    # Initial: all primes
    prods = {1}
    valid = True
    for p in primes:
        new_prods = list(prods)
        for prod in new_prods:
            if p * prod in prods:
                valid = False
                break
            prods.add(p * prod)
        if not valid:
            break
    if valid:
        best_set = primes[:]
        best_size = len(primes)
    
    # Try deterministic orderings
    orderings = [
        primes + sorted(composites),
        primes + sorted(composites, reverse=True),
        sorted(candidates),
        sorted(candidates, reverse=True),
        # Sort by largest prime factor descending
        sorted(candidates, key=lambda x: max(x, 2) if _is_prime(x) else max((d for d in range(2, isqrt(x)+1) if x % d == 0), default=x), reverse=True),
    ]
    
    for order in orderings:
        sol = try_greedy(order)
        if len(sol) > best_size:
            best_size = len(sol)
            best_set = sol[:]
    
    # Randomized restarts with diverse seeds
    rng = random.Random(42)
    
    # Phase 1: Pure random shuffles
    for i in range(3000):
        if time.time() > deadline:
            break
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        sol = try_greedy(shuffled)
        if len(sol) > best_size:
            best_size = len(sol)
            best_set = sol[:]
    
    # Phase 2: Biased random - primes more likely to be early
    for i in range(2000):
        if time.time() > deadline:
            break
        # Create a biased ordering: primes with high probability early
        order = []
        remaining_primes = primes[:]
        remaining_comps = composites[:]
        for _ in range(len(candidates)):
            if remaining_primes and (not remaining_comps or rng.random() < 0.6):
                idx = rng.randrange(len(remaining_primes))
                order.append(remaining_primes.pop(idx))
            elif remaining_comps:
                idx = rng.randrange(len(remaining_comps))
                order.append(remaining_comps.pop(idx))
            elif remaining_primes:
                idx = rng.randrange(len(remaining_primes))
                order.append(remaining_primes.pop(idx))
        sol = try_greedy(order)
        if len(sol) > best_size:
            best_size = len(sol)
            best_set = sol[:]
    
    # Phase 3: Local search on best solution found
    if best_size > 0 and time.time() < deadline:
        improved = local_search(best_set, max_swaps=100)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved[:]
    
    # Phase 4: More random restarts if time remains, with local search on improvements
    for i in range(5000):
        if time.time() > deadline:
            break
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        sol = try_greedy(shuffled)
        if len(sol) > best_size:
            best_size = len(sol)
            best_set = sol[:]
            # Try local search on the new best
            if time.time() < deadline:
                improved = local_search(best_set, max_swaps=50)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_set = improved[:]
    
    # Final verification
    if best_set and not check_set(best_set):
        # Fallback to primes if somehow invalid
        best_set = primes[:]
        best_size = len(primes)
    
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
