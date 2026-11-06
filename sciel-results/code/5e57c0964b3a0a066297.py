import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80 second deadline (10s safety margin from 90s limit)
    
    def is_prime(n):
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
    
    def generate_primes_up_to(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]
    
    def factorize(n):
        factors = set()
        temp = n
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            if p * p > temp:
                break
            while temp % p == 0:
                factors.add(p)
                temp //= p
        if temp > 1:
            factors.add(temp)
        return factors
    
    def count_distinct_prime_factors(arr):
        all_factors = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                all_factors |= factorize(s)
        return len(all_factors)
    
    # Try specific known good sets first
    # Powers of 2: sums of distinct powers of 2 are of form 2^a + 2^b = 2^min(a,b)*(1+2^|a-b|)
    # The factor (1+2^k) may introduce new primes
    # Let's just compute
    
    best_score = float('inf')
    best_result = list(range(1, 13))
    
    # Strategy 1: Try sets based on powers of small numbers
    # Strategy 2: Random search with local improvement
    # Strategy 3: Graph-based clique approach
    
    # Quick evaluation of some structured candidates
    candidates = []
    
    # Powers of 2 up to 10000
    powers_of_2 = [2**i for i in range(1, 14) if 2**i <= 10000]
    if len(powers_of_2) >= 12:
        for start in range(len(powers_of_2) - 11):
            subset = powers_of_2[start:start+12]
            candidates.append(subset)
    
    # Multiples of a base number
    for base in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16]:
        for offset in range(0, 20):
            subset = [base * i + offset for i in range(1, 13)]
            if all(1 <= x <= 10000 for x in subset) and len(set(subset)) == 12:
                candidates.append(subset)
    
    # Arithmetic progressions
    for a in range(1, 20):
        for d in range(1, 20):
            subset = [a + d*i for i in range(12)]
            if all(1 <= x <= 10000 for x in subset):
                candidates.append(subset)
    
    # Evaluate candidates
    for cand in candidates:
        if len(cand) != 12 or len(set(cand)) != 12:
            continue
        if any(x < 1 or x > 10000 for x in cand):
            continue
        score = count_distinct_prime_factors(cand)
        if score < best_score:
            best_score = score
            best_result = cand[:]
    
    # Local search / hill climbing
    if best_score <= 5:
        # Already good, try to improve
        pass
    
    # Random search with restarts
    random.seed(42)
    
    def random_subset():
        return random.sample(range(1, 10001), 12)
    
    # Try random sets
    max_random_iters = 5000
    iters_done = 0
    while iters_done < max_random_iters and time.time() < deadline - 5:
        subset = random_subset()
        score = count_distinct_prime_factors(subset)
        if score < best_score:
            best_score = score
            best_result = subset[:]
        iters_done += 1
    
    # Local improvement from best found
    if best_score > 3:
        current = best_result[:]
        for _ in range(1000):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(12):
                old_val = current[i]
                # Try replacing with a random value
                for _ in range(50):
                    if time.time() > deadline - 2:
                        break
                    new_val = random.randint(1, 10000)
                    if new_val in current and new_val != old_val:
                        continue
                    new_current = current[:]
                    new_current[i] = new_val
                    if len(set(new_current)) != 12:
                        continue
                    score = count_distinct_prime_factors(new_current)
                    if score < best_score:
                        best_score = score
                        best_result = new_current[:]
                        current = new_current[:]
                        improved = True
                        break
            if improved:
                break
    
    # Graph-based approach for small prime sets
    # For P = {2, 3}, find 12 numbers where all pairwise sums are 3-smooth
    def find_clique_for_prime_set(prime_set, max_val=10000, clique_size=12):
        if time.time() > deadline - 3:
            return None
        
        # Generate all numbers <= 2*max_val that are P-smooth
        smooth_numbers = set()
        for i in range(2, 2*max_val + 1):
            n = i
            for p in prime_set:
                while n % p == 0:
                    n //= p
            if n == 1:
                smooth_numbers.add(i)
        
        # Build adjacency: u connected to v if u+v is smooth and u != v
        # Use greedy clique finding
        # Sort by some heuristic
        all_nodes = list(range(1, max_val + 1))
        
        # Greedy with random restarts
        best_clique = []
        for restart in range(100):
            if time.time() > deadline - 2:
                break
            
            random.shuffle(all_nodes)
            clique = []
            candidates = list(all_nodes)
            
            while len(clique) < clique_size and candidates:
                # Pick a candidate that connects to all in clique
                found = False
                random.shuffle(candidates)
                for v in candidates:
                    if v in clique:
                        continue
                    ok = True
                    for u in clique:
                        if u + v not in smooth_numbers:
                            ok = False
                            break
                    if ok:
                        clique.append(v)
                        # Remove candidates that don't connect to v
                        candidates = [c for c in candidates if c != v and c + v in smooth_numbers and c not in clique]
                        found = True
                        break
                if not found:
                    break
            
            if len(clique) == clique_size:
                best_clique = clique
                break
        
        if len(best_clique) == clique_size:
            return sorted(best_clique)
        return None
    
    # Try small prime sets
    prime_sets_to_try = [
        [2],
        [2, 3],
        [2, 5],
        [2, 3, 5],
        [3, 5],
        [2, 7],
        [2, 3, 7],
    ]
    
    for ps in prime_sets_to_try:
        if time.time() > deadline - 5:
            break
        clique = find_clique_for_prime_set(ps)
        if clique is not None and len(clique) == 12:
            score = count_distinct_prime_factors(clique)
            if score < best_score:
                best_score = score
                best_result = clique[:]
    
    # Final local polish
    if time.time() < deadline - 1:
        current = best_result[:]
        for _ in range(200):
            if time.time() > deadline - 0.5:
                break
            improved = False
            for i in range(12):
                if time.time() > deadline - 0.5:
                    break
                old_val = current[i]
                # Try nearby values
                for delta in [-3, -2, -1, 1, 2, 3]:
                    new_val = old_val + delta
                    if new_val < 1 or new_val > 10000:
                        continue
                    if new_val in current and new_val != old_val:
                        continue
                    new_current = current[:]
                    new_current[i] = new_val
                    if len(set(new_current)) != 12:
                        continue
                    score = count_distinct_prime_factors(new_current)
                    if score < best_score:
                        best_score = score
                        best_result = new_current[:]
                        current = new_current[:]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return best_result
