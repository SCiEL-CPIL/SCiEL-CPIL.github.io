"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _sieve_primes(limit):
    """Return list of primes <= limit."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def _factorize(n, primes):
    """Return list of prime factors of n (with multiplicity not needed, distinct primes)."""
    factors = []
    temp = n
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            factors.append(p)
            while temp % p == 0:
                temp //= p
    if temp > 1:
        factors.append(temp)
    return factors

def _score(chosen):
    """Compute sum 1/(512 - a) for chosen integers."""
    return sum(1.0 / (512 - a) for a in chosen)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin within 150s limit
    
    N = 512
    # Primes below 512
    primes = _sieve_primes(N - 1)
    
    # Precompute prime factors for each number 1..511
    # For a=1, it's coprime to everything
    factor_cache = {}
    for a in range(1, N):
        if a == 1:
            factor_cache[a] = []
        else:
            factor_cache[a] = _factorize(a, primes)
    
    # Conflict: two numbers are in conflict if they share a prime factor
    # We want to select a subset of {1..511} that are pairwise coprime
    # Maximize sum of 1/(512-a)
    
    # Greedy: sort by weight descending, pick if coprime to all chosen
    # Weight of a is 1/(512-a), so larger a has larger weight
    # a=511 has weight 1/1, a=510 has weight 1/2, etc.
    
    # The greedy from high to low is already the seed. Let's try a smarter approach.
    
    # Key insight: 1 is coprime to everything, so always include it.
    # For primes: each prime p is coprime to all other primes and to 1.
    # But composite numbers may conflict with primes.
    
    # Strategy: 
    # 1. Always include 1 (weight 1/511, small but free)
    # 2. For primes, we can include all of them since they're pairwise coprime
    # 3. For composites, we need to check conflicts
    
    # Actually, let's think about this more carefully.
    # The maximum weight elements are: 511, 510, 509, 508, ...
    # 511 = 7*73, 510 = 2*3*5*17, 509 is prime, 508 = 4*127 = 2^2*127
    
    # A better approach: branch and bound with the greedy solution as lower bound.
    # But with 511 elements, this could be expensive.
    
    # Alternative: local search / simulated annealing on the set.
    
    # Let me try a different approach:
    # 1. Start with all primes < 512 (they're pairwise coprime) plus 1
    # 2. Then try to add composite numbers that don't conflict
    # 3. Also try swapping: remove some elements to add higher-weight ones
    
    # Actually, the greedy from high to low is quite good because it prioritizes high weights.
    # Let's enhance it with local search.
    
    # First, get the greedy solution
    def greedy_solution():
        chosen = []
        chosen_factors = set()
        for a in range(N - 1, 0, -1):
            fa = factor_cache[a]
            # Check if any factor conflicts
            conflict = False
            for p in fa:
                if p in chosen_factors:
                    conflict = True
                    break
            if not conflict:
                chosen.append(a)
                chosen_factors.update(fa)
        return chosen
    
    best_chosen = greedy_solution()
    best_score = _score(best_chosen)
    
    # Local search: try to improve by swapping elements
    # Represent solution as a set
    best_set = set(best_chosen)
    best_factors = set()
    for a in best_set:
        best_factors.update(factor_cache[a])
    
    def get_score(s):
        return sum(1.0 / (N - a) for a in s)
    
    # Local search: try adding, removing, swapping
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding a new element
        for a in range(N - 1, 0, -1):
            if a in best_set:
                continue
            fa = factor_cache[a]
            # Check if a conflicts with any element in best_set
            conflict = False
            for p in fa:
                if p in best_factors:
                    conflict = True
                    break
            if not conflict:
                # Add a
                new_set = best_set | {a}
                new_score = get_score(new_set)
                if new_score > best_score + 1e-12:
                    best_set = new_set
                    best_score = new_score
                    best_factors.update(fa)
                    improved = True
                    break
        
        if time.time() >= deadline:
            break
        
        # Try swapping: remove one element, add another
        # This is more complex; let's try simple removals first
        # Try removing an element and seeing if we can add a better one
        for a_remove in list(best_set):
            if a_remove == 1:
                continue  # Keep 1 always
            # Compute score without a_remove
            score_without = best_score - 1.0 / (N - a_remove)
            # Compute factors without a_remove
            factors_without = set(best_factors)
            for p in factor_cache[a_remove]:
                # Only remove if not used by other elements
                count = 0
                for b in best_set:
                    if b != a_remove and p in factor_cache[b]:
                        count += 1
                        break
                if count == 0:
                    factors_without.discard(p)
            
            # Now try to add the best possible element
            for a_add in range(N - 1, 0, -1):
                if a_add in best_set:
                    continue
                fa = factor_cache[a_add]
                conflict = False
                for p in fa:
                    if p in factors_without:
                        conflict = True
                        break
                if not conflict:
                    new_score = score_without + 1.0 / (N - a_add)
                    if new_score > best_score + 1e-12:
                        # Perform swap
                        best_set.remove(a_remove)
                        best_set.add(a_add)
                        best_factors = factors_without
                        best_factors.update(fa)
                        best_score = new_score
                        improved = True
                        break
            if improved:
                break
    
    # Convert back to list
    return list(best_set)
# EVOLVE-BLOCK-END
