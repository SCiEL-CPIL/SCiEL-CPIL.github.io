import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin: 145s out of 150s limit

    # 1. Generate primes up to a reasonable limit
    # We need primes up to ~10^6 to allow flexible swaps, 
    # but the product constraint is the main driver.
    # Primorial of first 15 primes is ~6e17, so we can fit 15 primes.
    # Let's generate primes up to 10000 to have a good pool for swaps.
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(10000)
    prime_set = set(primes)
    
    # Precompute logs for product checks? 
    # Actually, exact integer multiplication is fast enough for k~15.
    
    # 2. Fast tau_perp using Lattice Traversal (No Sorting)
    # For a square-free number with primes p_0, ..., p_{k-1},
    # the divisors form a hypercube lattice.
    # Each divisor corresponds to a bitmask m in [0, 2^k).
    # The value of the divisor is prod(p_i for i where m has bit i).
    # We need to count adjacent pairs (d, d') in sorted order with gcd(d, d')=1.
    # 
    # Key Insight: 
    # The sorted order of divisors is NOT the same as the bitmask order.
    # However, we can generate the sorted list of (value, mask) pairs 
    # using a "k-way merge" or simply generate all 2^k pairs and sort them.
    # Sorting 2^15 = 32768 items is very fast (~1ms).
    # The current program's bottleneck is likely the repeated sorting 
    # and the overhead of tuple creation in the inner loop of the search.
    #
    # Let's optimize the evaluation function specifically.
    
    def tau_perp_fast(prime_tuple):
        k = len(prime_tuple)
        if k == 0:
            return 0
        
        # Generate all divisors as (value, mask)
        # We can do this iteratively or using bit manipulation.
        # Iterative doubling is efficient.
        divs = [(1, 0)]
        for idx, p in enumerate(prime_tuple):
            bit = 1 << idx
            # Unpack and extend
            # To avoid creating too many intermediate lists, 
            # we can use a list comprehension on the existing list.
            # But we need to be careful with memory.
            # 2^15 is small.
            new_divs = []
            for v, m in divs:
                new_divs.append((v * p, m | bit))
            divs.extend(new_divs)
        
        # Sort by value
        divs.sort(key=lambda x: x[0])
        
        # Count coprime adjacent pairs
        score = 0
        for i in range(len(divs) - 1):
            if (divs[i][1] & divs[i+1][1]) == 0:
                score += 1
        return score

    # 3. Initial Candidate: Primorial of first 15 primes
    # Product of first 15 primes:
    # 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47
    # Let's calculate it to ensure it's <= 10^18
    
    init_primes = []
    prod = 1
    for p in primes:
        if prod * p > 10**18:
            break
        prod *= p
        init_primes.append(p)
        
    # If we can't fit 15, we fit as many as possible.
    # The prompt says "Large values below 10^18 are finite lower-bound constructions".
    # The 15th primorial is 614889782588491410, which is < 10^18.
    # The 16th prime is 53. 614889782588491410 * 53 = 3.2e19 > 10^18.
    # So max k is 15.
    
    current_primes = init_primes[:]
    current_prod = prod
    current_score = tau_perp_fast(tuple(current_primes))
    
    best_primes = current_primes[:]
    best_score = current_score
    best_prod = current_prod

    # 4. Simulated Annealing / Random Restart Search
    # We will try to improve the score by swapping primes.
    # Since k is fixed at 15 (max), we can only swap.
    # Swapping a small prime for a large one increases the product, 
    # potentially exceeding 10^18. 
    # We must ensure the new product <= 10^18.
    
    # Strategy: 
    # Randomly pick a prime in the set, and a candidate prime from the pool.
    # If the new product is valid, evaluate the new score.
    # Accept with probability based on delta score and "temperature".
    
    # However, since we want to MAXIMIZE score, and the landscape is discrete,
    # simple hill climbing with random restarts might be better than SA 
    # if we can afford many evaluations.
    # 
    # Let's estimate evaluation speed.
    # tau_perp_fast for k=15 involves generating 32768 pairs and sorting.
    # This is roughly 1-2 ms in Python.
    # In 140 seconds, we can do ~70,000 - 140,000 evaluations.
    # This is plenty for a good local search.
    
    # Improvement: Cache the sorted divisor structure?
    # The structure depends on the primes. If we swap one prime, 
    # we can't easily reuse the previous sort.
    # But we can optimize the generation.
    
    # Alternative: Precompute the "shape" of the lattice?
    # No, the values change.
    
    # Let's stick to the fast evaluation and do a robust local search.
    
    # Local Search: 
    # 1. Start with best known.
    # 2. For each position i, try swapping with every prime p in pool 
    #    such that new_prod <= 10^18.
    # 3. Keep the best swap.
    # 4. Repeat until no improvement.
    # 5. Random restart if stuck.
    
    # To make it faster, we can pre-filter candidates.
    # For a given p_out, max_p_in = floor(10^18 * p_out / current_prod).
    # We only need to check primes <= max_p_in.
    
    # Let's implement a more efficient local search.
    
    def local_search(primes_list, prod_val, score_val, time_limit):
        primes_list = primes_list[:]
        prod_val = prod_val
        score_val = score_val
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Sort to maintain canonical form? 
            # The score is invariant to order of primes in the tuple?
            # Yes, the set of divisors is the same.
            # But tau_perp_fast expects a tuple.
            # We can keep them sorted to make "max_p_in" calculation consistent?
            # No, max_p_in depends on the specific prime being swapped out.
            
            best_swap_score = score_val
            best_swap_idx = -1
            best_swap_prime = -1
            
            for i in range(len(primes_list)):
                p_out = primes_list[i]
                # Calculate max allowed incoming prime
                # new_prod = prod_val * p_in / p_out <= 10^18
                # p_in <= 10^18 * p_out / prod_val
                max_p_in = int(10**18 * p_out / prod_val)
                
                # Iterate through primes <= max_p_in
                # We can break early if primes are sorted
                for p_in in primes:
                    if p_in > max_p_in:
                        break
                    if p_in == p_out:
                        continue
                    if p_in in primes_list:
                        continue
                    
                    # Perform swap
                    new_primes = primes_list[:]
                    new_primes[i] = p_in
                    new_prod = prod_val * p_in // p_out
                    
                    # Check validity (integer division should be exact if p_out divides prod_val)
                    # Since p_out is a factor of prod_val, this is exact.
                    
                    s = tau_perp_fast(tuple(new_primes))
                    
                    if s > best_swap_score:
                        best_swap_score = s
                        best_swap_idx = i
                        best_swap_prime = p_in
                        # Don't break, try to find the absolute best swap in this pass
                        # Or break to speed up? 
                        # Given the cost of tau_perp_fast, finding the best swap 
                        # in a full pass is expensive. 
                        # Let's just take the first improvement to speed up the loop?
                        # No, "best improvement" is usually better for convergence.
                        # But 15 * 10000 evaluations per pass is 150k evals.
                        # 150k * 1ms = 150s. Too slow for a single pass!
                        
                        # We need a faster heuristic or fewer candidates.
                        # Let's sample candidates instead of checking all.
                        # Or, we can break after finding a significant improvement?
                        # Let's limit the number of candidates checked per position.
                        
                # Optimization: Only check a subset of candidates
                # If we check all primes <= max_p_in, it's too slow.
                # Let's check:
                # 1. The largest prime <= max_p_in
                # 2. The smallest prime not in set <= max_p_in
                # 3. A few random primes
                # This reduces the number of evaluations per pass.
                
                # Let's re-structure the inner loop to be sampled.
                
            # If we found a best swap, apply it
            if best_swap_idx != -1:
                p_out = primes_list[best_swap_idx]
                p_in = best_swap_prime
                primes_list[best_swap_idx] = p_in
                prod_val = prod_val * p_in // p_out
                score_val = best_swap_score
                improved = True
                
        return primes_list, prod_val, score_val

    # The "check all" approach is too slow. 
    # Let's implement a "Best of Sampled" strategy.
    
    def local_search_sampled(primes_list, prod_val, score_val, time_limit):
        primes_list = primes_list[:]
        prod_val = prod_val
        score_val = score_val
        
        # Pre-compute index of primes in the list for fast lookup?
        # Use a set for O(1) membership.
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_swap_score = score_val
            best_swap_idx = -1
            best_swap_prime = -1
            
            current_set = set(primes_list)
            
            for i in range(len(primes_list)):
                p_out = primes_list[i]
                max_p_in = int(10**18 * p_out / prod_val)
                
                # Sample candidates
                # We want primes <= max_p_in that are not in current_set.
                # Let's pick:
                # 1. Largest prime <= max_p_in
                # 2. Smallest prime not in set <= max_p_in
                # 3. 10 random primes <= max_p_in not in set
                
                candidates = []
                
                # Find largest
                # Binary search in primes for max_p_in
                # primes is sorted.
                lo, hi = 0, len(primes)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if primes[mid] <= max_p_in:
                        lo = mid + 1
                    else:
                        hi = mid
                # lo is the index of the first prime > max_p_in
                # So primes[lo-1] is the largest <= max_p_in
                if lo > 0:
                    p_max = primes[lo-1]
                    if p_max not in current_set:
                        candidates.append(p_max)
                
                # Find smallest not in set
                for p in primes:
                    if p > max_p_in:
                        break
                    if p not in current_set:
                        candidates.append(p)
                        break
                
                # Random samples
                # We need a list of valid primes to sample from.
                # Building this list is expensive if max_p_in is large.
                # Let's just pick random indices from the prime list 
                # and check validity.
                for _ in range(10):
                    idx = random.randint(0, lo-1)
                    p_rand = primes[idx]
                    if p_rand not in current_set:
                        candidates.append(p_rand)
                
                # Evaluate candidates
                for p_in in candidates:
                    if p_in == p_out:
                        continue
                    # Check if already in set (might be duplicate in candidates)
                    if p_in in current_set:
                        continue
                    
                    new_primes = primes_list[:]
                    new_primes[i] = p_in
                    new_prod = prod_val * p_in // p_out
                    
                    s = tau_perp_fast(tuple(new_primes))
                    
                    if s > best_swap_score:
                        best_swap_score = s
                        best_swap_idx = i
                        best_swap_prime = p_in
                        
            if best_swap_idx != -1:
                p_out = primes_list[best_swap_idx]
                p_in = best_swap_prime
                primes_list[best_swap_idx] = p_in
                prod_val = prod_val * p_in // p_out
                score_val = best_swap_score
                improved = True
                
        return primes_list, prod_val, score_val

    # Run local search
    best_primes, best_prod, best_score = local_search_sampled(
        best_primes, best_prod, best_score, deadline - 5.0
    )
    
    # If time remains, try a few random restarts from the best solution
    # to escape local optima.
    if time.time() < deadline - 10.0:
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb the best solution
            perturbed_primes = best_primes[:]
            perturbed_prod = best_prod
            perturbed_score = best_score
            
            # Do 3 random valid swaps
            for _ in range(3):
                i = random.randint(0, len(perturbed_primes) - 1)
                p_out = perturbed_primes[i]
                max_p_in = int(10**18 * p_out / perturbed_prod)
                
                # Find a valid random prime
                lo, hi = 0, len(primes)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if primes[mid] <= max_p_in:
                        lo = mid + 1
                    else:
                        hi = mid
                
                if lo == 0:
                    continue
                    
                # Pick a random prime from the valid range
                # Avoid picking one already in the set
                current_set = set(perturbed_primes)
                valid_candidates = [p for p in primes[:lo] if p not in current_set]
                if not valid_candidates:
                    continue
                
                p_in = random.choice(valid_candidates)
                perturbed_primes[i] = p_in
                perturbed_prod = perturbed_prod * p_in // p_out
                perturbed_primes.sort() # Keep sorted for consistency? Not strictly needed for score.
                
            # Evaluate perturbed
            s = tau_perp_fast(tuple(perturbed_primes))
            
            if s > best_score:
                best_primes = perturbed_primes
                best_prod = perturbed_prod
                best_score = s
                
                # Run local search from this new best
                best_primes, best_prod, best_score = local_search_sampled(
                    best_primes, best_prod, best_score, deadline - 5.0
                )

    # Final result
    result = 1
    for p in best_primes:
        result *= p
        
    return result

# Quick test to ensure it runs
if __name__ == "__main__":
    n = construct()
    print(f"Result: {n}")
    # Verify n <= 10^18
    assert 2 <= n <= 10**18
