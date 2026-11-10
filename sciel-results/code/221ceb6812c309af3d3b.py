import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115s deadline, 5s safety margin

    # 1. Generate a list of small primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    # Generate primes up to 1,000,000
    primes = sieve(1000000)
    prime_set = set(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    # 2. Define tau_perp calculation
    def tau_perp(prime_factors):
        divs = [1]
        for p in prime_factors:
            divs = divs + [d * p for d in divs]
        divs.sort()
        score = 0
        gcd = math.gcd
        # Unrolled loop or simple loop
        for i in range(len(divs) - 1):
            if gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # 3. Initialize
    max_k = 15
    current_primes = primes[:max_k]
    current_n = 1
    for p in current_primes:
        current_n *= p
        
    best_n = current_n
    best_score = tau_perp(current_primes)
    current_score = best_score

    # 4. Local Search
    # We try to swap each prime p_i with a prime q not in the set.
    # Constraint: (current_n // p_i) * q <= 10**18
    
    improved = True
    # Limit iterations to avoid infinite loops or excessive time
    max_iterations = 50
    iteration = 0
    
    while improved and time.time() < deadline - 5.0 and iteration < max_iterations:
        improved = False
        iteration += 1
        current_prime_set = set(current_primes)
        
        # Shuffle indices to randomize search order
        indices = list(range(len(current_primes)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 5.0:
                return best_n
                
            p_old = current_primes[idx]
            
            # Max value for new prime
            max_p_new = (10**18 * p_old) // current_n
            
            # Find best candidate for this slot
            best_candidate = None
            best_candidate_score = current_score # Must be strictly better? Or >=?
            # Using >= might lead to oscillation. Use > for strict improvement.
            # But if we use >, we might miss equivalent states that lead to better later.
            # Let's use > for now to ensure progress.
            
            # Generate candidate primes
            # If p_old is small, max_p_new is small.
            # If p_old is large (e.g. last prime), max_p_new is large.
            
            # Optimization: Don't scan all primes if max_p_new is large.
            # Identify the index of the largest prime <= max_p_new
            # We can use bisect on the primes list.
            
            import bisect
            # Find rightmost index of prime <= max_p_new
            # primes is sorted
            max_idx = bisect.bisect_right(primes, max_p_new)
            
            # If max_idx is large, we can't scan all.
            # We will sample candidates.
            # Candidates:
            # 1. The immediate neighbors of p_old in the primes list (if not in set).
            # 2. A few random primes in the range [p_old, max_p_new] or [2, max_p_new].
            # 3. The prime just below max_p_new.
            
            candidates = set()
            
            # Add immediate neighbors
            if p_old in prime_to_idx:
                idx_old = prime_to_idx[p_old]
                for offset in [-2, -1, 1, 2, 5, 10]:
                    ni = idx_old + offset
                    if 0 <= ni < len(primes):
                        p_cand = primes[ni]
                        if p_cand <= max_p_new and p_cand not in current_prime_set:
                            candidates.add(p_cand)
            
            # Add primes near max_p_new
            if max_idx > 0:
                for offset in [-1, -10, -100, -1000]:
                    ni = max_idx - 1 + offset
                    if ni >= 0:
                        p_cand = primes[ni]
                        if p_cand not in current_prime_set:
                            candidates.add(p_cand)
            
            # Add random samples if the range is large
            # Range size is max_idx.
            if max_idx > 100:
                # Sample 20 random primes
                for _ in range(20):
                    ri = random.randint(0, max_idx - 1)
                    p_cand = primes[ri]
                    if p_cand not in current_prime_set:
                        candidates.add(p_cand)
            
            # Also check the smallest primes (2, 3, 5...) just in case
            # (Though they are likely in the set or too small to be a "swap" for a large prime)
            # For small p_old, max_p_new is small, so candidates will be small.
            
            # Evaluate candidates
            for p_new in candidates:
                if p_new in current_prime_set:
                    continue
                if p_new > max_p_new:
                    continue
                
                new_primes = current_primes.copy()
                new_primes[idx] = p_new
                
                s_new = tau_perp(new_primes)
                
                if s_new > best_candidate_score:
                    best_candidate_score = s_new
                    best_candidate = p_new
            
            if best_candidate is not None:
                # Apply the best swap
                current_primes[idx] = best_candidate
                current_n = (current_n // p_old) * best_candidate
                current_score = best_candidate_score
                best_score = current_score
                best_n = current_n
                improved = True
                # Break to restart with new state
                break

    return best_n
