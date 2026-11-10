import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s deadline, 5s safety margin

    # 1. Generate a list of small primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    # We need primes up to a reasonably large number to allow swaps.
    # The product of the first 15 primes is ~6e17.
    # We can swap larger primes with smaller ones to fit more primes or rearrange.
    # Let's generate primes up to 200000 to have plenty of candidates for swaps.
    primes = sieve(200000)
    
    # 2. Define tau_perp calculation optimized for speed
    def tau_perp(n):
        # Factorize n using trial division by primes up to sqrt(n)
        temp = n
        factors = []
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors.append(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            factors.append(temp)
            
        # Generate divisors in sorted order using a min-heap-like approach
        # For square-free n with k distinct primes, there are 2^k divisors.
        # We can generate them in sorted order by starting with [1] and
        # for each prime, merging the existing list with existing*prime.
        divs = [1]
        for p in factors:
            # divs is sorted. We need to merge divs with [d*p for d in divs].
            # Both are sorted. Use two-pointer merge.
            merged = []
            i = 0  # index into divs
            j = 0  # index into divs (for the multiplied version)
            len_d = len(divs)
            while i < len_d or j < len_d:
                if j >= len_d:
                    merged.append(divs[i])
                    i += 1
                elif i >= len_d:
                    merged.append(divs[j] * p)
                    j += 1
                else:
                    a = divs[i]
                    b = divs[j] * p
                    if a <= b:
                        merged.append(a)
                        i += 1
                    else:
                        merged.append(b)
                        j += 1
            divs = merged
        
        # Count adjacent coprime pairs using a bitset of prime factors
        # For each divisor, we track which primes divide it as a bitmask.
        # Two adjacent divisors are coprime iff their bitmasks are disjoint.
        # Since n is square-free (we're working with distinct primes), each divisor
        # corresponds to a unique subset of primes.
        
        # Build a mapping from divisor to bitmask
        # Actually, we can compute bitmasks on the fly during generation.
        # Let's redo: generate (divisor, bitmask) pairs in sorted order by divisor.
        
        # Restart with bitmask tracking
        pairs = [(1, 0)]  # (divisor, bitmask)
        for idx, p in enumerate(factors):
            bit = 1 << idx
            # pairs is sorted by divisor. We merge with [(d*p, m|bit) for (d,m) in pairs]
            merged = []
            i = 0
            j = 0
            len_p = len(pairs)
            while i < len_p or j < len_p:
                if j >= len_p:
                    merged.append(pairs[i])
                    i += 1
                elif i >= len_p:
                    d, m = pairs[j]
                    merged.append((d * p, m | bit))
                    j += 1
                else:
                    a_d, a_m = pairs[i]
                    b_d, b_m = (pairs[j][0] * p, pairs[j][1] | bit)
                    if a_d <= b_d:
                        merged.append((a_d, a_m))
                        i += 1
                    else:
                        merged.append((b_d, b_m))
                        j += 1
            pairs = merged
        
        # Count adjacent coprime pairs
        score = 0
        for i in range(len(pairs) - 1):
            if (pairs[i][1] & pairs[i+1][1]) == 0:
                score += 1
        return score

    # 3. Initialize with the "greedy" baseline: product of the smallest primes
    # This maximizes the number of distinct prime factors (k) within 10^18.
    # Product of first 15 primes: 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47
    # Let's calculate max k.
    prod = 1
    max_k = 0
    for p in primes:
        if prod * p > 10**18:
            break
        prod *= p
        max_k += 1
        
    # Start with the first max_k primes
    initial_primes = primes[:max_k]
    best_n = 1
    for p in initial_primes:
        best_n *= p
        
    best_score = tau_perp(best_n)
    
    # 4. Local Search (Hill Climbing)
    # Strategy: Try to swap each prime in the current set with a prime not in the set.
    # If the product remains <= 10^18 and the score improves, accept the move.
    # Repeat until no improvement is found or time runs out.
    
    # To make this efficient, we track the current set of primes as a sorted list.
    current_primes = list(initial_primes)
    current_n = best_n
    current_score = best_score
    
    # Precompute the set of primes for O(1) lookup
    current_prime_set = set(current_primes)
    
    # We will iterate through the primes in the current set.
    # For each prime p_i, we try to replace it with a prime q_j not in the set
    # such that (current_n / p_i) * q_j <= 10^18.
    
    # To limit candidate pool, we only consider primes in a window around the current max prime.
    # This is because swapping a very small prime for a very large one is unlikely to help.
    
    improved = True
    no_improve_count = 0
    while improved and time.time() < deadline - 15.0:
        improved = False
        # Shuffle the order of checking primes to avoid getting stuck in a specific local minimum
        indices = list(range(len(current_primes)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 10.0:
                return best_n
                
            p_old = current_primes[idx]
            
            # Calculate the maximum value for the new prime
            # new_n = (current_n / p_old) * p_new <= 10^18
            # p_new <= 10^18 * p_old / current_n
            max_p_new = (10**18 * p_old) // current_n
            
            # Also set a minimum to avoid swapping in very small primes that are already in the set
            # We want p_new not in current_prime_set
            # Limit search to primes in a reasonable range: from just above the smallest in set
            # to max_p_new, but skip those already in the set.
            
            # Find candidate primes. We'll search primes from index 0 up to max_p_new,
            # but skip those in current_prime_set.
            # To be more efficient, we can start from the prime just below p_old and go up.
            # But since we want to explore both smaller and larger swaps, let's just scan
            # all primes up to max_p_new, skipping those in the set.
            
            found_swap = False
            # Limit the number of candidates we try per iteration to avoid spending too much time
            candidates_tried = 0
            max_candidates = 50  # Try at most 50 candidates per prime
            
            for p_new in primes:
                if p_new in current_prime_set:
                    continue
                if p_new > max_p_new:
                    break
                if candidates_tried >= max_candidates:
                    break
                candidates_tried += 1
                
                # Potential swap
                new_n = (current_n // p_old) * p_new
                
                # Evaluate score
                s_new = tau_perp(new_n)
                
                if s_new > current_score:
                    # Accept swap
                    current_score = s_new
                    current_n = new_n
                    current_primes[idx] = p_new
                    current_prime_set.remove(p_old)
                    current_prime_set.add(p_new)
                    
                    best_score = current_score
                    best_n = current_n
                    improved = True
                    found_swap = True
                    no_improve_count = 0
                    break
            
            if found_swap:
                # Restart the index loop with the new state
                break
        
        if not improved:
            no_improve_count += 1
            # If we've had several rounds without improvement, try a random restart
            # by perturbing the current solution slightly
            if no_improve_count >= 3 and time.time() < deadline - 30.0:
                # Try a random swap to escape local minimum
                if len(current_primes) > 1:
                    idx1 = random.randint(0, len(current_primes) - 1)
                    p_old = current_primes[idx1]
                    max_p_new = (10**18 * p_old) // current_n
                    # Find a random candidate
                    candidates = [p for p in primes if p not in current_prime_set and p <= max_p_new]
                    if candidates:
                        p_new = random.choice(candidates)
                        new_n = (current_n // p_old) * p_new
                        s_new = tau_perp(new_n)
                        # Accept even if it doesn't improve, to escape local minimum
                        current_n = new_n
                        current_primes[idx1] = p_new
                        current_prime_set.remove(p_old)
                        current_prime_set.add(p_new)
                        current_score = s_new
                        no_improve_count = 0
    
    # 5. Final polish: try adding or removing single primes if possible
    if time.time() < deadline - 5.0:
        # Try to add a prime if product * p <= 10^18
        for p in primes:
            if p in current_prime_set:
                continue
            if current_n * p > 10**18:
                break
            new_n = current_n * p
            s_new = tau_perp(new_n)
            if s_new > current_score:
                current_n = new_n
                current_prime_set.add(p)
                current_primes.append(p)
                current_score = s_new
                best_score = current_score
                best_n = current_n
        
        # Try to remove a prime if it improves the score
        for idx in range(len(current_primes)):
            if time.time() > deadline - 5.0:
                break
            p_old = current_primes[idx]
            new_n = current_n // p_old
            s_new = tau_perp(new_n)
            if s_new > current_score:
                current_n = new_n
                current_prime_set.remove(p_old)
                current_primes.pop(idx)
                current_score = s_new
                best_score = current_score
                best_n = current_n
                idx -= 1  # Adjust index after removal
    
    return best_n
