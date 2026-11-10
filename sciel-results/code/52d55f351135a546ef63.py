import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s deadline, 5s safety margin

    # 1. Generate a list of small primes
    # We need enough primes to potentially reach a high count.
    # Product of first 15 primes is ~6e17. First 16 is ~3e19 (too big).
    # So max distinct primes is likely around 15 or 16 if we pick small ones.
    # Let's generate primes up to 1000 to be safe for combinations.
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(2000)
    
    # 2. Define fast tau_perp calculation for square-free numbers
    # Optimization: Precompute divisors using bitmask or iterative list extension.
    # For k=15, 2^15 = 32768 divisors. Sorting and gcd is fast enough.
    # To speed up local search, we can cache scores for specific sets of primes (tuples).
    score_cache = {}

    def tau_perp_squarefree(prime_tuple):
        """
        Calculate tau_perp for a square-free number given its sorted prime factors.
        """
        if prime_tuple in score_cache:
            return score_cache[prime_tuple]
            
        k = len(prime_tuple)
        if k == 0:
            return 0
            
        # Generate all divisors
        # Bitmask approach is clean, but iterative might be faster for small k
        divs = [1]
        for p in prime_tuple:
            # For each existing divisor, multiply by p
            new_divs = [d * p for d in divs]
            divs.extend(new_divs)
            
        divs.sort()
        
        # Count adjacent coprime pairs
        # For square-free numbers, gcd(d_i, d_{i+1}) == 1 iff they share no prime factors.
        # math.gcd is fast in C, so this is efficient.
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
                
        score_cache[prime_tuple] = score
        return score

    # 3. Strategy: Start with the product of the first k primes that fits in 10^18
    # Find max k
    max_k = 0
    prod = 1
    for p in primes:
        if prod * p > 10**18:
            break
        prod *= p
        max_k += 1
        
    # Initial candidate: first max_k primes
    initial_primes = primes[:max_k]
    best_primes = list(initial_primes)
    best_primes.sort()
    best_score = tau_perp_squarefree(tuple(best_primes))
    
    # 4. Local Search (Hill Climbing)
    # We try to swap primes in the set.
    # Constraint: Product must remain <= 10^18.
    # If we swap p_out with p_in, new_prod = current_prod * p_in / p_out.
    # So p_in <= 10^18 * p_out / current_prod.
    
    current_prod = 1
    for p in best_primes:
        current_prod *= p
        
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        best_primes.sort()
        
        # Randomize the order of indices to avoid local minima bias
        indices = list(range(len(best_primes)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 10.0:
                break
                
            p_out = best_primes[i]
            # Max value for p_in
            max_p_in = int(10**18 * p_out / current_prod)
            
            # Candidate primes are those in primes list that are:
            # 1. Not already in best_primes
            # 2. <= max_p_in
            # 3. Not equal to p_out
            
            # We iterate through primes. Since primes is sorted, we break early.
            # To find good swaps, we might want to look at primes near p_out or slightly different.
            # However, scanning all valid candidates is safe.
            
            # Optimization: Precompute set for O(1) lookup
            current_set = set(best_primes)
            
            for p_in in primes:
                if p_in > max_p_in:
                    break
                if p_in in current_set:
                    continue
                    
                # Try the swap
                # Create new list
                new_primes = best_primes[:]
                new_primes[i] = p_in
                new_primes.sort()
                new_tuple = tuple(new_primes)
                
                s = tau_perp_squarefree(new_tuple)
                if s > best_score:
                    best_score = s
                    best_primes = new_primes
                    current_prod = current_prod * p_in // p_out
                    improved = True
                    break # Restart the outer loop with the new set to re-evaluate all positions
    
    # 5. Final Random Perturbation Search
    # If we have time, do some random restarts or perturbations to escape local optima
    if time.time() < deadline - 5.0:
        # We can try multi-start: generate a few random valid sets and hill-climb from there?
        # Or just perturb the current best.
        
        # Let's try a few random perturbations of the current best
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            
            curr_primes = best_primes[:]
            curr_prod = current_prod
            
            # Do a few random valid swaps
            for _ in range(3):
                i = random.randint(0, len(curr_primes) - 1)
                p_out = curr_primes[i]
                max_p_in = int(10**18 * p_out / curr_prod)
                
                # Find candidates
                current_set = set(curr_primes)
                candidates = [p for p in primes if p <= max_p_in and p not in current_set]
                
                if not candidates:
                    continue
                    
                p_in = random.choice(candidates)
                curr_primes[i] = p_in
                curr_prod = curr_prod * p_in // p_out
                curr_primes.sort()
                
            s = tau_perp_squarefree(tuple(curr_primes))
            if s > best_score:
                best_score = s
                best_primes = curr_primes
                current_prod = curr_prod

    # Construct final number
    result = 1
    for p in best_primes:
        result *= p
        
    return result
