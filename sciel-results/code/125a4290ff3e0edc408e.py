import math
import itertools

def _get_divisors_squarefree(primes):
    divs = [1]
    for p in primes:
        new_divs = []
        for d in divs:
            new_divs.append(d * p)
        divs.extend(new_divs)
    return divs

def _score_n_squarefree(primes):
    if not primes:
        return 0
    divs = _get_divisors_squarefree(primes)
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def _generate_primes(limit):
    sieve = [True] * (limit + 1)
    if limit >= 0: sieve[0] = False
    if limit >= 1: sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def construct():
    primes = _generate_primes(200)
    
    best_n = 2
    best_score = 1
    
    # Candidate sizes
    for k in [15, 14, 13]:
        # Start with the first k primes
        current_set = set(primes[:k])
        current_prod = 1
        for p in current_set:
            current_prod *= p
            
        if current_prod > 10**18:
            continue
            
        current_score = _score_n_squarefree(list(current_set))
        if current_score > best_score:
            best_score = current_score
            best_n = current_prod
            
        # Local search
        improved = True
        while improved:
            improved = False
            # Try swapping one prime
            for p_out in list(current_set):
                # Calculate max allowed p_in
                # (current_prod / p_out) * p_in <= 10^18
                max_p_in = int((10**18 / current_prod) * p_out)
                
                for p_in in primes:
                    if p_in in current_set:
                        continue
                    if p_in > max_p_in:
                        break
                    # Also, to avoid redundant work, we can restrict p_in to be "close"
                    # But checking all primes up to max_p_in is fine if max_p_in is small
                    # max_p_in is around 1.6 * p_out
                    new_set = (current_set - {p_out}) | {p_in}
                    new_prod = int(current_prod / p_out * p_in)
                    if new_prod > 10**18:
                        continue
                    new_score = _score_n_squarefree(list(new_set))
                    if new_score > current_score:
                        current_score = new_score
                        current_set = new_set
                        current_prod = new_prod
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_prod
                        improved = True
                        break # Restart loop with new set
                
            # Try swapping two primes?
            # This might be necessary but adds complexity.
            # Let's stick to single swaps first.
            
    return best_n
