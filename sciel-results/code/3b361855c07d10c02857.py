# EVOLVE-BLOCK-START
import time
from itertools import product

P = {'n': 40, 'maximum_size': 18}

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def _exponent_vector(x, primes):
    vec = []
    for p in primes:
        e = 0
        while x % p == 0:
            x //= p
            e += 1
        vec.append(e)
    return tuple(vec)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    primes = _primes_up_to(P['n'])
    k = len(primes)
    
    # Precompute exponent vectors for all candidates 2..40
    candidates = []
    for x in range(2, P['n'] + 1):
        vec = _exponent_vector(x, primes)
        candidates.append((x, vec))
    
    # Greedy selection: at each step, pick the candidate that maximizes the number of new subset sums
    selected = []
    selected_vecs = []
    # Current set of all subset sums (exponent vectors of products)
    # Start with empty product: zero vector
    current_sums = {tuple([0]*k)}
    
    remaining = list(candidates)
    
    while len(selected) < P['maximum_size']:
        if time.time() > deadline:
            break
        
        best_candidate = None
        best_gain = -1
        best_new_sums = None
        
        for idx, (val, vec) in enumerate(remaining):
            # Compute new subset sums if we add this vector
            # New sums = {s + vec for s in current_sums}
            # But we only care about how many are new
            new_sums = set()
            for s in current_sums:
                new_s = tuple(s[i] + vec[i] for i in range(k))
                if new_s not in current_sums:
                    new_sums.add(new_s)
            
            gain = len(new_sums)
            # Tie-breaker: prefer smaller values to keep numbers small
            if gain > best_gain or (gain == best_gain and (best_candidate is None or val < best_candidate[0])):
                best_gain = gain
                best_candidate = (val, vec)
                best_new_sums = new_sums
                if gain == 0:
                    break  # No point checking others if gain is 0
        
        if best_candidate is None or best_gain <= 0:
            break
        
        # Add the best candidate
        selected.append(best_candidate[0])
        selected_vecs.append(best_candidate[1])
        
        # Update current_sums: add all new sums
        for ns in best_new_sums:
            current_sums.add(ns)
        
        # Remove from remaining
        remaining.remove(best_candidate)
        
        # Optional: early termination if we can't add more
        if not remaining:
            break
    
    # Verify distinctness of subset products (safety check)
    if selected:
        products = set()
        products.add(1)
        for mask in range(1, 1 << len(selected)):
            prod = 1
            for i in range(len(selected)):
                if mask & (1 << i):
                    prod *= selected[i]
            if prod in products:
                # Should not happen if our vector logic is correct, but just in case
                # Fall back to a simple valid set
                return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
            products.add(prod)
    
    return selected
# EVOLVE-BLOCK-END
