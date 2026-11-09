"""
Constructs the largest admissible tuple within [0, 5000].
"""

N = 5000

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    primes = _primes(N)
    num_primes = len(primes)
    
    # Initialize counts and nonzero_classes
    # counts[p_index][r] = number of elements in S congruent to r mod p
    # We use a list of lists.
    # p_index corresponds to the index in the primes list.
    
    counts = []
    nonzero_classes = []
    
    for p in primes:
        counts.append([0] * p)
        nonzero_classes.append(0)
        
    # Initialize with all elements 0..N
    # We can do this efficiently by iterating x and updating counts
    for x in range(N + 1):
        for i, p in enumerate(primes):
            r = x % p
            if counts[i][r] == 0:
                nonzero_classes[i] += 1
            counts[i][r] += 1
            
    S = set(range(N + 1))
    
    while True:
        current_size = len(S)
        if current_size == 0:
            break
            
        # Find the largest prime p <= current_size that is violated
        # Violated means nonzero_classes[i] == p
        best_i = -1
        best_p = 0
        best_min_count = float('inf')
        
        # We want to pick the prime that minimizes the number of elements removed.
        # Elements removed = min(counts[i])
        # We should prioritize smaller min_count.
        # If tie, larger p (fewer elements on average per class? No, min_count is absolute).
        # Actually, if min_count is the same, removing from larger p is "safer" 
        # because it affects fewer residue classes for smaller primes? 
        # Not necessarily. Let's just pick smallest min_count.
        
        for i, p in enumerate(primes):
            if p > current_size:
                break
            if nonzero_classes[i] == p:
                # Find min count in counts[i]
                # Since p can be up to 5000, min() is O(p)
                # But we only do this for violating primes.
                min_count = min(counts[i])
                if min_count < best_min_count:
                    best_min_count = min_count
                    best_i = i
                    best_p = p
                    
        if best_i == -1:
            break
            
        # Remove all elements in S that are congruent to r mod best_p
        # Find r with min count
        r = counts[best_i].index(best_min_count)
        
        # Identify elements to remove
        to_remove = {x for x in S if x % best_p == r}
        
        # Update S
        S.difference_update(to_remove)
        
        # Update counts and nonzero_classes
        for x in to_remove:
            for i, p in enumerate(primes):
                rr = x % p
                counts[i][rr] -= 1
                if counts[i][rr] == 0:
                    nonzero_classes[i] -= 1
                    
    return list(S)
