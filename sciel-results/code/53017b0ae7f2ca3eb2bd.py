import time

N = 5000
DEADLINE = 300

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    primes = _primes(N)
    
    T = [True] * (N + 1)
    current_size = N + 1
    
    counts = {}
    active_residues = {}
    
    for p in primes:
        counts[p] = [0] * p
        active_residues[p] = 0
        for r in range(p):
            if r <= N:
                c = (N - r) // p + 1
                counts[p][r] = c
                if c > 0:
                    active_residues[p] += 1
    
    while True:
        if time.time() - start_time > DEADLINE - 2:
            break
            
        violated_prime = None
        min_residue = 0
        min_count = float('inf')
        
        # Find first violated prime
        for p in primes:
            if p > current_size:
                break
            if active_residues[p] == p:
                # Find min count
                for r in range(p):
                    if counts[p][r] < min_count:
                        min_count = counts[p][r]
                        min_residue = r
                violated_prime = p
                break
        
        if violated_prime is None:
            break
        
        p = violated_prime
        r = min_residue
        
        # Find elements to remove
        to_remove = []
        for x in range(N + 1):
            if T[x] and x % p == r:
                to_remove.append(x)
                
        if not to_remove:
            # Should not happen if active_residues[p] == p
            break
            
        # Remove them
        for x in to_remove:
            T[x] = False
            current_size -= 1
            for q in primes:
                if q > current_size:
                    break
                rq = x % q
                old_c = counts[q][rq]
                if old_c > 0:
                    counts[q][rq] = old_c - 1
                    if old_c == 1:
                        active_residues[q] -= 1
                        
    result = [x for x in range(N + 1) if T[x]]
    return result
