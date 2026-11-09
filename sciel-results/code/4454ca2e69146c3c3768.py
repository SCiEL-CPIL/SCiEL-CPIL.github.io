"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

def construct():
    start_time = time.time()
    deadline = start_time + 300  # 300 seconds
    
    # Start with all integers in [0, N]
    S = set(range(N + 1))
    
    # Sort primes in descending order to potentially remove fewer elements per step
    primes_desc = sorted(PRIMES, reverse=True)
    
    while True:
        if time.time() > deadline - 10:  # Safety margin
            break
        
        m = len(S)
        if m == 0:
            break
            
        # Find primes p <= m
        violated = False
        # We need to check primes p <= m.
        # Since primes_desc is sorted descending, we can iterate and skip those > m.
        # But it's better to just iterate all and check condition.
        for p in primes_desc:
            if p > m:
                continue
                
            # Count residues
            counts = [0] * p
            for x in S:
                counts[x % p] += 1
                
            # Check if any residue class is empty
            if min(counts) == 0:
                continue
                
            # All classes are present. We must remove one.
            # Remove the one with the minimum count.
            min_count = min(counts)
            min_residue = counts.index(min_count)
            
            # Remove all x in S such that x % p == min_residue
            # Using a list comprehension to find candidates then removing
            to_remove = [x for x in S if x % p == min_residue]
            for x in to_remove:
                S.remove(x)
                
            violated = True
            break  # Restart loop with new S
            
        if not violated:
            break
            
    return list(S)
# EVOLVE-BLOCK-END
