import time
import random
import heapq

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N + 1)
PRIME_SET = set(PRIMES)
PRIME_TO_IDX = {p: i for i, p in enumerate(PRIMES)}

# Precompute residues
# residues[x][i] = x % PRIMES[i]
# Only need for primes up to N+1
# Number of primes ~ 669
# N+1 = 5001
# 5001 * 669 ~ 3.3M integers.
# List of lists.
residues = [[0] * len(PRIMES) for _ in range(N + 1)]
for x in range(N + 1):
    for i, p in enumerate(PRIMES):
        residues[x][i] = x % p

def construct():
    deadline = time.time() + 290
    best_size = 0
    best_set = set()
    
    # We will run multiple random restarts
    # The order of candidates is randomized in each restart
    
    # To avoid re-initializing the huge `residues` array, we do it once.
    
    # We need to be careful with the time budget.
    # Each restart might take some time.
    # We'll do as many as we can.
    
    while time.time() < deadline:
        rng = random.Random()
        
        available = list(range(N + 1))
        rng.shuffle(available)
        ptr = 0
        
        # State
        counts = {}
        empty_sets = {}
        forbidden_res = {}
        
        # Initialize for all primes
        for p in PRIMES:
            counts[p] = [0] * p
            empty_sets[p] = set(range(p))
            forbidden_res[p] = None
        
        current_set = set()
        k = 0
        
        # Max k is N+1
        while k <= N:
            if time.time() > deadline:
                break
            
            found = False
            # Scan available candidates
            while ptr < len(available):
                x = available[ptr]
                ptr += 1
                
                # Check validity
                valid = True
                # Check primes p <= k+1
                # We iterate through PRIMES
                for i, p in enumerate(PRIMES):
                    if p > k + 1:
                        break
                    
                    r = residues[x][i]
                    if forbidden_res[p] is not None and r == forbidden_res[p]:
                        valid = False
                        break
                
                if valid:
                    # Add x
                    current_set.add(x)
                    k += 1
                    
                    # Update state for primes p <= k
                    for i, p in enumerate(PRIMES):
                        if p > k:
                            break
                        
                        r = residues[x][i]
                        if counts[p][r] == 0:
                            counts[p][r] = 1
                            empty_sets[p].remove(r)
                            if len(empty_sets[p]) == 1:
                                forbidden_res[p] = next(iter(empty_sets[p]))
                            else:
                                forbidden_res[p] = None
                        else:
                            counts[p][r] += 1
                    
                    # Handle new prime p = k+1
                    new_p = k + 1
                    if new_p in PRIME_TO_IDX:
                        idx = PRIME_TO_IDX[new_p]
                        r = residues[x][idx]
                        # counts[new_p][r] was 0
                        counts[new_p][r] = 1
                        empty_sets[new_p].remove(r)
                        if len(empty_sets[new_p]) == 1:
                            forbidden_res[new_p] = next(iter(empty_sets[new_p]))
                        else:
                            forbidden_res[new_p] = None
                    
                    found = True
                    break # Go to next k
                
                # If invalid, continue scanning (x is effectively discarded)
            
            if not found:
                break
            
            if k > best_size:
                best_size = k
                best_set = set(current_set)
        
    return list(best_set)
# EVOLVE-BLOCK-END
