import math
import random
import time
import sys

# Precompute prime factors for numbers up to 20000
MAX_SUM = 20000
pf_table = [None] * (MAX_SUM + 1)

def _init_pf_table():
    # Sieve of Eratosthenes style to find smallest prime factor, then derive factors
    spf = [0] * (MAX_SUM + 1)
    for i in range(2, MAX_SUM + 1):
        if spf[i] == 0:
            for j in range(i, MAX_SUM + 1, i):
                if spf[j] == 0:
                    spf[j] = i
    
    for i in range(2, MAX_SUM + 1):
        factors = set()
        n = i
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        pf_table[i] = factors

_init_pf_table()

def _score(A):
    """Calculate the number of distinct prime factors of all pairwise sums."""
    primes = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i + 1, n):
            s = ai + A[j]
            if s <= MAX_SUM:
                primes.update(pf_table[s])
            else:
                # Fallback for safety, though unlikely if A <= 10000
                # But 10000+10000=20000 is within limit
                pass 
    return len(primes)

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    n = 12
    max_val = 10000
    
    best_score = float('inf')
    best_A = None
    
    # Initial candidates
    candidates = []
    
    # 1. Small consecutive
    candidates.append(list(range(1, 13)))
    
    # 2. Multiples of 2
    candidates.append([2 * i for i in range(1, 13)])
    
    # 3. Multiples of 6
    candidates.append([6 * i for i in range(1, 13)])
    
    # 4. Multiples of 30 (primes 2,3,5 forced)
    candidates.append([30 * i for i in range(1, 13)])
    
    # 5. Random small numbers
    for _ in range(20):
        A = random.sample(range(1, 100), 12)
        candidates.append(A)
        
    # 6. Random multiples of 2
    for _ in range(20):
        A = random.sample(range(2, 1000, 2), 12)
        candidates.append(A)

    # Evaluate initial candidates
    for A in candidates:
        s = _score(A)
        if s < best_score:
            best_score = s
            best_A = A[:]
            
    # Local Search
    # We will maintain a set of "elite" solutions to restart from
    elites = [best_A[:]]
    
    while time.time() < deadline:
        # Pick a starting point: either best, an elite, or random
        if random.random() < 0.5 and elites:
            curr_A = random.choice(elites)[:].copy()
        else:
            # Random restart
            # Bias towards small numbers or numbers with common factors
            mode = random.randint(0, 2)
            if mode == 0:
                curr_A = random.sample(range(1, 100), 12)
            elif mode == 1:
                curr_A = random.sample(range(2, 1000, 2), 12)
            else:
                curr_A = random.sample(range(1, 10000), 12)
                
        curr_score = _score(curr_A)
        
        # Hill Climbing
        stuck_count = 0
        while time.time() < deadline:
            improved = False
            # Try mutations
            # 1. Replace one element
            if random.random() < 0.7:
                idx = random.randint(0, 11)
                old_val = curr_A[idx]
                # Choose a new value
                # Bias towards values that share factors with others or are small
                if random.random() < 0.5:
                    new_val = random.randint(1, 100)
                else:
                    new_val = random.randint(1, max_val)
                
                if new_val not in curr_A:
                    curr_A[idx] = new_val
                    new_score = _score(curr_A)
                    if new_score < curr_score:
                        curr_score = new_score
                        improved = True
                        if curr_score < best_score:
                            best_score = curr_score
                            best_A = curr_A[:]
                            # Add to elites if it's good
                            if len(elites) < 20 or curr_score <= best_score:
                                if curr_A not in elites:
                                    elites.append(curr_A[:])
                                    if len(elites) > 20:
                                        elites.pop(0)
                    else:
                        curr_A[idx] = old_val
            else:
                # Swap two elements (doesn't change score, but might help if we had ordering constraints, which we don't, so this is useless for score, but let's stick to replacement)
                # Actually, swapping doesn't change the set of sums. So skip swaps.
                pass
                
            if not improved:
                stuck_count += 1
                if stuck_count > 50:
                    break
        # End of local search for this start

    # Return the best found
    # Ensure validity: distinct, <= 10000
    if best_A is None:
        return list(range(1, 13))
    
    # Check distinctness and bounds just in case
    if len(set(best_A)) != 12 or any(x > 10000 or x < 1 for x in best_A):
        # Fallback
        return list(range(1, 13))
        
    return best_A
