"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

def _compute_L(res, primes, cap=10**5):
    # res[i] is residue for primes[i]
    counts = [0] * (cap + 1)
    for i, r in enumerate(res):
        p = primes[i]
        # count occurrences of r mod p in 1..cap
        # x = r + k*p for k>=0, x>=1
        # first x >= 1
        if r == 0:
            start = p
        else:
            start = r
        if start <= cap:
            counts[start] += 1
            for x in range(start + p, cap + 1, p):
                counts[x] += 1
    L = 0
    for x in range(1, cap + 1):
        if counts[x] >= 2:
            L = x
        else:
            break
    return L

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s

    primes = _primes_upto(1000)
    n_primes = len(primes)  # 168

    # Initial heuristic: assign residue i to cover i+1
    # For prime p_i, to cover i+1: r_i = (i+1) % p_i
    # But we want double coverage. Let's try a better initial.
    # Start with seed-like: r_i = ceil((i+1)/2) mod p_i
    # Actually seed uses ((i+2)//2) % p
    # Let's use a slightly different start and then improve.
    # Or just random start and SA.
    # SA is known to work well.
    
    def random_res():
        return [random.randint(0, p-1) for p in primes]
    
    def hill_climb(res):
        # Greedy local search: change one residue at a time if it improves L
        # We need efficient L evaluation or incremental.
        # Given time constraints, maybe just random restarts with SA?
        # Or simple local search with full re-eval is too slow (168 * 10^5 ops per eval).
        # Let's use incremental SA.
        pass

    best_res = None
    best_L = 0

    # Incremental SA
    # State: res, L, counts (optional if we recalc L from scratch? No, too slow)
    # But full re-eval of L takes ~0.1-0.2s. 120s / 0.2s = 600 evals.
    # 600 random restarts of 100 steps each?
    # Or 10 restarts of 60 steps?
    
    # Let's try: 
    # 1. Try a few deterministic heuristics.
    # 2. Run SA with full re-eval if incremental is too complex to implement correctly quickly.
    # Actually, implementing incremental L correctly is hard because L changes non-locally.
    # However, changing r_i by 1 changes counts at positions where x ≡ r_i (mod p).
    # These are sparse.
    # Let's try to implement incremental counts.
    
    # counts[x] = number of i such that x ≡ r_i (mod p_i)
    # If we change r_i from old_r to new_r:
    # For all x in 1..cap where x ≡ old_r (mod p_i), counts[x] -= 1
    # For all x in 1..cap where x ≡ new_r (mod p_i), counts[x] += 1
    # Then we need to find the largest L such that counts[x] >= 2 for all 1..L.
    # This requires scanning from 1. 
    # If we maintain a data structure for "minimum counts in prefix"? 
    # Or just scan. Scanning 10^5 takes ~5ms in Python. 
    # 120s / 5ms = 24,000 steps. 
    # This is feasible!
    
    def compute_L_from_counts(counts):
        L = 0
        # Use numpy for speed? np.where(counts < 2)[0]
        # If no index < L+1 has count < 2, then L is the min index with count < 2 minus 1.
        # But we need prefix property.
        # So find first index x where counts[x] < 2.
        # If none, L = cap.
        # Using numpy is faster.
        import numpy as np
        idx = np.where(np.array(counts[1:]) < 2)[0]
        if len(idx) == 0:
            return cap
        else:
            return idx[0] # first index where count < 2 is idx[0]+1, so L is idx[0]
    
    # Let's use numpy for counts to speed up the check.
    import numpy as np
    
    def init_counts(res):
        counts = np.zeros(cap+1, dtype=np.int32)
        for i, r in enumerate(res):
            p = primes[i]
            if r == 0:
                start = p
            else:
                start = r
            if start <= cap:
                counts[start::p] += 1
        return counts

    def get_L(counts):
        # Find first index x in 1..cap where counts[x] < 2
        # counts is 1-indexed effectively (index 0 unused)
        bad = np.where(counts[1:] < 2)[0]
        if len(bad) == 0:
            return cap
        else:
            return bad[0]
    
    best_L = 0
    best_res = None
    
    # Heuristic 1: Seed
    res_seed = [((i + 2) // 2) % primes[i] for i in range(n_primes)]
    counts_seed = init_counts(res_seed)
    L_seed = get_L(counts_seed)
    if L_seed > best_L:
        best_L = L_seed
        best_res = res_seed[:]
        
    # Heuristic 2: Cover 1..n_primes with double coverage
    # Assign primes to cover pairs.
    # For i in 0..n_primes-1, we have prime p_i.
    # We want to cover 1..n_primes twice.
    # Assign p_i to cover i+1 and i+2?
    # Let's just random start and SA.
    
    # SA Parameters
    T_start = 50.0
    T_end = 0.1
    steps = 20000 # Adjust based on time
    restarts = 5
    
    for restart in range(restarts):
        # Random start
        res = random_res()
        counts = init_counts(res)
        L = get_L(counts)
        if L > best_L:
            best_L = L
            best_res = res[:]
        
        T = T_start
        cooling = (T_end / T_start) ** (1.0 / steps)
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
            
            # Pick random index
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = res[i]
            new_r = random.randint(0, p - 1)
            
            if old_r == new_r:
                continue
            
            # Apply change
            # Decrement old
            if old_r == 0:
                start_old = p
            else:
                start_old = old_r
            if start_old <= cap:
                counts[start_old::p] -= 1
            
            # Increment new
            if new_r == 0:
                start_new = p
            else:
                start_new = new_r
            if start_new <= cap:
                counts[start_new::p] += 1
            
            new_L = get_L(counts)
            
            delta_L = new_L - L
            
            if delta_L > 0 or random.random() < math.exp(delta_L / T):
                res[i] = new_r
                L = new_L
                if L > best_L:
                    best_L = L
                    best_res = res[:]
            else:
                # Revert
                counts[start_old::p] += 1
                if new_r == 0:
                    start_new = p
                else:
                    start_new = new_r
                if start_new <= cap:
                    counts[start_new::p] -= 1
            
            T *= cooling
        
        if time.time() > deadline - 10:
            break

    # Final check
    return best_res

# EVOLVE-BLOCK-END
