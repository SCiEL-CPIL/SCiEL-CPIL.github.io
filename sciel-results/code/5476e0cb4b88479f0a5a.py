import time
import random
import numpy as np

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, v in enumerate(sieve) if v]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    primes = _primes_up_to(1000)
    n_primes = len(primes)
    
    # Initialize with random residues
    residues = np.array([random.randint(0, primes[i] - 1) for i in range(n_primes)], dtype=np.int64)
    
    def compute_L(res, max_check):
        """Compute the largest L such that every x in 1..L has coverage >= 2."""
        cov = np.zeros(max_check + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes[i]
            r = res[i]
            if r == 0:
                start = p
            else:
                start = r
            if start <= max_check:
                cov[start::p] += 1
        # Find largest L where cov[1..L] all >= 2
        bad = np.where(cov[1:max_check+1] < 2)[0]
        if len(bad) == 0:
            return max_check
        return bad[0] + 1 # The bad index is 0-based relative to cov[1:], so add 1 to get the actual x? 
        # Wait. cov[1] is x=1. If cov[1] < 2, bad[0] is 0. L should be 0? 
        # "largest L such that every integer 1..L satisfies". If x=1 fails, L=0.
        # If x=1,2 pass and x=3 fails, bad[0] for cov[1:4] (indices 0,1,2) is 2. L=2.
        # So return bad[0] is correct if bad is index in cov[1:].
        # Let's trace: cov[1] is index 0 in cov[1:]. If cov[1]<2, bad[0]=0. L=0. Correct.
        # If cov[1]>=2, cov[2]>=2, cov[3]<2. bad[0]=2. L=2. Correct.
        # So return bad[0] is correct.
        
    # Correction on compute_L return logic
    def compute_L_fixed(res, max_check):
        cov = np.zeros(max_check + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes[i]
            r = res[i]
            if r == 0:
                start = p
            else:
                start = r
            if start <= max_check:
                cov[start::p] += 1
        # We want largest L such that 1..L are all >= 2.
        # This is the index of the first element < 2 in cov[1:], minus 1? No.
        # If cov[1] < 2, L=0.
        # If cov[1]>=2, cov[2]<2, L=1.
        # bad = np.where(cov[1:max_check+1] < 2)[0]
        # If bad is empty, L = max_check.
        # If bad[0] is k, then cov[1+k] is the first failure. So 1..1+k-1 are good. L = 1+k-1 = k.
        # So L = bad[0].
        
        bad = np.where(cov[1:max_check+1] < 2)[0]
        if len(bad) == 0:
            return max_check
        return bad[0]

    best_res = residues.copy()
    # Start with a small limit for search, expand later
    search_limit = 2000
    best_L = compute_L_fixed(best_res, search_limit)
    
    no_improve = 0
    max_no_improve = 2000
    temp = 5.0
    
    # Pre-compute primes list for speed
    p_list = primes
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random prime index
        i = random.randint(0, n_primes - 1)
        p = p_list[i]
        old_r = best_res[i]
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue
        
        best_res[i] = new_r
        new_L = compute_L_fixed(best_res, search_limit)
        
        if new_L >= best_L:
            best_L = new_L
            no_improve = 0
            # Gradually increase search limit to push boundary
            if best_L == search_limit and search_limit < 10000:
                search_limit = min(search_limit * 2, 10000)
        else:
            # Accept with some probability
            delta = best_L - new_L
            if random.random() < np.exp(-delta / temp):
                no_improve += 1
            else:
                best_res[i] = old_r
                no_improve += 1
            
            if no_improve > 1000:
                # Kick: change several residues
                for _ in range(10):
                    k = random.randint(0, n_primes - 1)
                    best_res[k] = random.randint(0, p_list[k] - 1)
                best_L = compute_L_fixed(best_res, search_limit)
                no_improve = 0
                temp = 5.0

    # Final verification with larger limit
    final_L = compute_L_fixed(best_res, 100000)
    
    return [int(r) for r in best_res]
