import math
import random
import time

def _get_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

PRIMES = _get_primes(1000)
N_PRIMES = len(PRIMES)
L_MAX = 5000

def _calc_L(counts):
    for x in range(1, L_MAX + 1):
        if counts[x] < 2:
            return x - 1
    return L_MAX

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_res = None
    best_score = 0
    
    # Helper to init counts for a residue set
    def init_counts(res):
        counts = [0] * (L_MAX + 1)
        for i in range(N_PRIMES):
            p = PRIMES[i]
            r = res[i]
            start = p if r == 0 else r
            if start > L_MAX:
                continue
            for x in range(start, L_MAX + 1, p):
                counts[x] += 1
        return counts

    # 1. Try seed
    res_seed = [((i + 2) // 2) % p for i, p in enumerate(PRIMES)]
    counts_seed = init_counts(res_seed)
    score_seed = _calc_L(counts_seed)
    if score_seed > best_score:
        best_score = score_seed
        best_res = res_seed[:]
        best_counts = counts_seed

    # 2. Try random starts
    for _ in range(10):
        res = [random.randint(0, p-1) for p in PRIMES]
        counts = init_counts(res)
        score = _calc_L(counts)
        if score > best_score:
            best_score = score
            best_res = res[:]
            best_counts = counts

    # 3. Simulated Annealing with Incremental Updates
    sa_time_budget = 5.0
    
    while time.time() < deadline:
        # Start from best, perturb
        current_res = best_res[:]
        for _ in range(N_PRIMES // 5):
            idx = random.randint(0, N_PRIMES - 1)
            current_res[idx] = random.randint(0, PRIMES[idx] - 1)
            
        current_counts = init_counts(current_res)
        current_score = _calc_L(current_counts)
        
        T = 5.0
        T_min = 0.01
        alpha = 0.999
        
        sa_start = time.time()
        
        while time.time() < sa_start + sa_time_budget and time.time() < deadline:
            if T < T_min:
                break
                
            idx = random.randint(0, N_PRIMES - 1)
            p = PRIMES[idx]
            old_r = current_res[idx]
            new_r = random.randint(0, p - 1)
            
            if new_r == old_r:
                continue
                
            # Update counts
            # Remove old
            old_start = p if old_r == 0 else old_r
            if old_start <= L_MAX:
                for x in range(old_start, L_MAX + 1, p):
                    current_counts[x] -= 1
                    
            # Add new
            new_start = p if new_r == 0 else new_r
            if new_start <= L_MAX:
                for x in range(new_start, L_MAX + 1, p):
                    current_counts[x] += 1
                    
            current_res[idx] = new_r
            new_score = _calc_L(current_counts)
            
            if new_score >= current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_res = current_res[:]
                    # No need to copy counts, just keep reference or copy if we want to restore?
                    # We are modifying current_counts in place.
                    # If we reject, we must revert.
                    # So we don't update best_counts yet.
            else:
                prob = math.exp((new_score - current_score) / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Reject: Revert counts and res
                    # Remove new
                    if new_start <= L_MAX:
                        for x in range(new_start, L_MAX + 1, p):
                            current_counts[x] -= 1
                    # Add old
                    if old_start <= L_MAX:
                        for x in range(old_start, L_MAX + 1, p):
                            current_counts[x] += 1
                    current_res[idx] = old_r
                    
            T *= alpha
            
    return best_res
