import numpy as np
import random
import time
import math

def _primes_up_to(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

def construct():
    primes = _primes_up_to(1000)
    n_primes = len(primes)
    N = 100000
    
    # Precompute weights for biased move selection in SA
    # Smaller primes are more impactful
    weights = np.array([1.0 / p for p in primes])
    weights /= weights.sum()
    
    global_best_r = None
    global_best_L = 0
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Helper to rebuild cov from r
    def build_cov(r):
        cov = np.zeros(N + 1, dtype=np.int16)
        for i, p in enumerate(primes):
            rr = r[i]
            if rr == 0:
                cov[p::p] += 1
            else:
                cov[rr::p] += 1
        cov[0] = 100 # Ignore 0
        return cov

    def get_L(cov):
        bad = np.nonzero(cov[1:] < 2)[0]
        if len(bad) == 0:
            return N
        return bad[0] + 1

    def apply_change(cov, r, i, new_r):
        p = primes[i]
        old_r = r[i]
        if old_r == new_r:
            return
        
        # Remove old
        if old_r == 0:
            cov[p::p] -= 1
        else:
            cov[old_r::p] -= 1
            
        # Add new
        r[i] = new_r
        if new_r == 0:
            cov[p::p] += 1
        else:
            cov[new_r::p] += 1
            
    def revert_change(cov, r, i, old_r):
        p = primes[i]
        new_r = r[i]
        if old_r == new_r:
            return
        
        # Remove new
        if new_r == 0:
            cov[p::p] -= 1
        else:
            cov[new_r::p] -= 1
            
        # Add old
        r[i] = old_r
        if old_r == 0:
            cov[p::p] += 1
        else:
            cov[old_r::p] += 1

    # Initializations
    inits = [
        np.zeros(n_primes, dtype=np.int32),
        np.ones(n_primes, dtype=np.int32),
        np.array([random.randint(0, p-1) for p in primes], dtype=np.int32),
        np.array([random.randint(0, p-1) for p in primes], dtype=np.int32),
        np.array([random.randint(0, p-1) for p in primes], dtype=np.int32),
        np.array([random.randint(0, p-1) for p in primes], dtype=np.int32)
    ]
    
    for init_r in inits:
        if time.time() > deadline:
            break
            
        r = init_r.copy()
        cov = build_cov(r)
        L = get_L(cov)
        
        # Greedy Phase: Fix the first uncovered point repeatedly
        # We try to increase L by changing residues to cover L+1
        while L < N and time.time() < deadline:
            x = L + 1
            # Find primes that can cover x
            # A prime p covers x if r_p == x % p
            # We want to change some r_p to x % p to increase coverage of x
            # But we must not break coverage of any y < x
            
            best_i = -1
            best_score = -1
            target_r = x % primes[best_i] if best_i != -1 else -1 # placeholder
            
            # Heuristic: Try to find a prime p where changing r_p to x%p is "safe"
            # Safe means min cov in the old residue class (for indices < x) is >= 3
            # And ideally, the new residue class also has good coverage?
            # Actually, if we add coverage to x, we might break x if we remove it from elsewhere?
            # No, we are changing r_p.
            # Old residue class loses 1 coverage. New gains 1.
            # We need min(cov[old_class] for y < x) >= 3 to be safe.
            
            # To speed up, we only check a subset of primes or use a random order
            # Let's iterate through primes in random order to avoid bias
            order = np.random.permutation(n_primes)
            
            for i in order:
                p = primes[i]
                target_r = x % p
                if r[i] == target_r:
                    continue
                
                old_r = r[i]
                # Check safety of removing old_r coverage
                # We need min cov in old_r class for indices 1..x-1 to be >= 3
                # Note: indices in old_r class are old_r, old_r+p, ...
                # We only care about indices < x
                
                if old_r == 0:
                    # Indices: p, 2p, ... < x
                    if x <= p:
                        min_cov = 100
                    else:
                        # np.min on empty slice is error, but x>p implies at least one element if p < x
                        # If p >= x, no elements < x.
                        min_cov = np.min(cov[p:x:p]) if p < x else 100
                else:
                    # Indices: old_r, old_r+p, ... < x
                    # If old_r >= x, no elements.
                    if old_r >= x:
                        min_cov = 100
                    else:
                        # Slice old_r to x with step p
                        # If old_r < x, there is at least one element (old_r itself, if old_r >= 1)
                        # Note: old_r can be 0, handled above. So old_r >= 1.
                        min_cov = np.min(cov[old_r:x:p])
                
                if min_cov >= 3:
                    # Prefer moves with higher min_cov (safer)
                    # Or smaller p (more impact)?
                    # Let's use min_cov as primary, p as secondary
                    score = min_cov * 10000 - p
                    if score > best_score:
                        best_score = score
                        best_i = i
                        
            if best_i != -1:
                apply_change(cov, r, best_i, x % primes[best_i])
                new_L = get_L(cov)
                if new_L > L:
                    L = new_L
                    if L > global_best_L:
                        global_best_L = L
                        global_best_r = r.copy()
                else:
                    # If it didn't improve L, revert?
                    # The greedy phase assumes safe moves improve L.
                    # If it doesn't, it might have broken something else or didn't fix x.
                    # We should revert if it didn't improve L to stay in a good state?
                    # Or keep it if it's neutral?
                    # Let's revert to be safe in greedy phase.
                    revert_change(cov, r, best_i, x % primes[best_i])
                    break
            else:
                break
        
        # Update Global Best
        if L > global_best_L:
            global_best_L = L
            global_best_r = r.copy()
            
        # SA Phase
        T = 1.0
        cooling = 0.9995
        no_improve_count = 0
        max_no_improve = 1000
        
        while time.time() < deadline:
            # Guided Move Selection
            if random.random() < 0.5:
                # Targeted move: Try to fix L+1
                x = L + 1
                if x > N:
                    break
                i = random.choices(range(n_primes), weights=weights, k=1)[0]
                p = primes[i]
                new_r = x % p
            else:
                # Random move
                i = random.choices(range(n_primes), weights=weights, k=1)[0]
                p = primes[i]
                new_r = random.randint(0, p - 1)
            
            old_r = r[i]
            
            if old_r == new_r:
                continue
                
            apply_change(cov, r, i, new_r)
            new_L = get_L(cov)
            
            delta = new_L - L
            
            if delta > 0 or random.random() < math.exp(delta / T):
                L = new_L
                if L > global_best_L:
                    global_best_L = L
                    global_best_r = r.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                revert_change(cov, r, i, old_r)
                no_improve_count += 1
                
            T *= cooling
            
            if T < 0.01 or no_improve_count > max_no_improve:
                # Restart from current best of this run? Or global best?
                # Let's restart from the global best to escape local minima
                if time.time() > deadline - 5:
                    break
                r = global_best_r.copy()
                cov = build_cov(r)
                L = get_L(cov)
                T = 1.0
                no_improve_count = 0

    if global_best_r is None:
        return [0] * n_primes
        
    return global_best_r.tolist()
