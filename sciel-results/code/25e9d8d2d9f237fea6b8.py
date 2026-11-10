import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Generate primes up to 1000
    primes = []
    for n in range(2, 1001):
        is_prime = True
        for q in range(2, int(n**0.5) + 1):
            if n % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
    
    n_primes = len(primes)  # 168
    L_max = 3000
    
    # Precompute covered indices
    covered_indices = []
    for p in primes:
        arr = []
        for r in range(p):
            if r == 0:
                idx = np.arange(p, L_max + 1, p)
            else:
                idx = np.arange(r, L_max + 1, p)
            arr.append(idx)
        covered_indices.append(arr)
    
    def build_coverage(res):
        cov = np.zeros(L_max + 1, dtype=np.int32)
        for i in range(n_primes):
            cov[covered_indices[i][res[i]]] += 1
        return cov
    
    def score_from_cov(cov):
        cmin = np.minimum.accumulate(cov[1:L_max+1])
        valid = cmin >= 2
        if not valid.any():
            return 0
        return int(np.where(valid)[0][-1]) + 1
    
    # Seed initialization
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    def local_search(res, max_iters=100000):
        res = list(res)
        cov = build_coverage(res)
        current_score = score_from_cov(cov)
        best_res = res[:]
        best_score = current_score
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            i = np.random.randint(n_primes)
            p = primes[i]
            old_r = res[i]
            new_r = np.random.randint(p)
            if new_r == old_r:
                continue
            
            old_idx = covered_indices[i][old_r]
            new_idx = covered_indices[i][new_r]
            
            cov[old_idx] -= 1
            cov[new_idx] += 1
            
            new_score = score_from_cov(cov)
            
            if new_score >= current_score:
                current_score = new_score
                res[i] = new_r
                if new_score > best_score:
                    best_score = new_score
                    best_res = res[:]
            else:
                cov[old_idx] += 1
                cov[new_idx] -= 1
        
        return best_res, best_score
    
    best_res = residues[:]
    best_score = score_from_cov(build_coverage(best_res))
    
    while time.time() < deadline - 5:
        start_res = best_res[:]
        n_perturb = np.random.randint(3, 25)
        for _ in range(n_perturb):
            i = np.random.randint(n_primes)
            start_res[i] = int(np.random.randint(primes[i]))
        
        res, score = local_search(start_res, max_iters=80000)
        if score > best_score:
            best_score = score
            best_res = res
        
        if best_score >= 1000:
            break
    
    return best_res
