import numpy as np
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 600.0
    safety_margin = 10.0
    
    K_max = 2000
    mu = np.ones(K_max + 1, dtype=np.int64)
    is_prime = np.ones(K_max + 1, dtype=bool)
    
    mu[0] = 0
    mu[1] = 1
    for i in range(2, K_max + 1):
        if is_prime[i]:
            for j in range(i, K_max + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K_max:
                for j in range(i2, K_max + 1, i2):
                    mu[j] = 0

    k_vals = np.arange(1, K_max + 1, dtype=np.float64)
    term_S = mu[1:K_max+1].astype(np.float64) / k_vals
    prefix_S = np.cumsum(term_S)
    
    log_k = np.log(k_vals)
    term_A = -mu[1:K_max+1].astype(np.float64) * log_k / k_vals
    prefix_A = np.cumsum(term_A)

    candidates = []
    
    def compute_M2(K):
        x_vals = np.arange(K + 1, 10 * K + 1)
        R_vals = np.zeros_like(x_vals)
        nz_mu = mu[1:K+1]
        nz_k = np.arange(1, K+1)
        mask = nz_mu != 0
        nz_mu = nz_mu[mask]
        nz_k = nz_k[mask]
        
        for k, m in zip(nz_k, nz_mu):
            R_vals += m * (x_vals / k - np.floor(x_vals / k))
            
        return -np.min(R_vals)

    for K in range(2, K_max + 1):
        if time.time() > deadline - safety_margin:
            break
            
        S_mu = prefix_S[K-1]
        
        if S_mu >= 0:
            M1 = 1.0 - S_mu
        else:
            M1 = 1.0 - S_mu * K
        
        # Estimate M2
        if K <= 100:
            M2 = compute_M2(K)
        else:
            # Sample based estimate
            x_samples = np.arange(K + 1, 10 * K + 1, max(1, K // 20))
            R_samples = np.zeros_like(x_samples)
            nz_mu = mu[1:K+1]
            nz_k = np.arange(1, K+1)
            mask = nz_mu != 0
            nz_mu = nz_mu[mask]
            nz_k = nz_k[mask]
            for k, m in zip(nz_k, nz_mu):
                R_samples += m * (x_samples / k - np.floor(x_samples / k))
            M2 = -np.min(R_samples)

        M_est = max(M1, M2)
        if M_est <= 0:
            continue
            
        c_est = 1.0001 / M_est
        A_K = prefix_A[K-1]
        score_est = c_est * A_K
        
        candidates.append((score_est, K))
        
        # Keep top 5
        if len(candidates) > 5:
            candidates.sort(reverse=True)
            candidates = candidates[:5]

    # Verify top candidates
    best_score = -np.inf
    best_K = 1
    best_c = 1.0
    
    for score_est, K in candidates:
        if time.time() > deadline - safety_margin:
            break
            
        S_mu = prefix_S[K-1]
        if S_mu >= 0:
            M1 = 1.0 - S_mu
        else:
            M1 = 1.0 - S_mu * K
            
        M2 = compute_M2(K)
        M_true = max(M1, M2)
        
        if M_true <= 0:
            continue
            
        c = 1.0001 / M_true
        A_K = prefix_A[K-1]
        score = c * A_K
        
        if score > best_score:
            best_score = score
            best_K = K
            best_c = c

    domain = np.arange(1, best_K + 1, dtype=np.int64)
    values = best_c * mu[1:best_K + 1].astype(np.float64)
    
    return domain, values
