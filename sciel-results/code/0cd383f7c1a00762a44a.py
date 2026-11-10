# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    DEADLINE = 625.0  # safety margin before 630s

    def compute_mobius(K):
        """Compute Mobius function mu[1..K] using a linear sieve."""
        mu = np.ones(K + 1, dtype=np.float64)
        is_composite = np.zeros(K + 1, dtype=bool)
        primes = []
        for i in range(2, K + 1):
            if not is_composite[i]:
                primes.append(i)
                mu[i] = -1.0
            for p in primes:
                if i * p > K:
                    break
                is_composite[i * p] = True
                if i % p == 0:
                    mu[i * p] = 0.0
                    break
                else:
                    mu[i * p] = -mu[i]
        return mu

    def check_feasibility_fast(domain, values, K):
        """Check M(x) <= 1.0001 for all x in [1, 10K] using efficient range updates."""
        max_x = 10 * K
        M = np.zeros(max_x + 1, dtype=np.float64)
        
        for k, f in zip(domain, values):
            if f == 0.0:
                continue
            j = 1
            while j * k <= max_x:
                lo = j * k
                hi = min((j + 1) * k, max_x + 1)
                M[lo:hi] += f
                j += 1
        
        return np.max(M[1:max_x + 1])

    def compute_objective(domain, values):
        """Compute a = -sum f(k)*log(k)/k"""
        return -np.sum(values * np.log(domain) / domain)

    # Try progressively larger K values
    K_candidates = [100, 500, 1000, 5000, 10000, 50000, 100000, 200000]
    
    best_domain = None
    best_values = None
    best_score = -np.inf

    for K in K_candidates:
        if time.time() > DEADLINE - 120:
            break
        
        mu = compute_mobius(K)
        
        # Build domain and values from non-zero mu
        nonzero_mask = mu[1:K + 1] != 0.0
        domain_base = np.where(nonzero_mask)[0] + 1
        values_base = mu[1:K + 1][nonzero_mask].copy()
        
        # Ensure 1 is in domain (mu[1] = 1)
        if domain_base[0] != 1:
            domain_base = np.concatenate([[1], domain_base])
            values_base = np.concatenate([[1.0], values_base])
        
        # Now optimize: try adding a large f(1) value
        # The constraint is sum f(k)/k = 0 (after S correction)
        # S = sum f(k)/k, then f(1) becomes f(1) - S
        # So we need: (f(1) - S) * 1 + sum_{k>1} f(k)/k = 0
        # Which means: f(1) - S + S = 0, i.e., f(1) = S... wait
        # Actually: after correction, sum_k f'(k)/k = 0 where f'(1) = f(1) - S
        # sum f'(k)/k = f'(1) + sum_{k>1} f(k)/k = (f(1) - S) + sum_{k>1} f(k)/k
        # S = f(1) + sum_{k>1} f(k)/k
        # So sum f'(k)/k = f(1) - S + sum_{k>1} f(k)/k = f(1) - f(1) - sum_{k>1} f(k)/k + sum_{k>1} f(k)/k = 0
        # This is automatically satisfied! So the S correction always makes sum f'/k = 0.
        
        # The real constraint is M(x) <= 1.0001 where M uses the CLIPPED values
        # Values are clipped to [-10, 10]
        # So we need to clip values first, then check feasibility
        
        # Let's try different f(1) values (which get clipped to [-10, 10])
        # For each candidate f(1) in [10, 9.5, 9, ..., -10], check feasibility
        
        m_base = len(domain_base)
        domain_no1 = domain_base[1:] if m_base > 1 else np.array([], dtype=np.int64)
        values_no1 = values_base[1:] if m_base > 1 else np.array([], dtype=np.float64)
        
        # Compute S for the non-1 part: sum_{k>1} f(k)/k
        if len(values_no1) > 0:
            S_no1 = np.sum(values_no1 / domain_no1)
        else:
            S_no1 = 0.0
        
        # Try f(1) values from 10 down to -10
        f1_candidates = np.arange(10.0, -10.01, -0.5)
        
        best_f1 = None
        best_f1_score = -np.inf
        
        for f1 in f1_candidates:
            if time.time() > DEADLINE - 120:
                break
            
            # Build full domain and values
            domain = np.concatenate([[1], domain_no1])
            values = np.concatenate([[f1], values_no1])
            
            # Clip values to [-10, 10]
            values_clipped = np.clip(values, -10.0, 10.0)
            
            # Check feasibility
            max_val = check_feasibility_fast(domain, values_clipped, K)
            
            if max_val <= 1.0001:
                # Compute objective with clipped values
                a = compute_objective(domain, values_clipped)
                
                if a > best_f1_score:
                    best_f1_score = a
                    best_f1 = f1
        
        if best_f1 is not None:
            domain_final = np.concatenate([[1], domain_no1])
            values_final = np.concatenate([[best_f1], values_no1])
            values_final = np.clip(values_final, -10.0, 10.0)
            
            if best_f1_score > best_score:
                best_score = best_f1_score
                best_domain = domain_final.copy()
                best_values = values_final.copy()
        
        if time.time() > DEADLINE - 120:
            break
    
    if best_domain is not None:
        return best_domain, best_values
    
    # Fallback: use the original seed
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END
