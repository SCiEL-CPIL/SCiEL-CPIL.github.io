# EVOLVE-BLOCK-START
import time

import numpy as np


def prime_number_theorem_partial_function():
    """
    Return a partial function f (as (domain, values)) that maximizes
        a = -sum_k f(k) * log(k) / k
    subject to the feasibility constraint
        M(x) = sum_k f(k) * floor(x/k) <= 1.0001  for every integer x in [1, 10*K].

    We use the Mobius function mu(k).
    """
    t0 = time.time()
    deadline = t0 + 590.0   # safety margin below the 600 s limit

    def _compute_mobius(n):
        """Compute mu(1)..mu(n) using a linear sieve."""
        if n < 1:
            return np.array([], dtype=np.int64)
        mu = np.ones(n + 1, dtype=np.int64)
        is_prime = np.ones(n + 1, dtype=bool)
        primes = []
        mu[0] = 0
        for i in range(2, n + 1):
            if is_prime[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                if i * p > n:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    mu[i * p] = 0
                    break
                else:
                    mu[i * p] = -mu[i]
        return mu[1:]

    def _is_feasible(N, values):
        """
        Check if M(x) <= 1.0001 for all x in [1, 10*N].
        values is array of f(1)..f(N).
        """
        X_max = 10 * N
        g = np.zeros(X_max + 1, dtype=np.float64)
        
        for k in range(1, N + 1):
            v = values[k - 1]
            if v == 0:
                continue
            idx = np.arange(k, X_max + 1, k)
            g[idx] += v
        
        M_vals = np.cumsum(g)
        # M(x) for x in [1, X_max] corresponds to M_vals[1:]
        # Note: M(0)=0.
        if len(M_vals) > 1:
            max_M = np.max(M_vals[1:])
            return max_M <= 1.0001
        return True

    # Start with a small N
    N = 100
    domain = np.arange(1, N + 1, dtype=np.int64)
    values = _compute_mobius(N).astype(np.float64)
    
    if not _is_feasible(N, values):
        # Fallback
        return np.array([1], dtype=np.int64), np.array([1.0], dtype=np.float64)

    # Grow N
    while time.time() < deadline - 10.0:
        if N >= 200000:
            break
            
        # Increase N
        # Use a step that allows checking large ranges
        # If N is small, grow fast. If N is large, grow slower to save time?
        # Actually, check cost is O(N log N).
        # Let's just try to reach 200000 if feasible.
        
        # Try doubling or adding a large chunk
        # But we must verify feasibility at each step?
        # If we skip steps, we might miss an infeasible intermediate?
        # No, we just need the final result to be feasible.
        # But if we jump to 200000 and it's infeasible, we fail.
        # So we should check at the target.
        
        # Let's try to jump to 200000 directly if time permits, 
        # but check feasibility. If infeasible, binary search or step back?
        # Given the nature of mu, it's likely feasible for all N.
        # Let's just go for 200000.
        
        N_target = 200000
        if N_target <= N:
            break
            
        # Estimate time for check at N_target
        # If N is small, we can do it quickly.
        
        domain_next = np.arange(1, N_target + 1, dtype=np.int64)
        values_next = _compute_mobius(N_target).astype(np.float64)
        
        if _is_feasible(N_target, values_next):
            domain = domain_next
            values = values_next
            N = N_target
        else:
            # If 200000 is infeasible, try smaller?
            # For robustness, let's step up in chunks.
            step = max(1000, int(0.1 * N))
            N_next = min(200000, N + step)
            if N_next <= N:
                break
            domain_next = np.arange(1, N_next + 1, dtype=np.int64)
            values_next = _compute_mobius(N_next).astype(np.float64)
            if _is_feasible(N_next, values_next):
                domain = domain_next
                values = values_next
                N = N_next
            else:
                break

    return domain, values

# EVOLVE-BLOCK-END
