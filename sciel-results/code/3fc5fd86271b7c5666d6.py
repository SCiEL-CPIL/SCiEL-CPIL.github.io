"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin before 150s kill


def _compute_overlap(h):
    """
    Compute the maximum overlap of h with shifted 1-h.
    The overlap at shift k is sum_i h[i] * (1 - h[(i+k) % N]).
    We want to minimize the maximum over all shifts k.
    """
    N_local = len(h)
    h_shifted = np.zeros((N_local, N_local))
    for k in range(N_local):
        h_shifted[k] = np.roll(h, k)
    # Overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N])
    # = sum_i h[i] - sum_i h[i] * h[(i+k) % N]
    # = 0.5 - sum_i h[i] * h[(i+k) % N]  (since sum h = N/2)
    # Wait, sum h[i] = N/2, so sum_i h[i] * (1 - h[(i+k)%N]) = N/2 - sum_i h[i]*h[(i+k)%N]
    # We want to minimize the max of this.
    # Equivalently, maximize the min of sum_i h[i]*h[(i+k)%N].
    
    # Let's compute autocorrelation-like quantity
    # C[k] = sum_i h[i] * h[(i+k) % N]
    # Overlap[k] = N/2 - C[k]
    # Max overlap = N/2 - min_k C[k]
    
    # Compute C[k] for all k
    C = np.zeros(N_local)
    for k in range(N_local):
        C[k] = np.dot(h, np.roll(h, k))
    
    min_C = np.min(C)
    max_overlap = N_local / 2.0 - min_C
    return max_overlap


def _objective(x):
    """
    Objective: maximize the minimum autocorrelation (equivalently minimize max overlap).
    We use a smooth surrogate: minimize -log-sum-exp of C[k] with negative sign,
    or more directly, use a soft-min approximation.
    
    Actually, let's minimize the max overlap directly using a smooth max approximation.
    max_overlap = N/2 - min_k C[k]
    We want to minimize this, i.e., maximize min_k C[k].
    
    Use soft-min: min_k C[k] ≈ -log-sum-exp(-C[k]) / log(N) ... 
    Actually, soft-min of C[k] = -log(sum(exp(-C[k]))). 
    But we want to maximize this, so minimize -soft_min = log(sum(exp(-C[k]))).
    
    Alternatively, we can just compute the raw max overlap and use it as objective,
    but it's non-smooth. For optimization, let's use a smooth approximation.
    
    Let's use: f(x) = log(sum_k exp(-C[k] / tau)) * tau, which approximates min C[k] * (-1) ... 
    Actually, soft_min(C) = -tau * log(sum exp(-C[k]/tau))
    We want to maximize soft_min(C), so minimize -soft_min(C) = tau * log(sum exp(-C[k]/tau))
    """
    h = x
    N_local = len(h)
    # Compute C[k] = sum_i h[i] * h[(i+k) % N]
    C = np.zeros(N_local)
    for k in range(N_local):
        C[k] = np.dot(h, np.roll(h, k))
    
    # Smooth minimum of C
    tau = 0.1
    # soft_min = -tau * log(sum(exp(-C/tau)))
    # We want to minimize max_overlap = N/2 - min_C ≈ N/2 - soft_min
    # So objective = N/2 + tau * log(sum(exp(-C/tau)))
    max_C = np.max(C)
    # For numerical stability
    shifted_C = C - max_C  # This makes exp(-C/tau) potentially underflow, but we want exp(-C/tau)
    # Actually, let's compute log(sum(exp(-C/tau))) safely
    # -C/tau can be large positive, so exp can overflow
    # Let m = max(-C/tau) = -min(C)/tau
    neg_C_over_tau = -C / tau
    m = np.max(neg_C_over_tau)
    log_sum_exp = m + np.log(np.sum(np.exp(neg_C_over_tau - m)))
    soft_min_neg = log_sum_exp  # This is log(sum(exp(-C/tau)))
    # soft_min(C) = -tau * log_sum_exp
    # max_overlap ≈ N/2 - soft_min(C) = N/2 + tau * log_sum_exp
    obj = N_local / 2.0 + tau * log_sum_exp
    return obj


def construct():
    global _START_TIME, _DEADLINE
    
    # Deadline check
    if time.time() > _DEADLINE - 10:
        return np.full(N, 0.5)
    
    best_h = np.full(N, 0.5)
    best_score = _compute_overlap(best_h)
    
    # Try different initializations
    np.random.seed(42)
    
    # Bang-bang patterns
    inits = []
    
    # 1. Alternating 0/1
    alt = np.zeros(N)
    alt[::2] = 1.0
    # Adjust to sum = N/2
    if N % 2 != 0:
        alt[N//2] = 0.5
        alt = alt / alt.sum() * (N/2)
    else:
        alt = alt / alt.sum() * (N/2)
    inits.append(alt.copy())
    
    # 2. Block patterns
    for block_size in [1, 2, 3, 5, 10]:
        block = np.zeros(N)
        for i in range(0, N, block_size):
            block[i:i+block_size//2] = 1.0
        block = block / block.sum() * (N/2)
        inits.append(block.copy())
    
    # 3. Random
    for _ in range(20):
        r = np.random.rand(N)
        r = r / r.sum() * (N/2)
        inits.append(r)
    
    # 4. Sine patterns
    for freq in [1, 2, 3, 5]:
        x = np.linspace(0, 2*np.pi*freq, N, endpoint=False)
        s = 0.5 + 0.5 * np.sin(x)
        s = s / s.sum() * (N/2)
        inits.append(s)
    
    # Optimize each init
    for idx, x0 in enumerate(inits):
        if time.time() > _DEADLINE - 20:
            break
        
        # Ensure bounds [0, 1]
        x0 = np.clip(x0, 0, 1)
        # Ensure sum = N/2
        s = x0.sum()
        if s > 0:
            x0 = x0 / s * (N/2)
        x0 = np.clip(x0, 0, 1)
        
        try:
            res = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                bounds=[(0, 1)] * N,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_score:
                h_candidate = res.x
                # Ensure sum = N/2
                s = h_candidate.sum()
                if s > 0:
                    h_candidate = h_candidate / s * (N/2)
                h_candidate = np.clip(h_candidate, 0, 1)
                
                score = _compute_overlap(h_candidate)
                if score < best_score:
                    best_score = score
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Polish phase with tighter tolerance
    if time.time() < _DEADLINE - 5:
        try:
            res2 = minimize(
                _objective,
                best_h,
                method='L-BFGS-B',
                bounds=[(0, 1)] * N,
                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_candidate = res2.x
            s = h_candidate.sum()
            if s > 0:
                h_candidate = h_candidate / s * (N/2)
            h_candidate = np.clip(h_candidate, 0, 1)
            
            score = _compute_overlap(h_candidate)
            if score < best_score:
                best_score = score
                best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Final adjustment to ensure exact sum
    best_h = np.clip(best_h, 0, 1)
    s = best_h.sum()
    if s > 0:
        best_h = best_h / s * (N/2)
    
    return best_h

# EVOLVE-BLOCK-END
