"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve

def _score_sequence(s):
    """True score: max|autoconv| / (sum s)^2."""
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return np.inf
    ac = fftconvolve(s, s, mode='full')
    max_ac = np.max(np.abs(ac))
    return max_ac / sq_int

def _fast_flip_eval(s, idx):
    """
    Evaluate the score after flipping s[idx] without recomputing full FFT.
    Uses the fact that autoconv of s with one element flipped can be updated.
    Actually, for speed, we just do a quick estimate or full recompute.
    For N=400, fftconvolve is fast enough. Let's just use it directly.
    """
    old_val = s[idx]
    s[idx] = -old_val
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        s[idx] = old_val
        return np.inf
    ac = fftconvolve(s, s, mode='full')
    max_ac = np.max(np.abs(ac))
    new_score = max_ac / sq_int
    s[idx] = old_val
    return new_score

def _greedy_flip_pass(s, max_passes=5):
    """
    Greedy local search: repeatedly flip the best single sign flip.
    """
    current_score = _score_sequence(s)
    for p in range(max_passes):
        improved = False
        best_gain = 0
        best_idx = -1
        for idx in range(N):
            new_score = _fast_flip_eval(s, idx)
            gain = current_score - new_score
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx >= 0:
            s[best_idx] = -s[best_idx]
            current_score -= best_gain
            improved = True
        if not improved:
            break
    return s, current_score

def _simulated_annealing(s, n_steps=1000, t_start=0.5, t_end=0.01):
    """
    Simulated annealing on sign flips.
    """
    current_score = _score_sequence(s)
    best_score = current_score
    best_s = s.copy()
    
    for i in range(n_steps):
        if time.time() > _deadline:
            break
        t = t_start * (t_end / t_start) ** (i / n_steps)
        idx = np.random.randint(N)
        old_val = s[idx]
        s[idx] = -old_val
        new_score = _score_sequence(s)
        delta = new_score - current_score
        if delta < 0 or np.random.rand() < np.exp(-delta / max(t, 1e-10)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_s = s.copy()
        else:
            s[idx] = old_val
    return best_s, best_score

def _continuous_optimize(x0, max_iter=100):
    """
    Optimize continuous values using gradient descent on a smooth surrogate.
    We minimize max|ac| / (sum s)^2.
    Use log-sum-exp for max|ac|.
    """
    s = np.tanh(x0)
    lr = 0.01
    for i in range(max_iter):
        if time.time() > _deadline:
            break
        # Compute autoconv
        ac = fftconvolve(s, s, mode='full')
        # Soft max of |ac|
        temp = 0.1
        ac_abs = np.abs(ac)
        logsumexp = np.log(np.sum(np.exp(ac_abs / temp))) * temp
        # Approximate gradient of soft max
        weights = np.exp(ac_abs / temp) / np.sum(np.exp(ac_abs / temp))
        grad_ac = weights * np.sign(ac)
        
        # Gradient of max|ac| w.r.t. s
        # d(max|ac|)/ds[i] = sum_k grad_ac[k] * dac[k]/ds[i]
        # dac[k]/ds[i] = s[i+k] + s[k-i] (for valid ranges)
        # This is expensive to compute fully. Let's use a simpler approach.
        
        # Actually, let's use scipy minimize with a smooth objective.
        pass
    
    # Fallback: just use the current s
    return s

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 130.0  # 130s deadline, 20s safety margin
    
    np.random.seed(42)
    
    best_score = np.inf
    best_s = None
    
    # Phase 1: Try several random initializations with greedy + SA
    for init_iter in range(5):
        if time.time() > _deadline:
            break
        
        # Random binary initialization
        s = np.random.choice([-1.0, 1.0], size=N)
        
        # Greedy pass
        s, score = _greedy_flip_pass(s, max_passes=3)
        
        if score < best_score:
            best_score = score
            best_s = s.copy()
        
        # Simulated annealing
        s_sa, score_sa = _simulated_annealing(s, n_steps=500, t_start=0.3, t_end=0.02)
        if score_sa < best_score:
            best_score = score_sa
            best_s = s_sa.copy()
    
    # Phase 2: More aggressive local search on best solution
    if best_s is not None:
        # Multiple greedy passes
        for _ in range(10):
            if time.time() > _deadline:
                break
            best_s, best_score = _greedy_flip_pass(best_s, max_passes=2)
        
        # Final SA polish
        s_final, score_final = _simulated_annealing(best_s, n_steps=2000, t_start=0.1, t_end=0.005)
        if score_final < best_score:
            best_score = score_final
            best_s = s_final.copy()
        
        # One more greedy pass
        best_s, best_score = _greedy_flip_pass(best_s, max_passes=5)
    
    # If we didn't find anything, return all ones
    if best_s is None:
        return np.ones(N)
    
    return best_s
# EVOLVE-BLOCK-END
