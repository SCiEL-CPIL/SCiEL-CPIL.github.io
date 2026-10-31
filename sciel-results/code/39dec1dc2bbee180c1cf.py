"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random


def _max_overlap_fast(h):
    """Compute max overlap using FFT-based circular cross-correlation."""
    N = len(h)
    one_minus_h = 1.0 - h
    H = np.fft.fft(h)
    G = np.fft.fft(one_minus_h)
    overlaps = np.real(np.fft.ifft(H * np.conj(G)))
    return np.max(overlaps)


def _score(h):
    """Score is the upper bound on C5, which is the max overlap."""
    return _max_overlap_fast(h)


def _fix_sum(h, N):
    """Ensure h sums to exactly N/2 and is in [0,1]."""
    h = np.clip(h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-14:
        if diff > 0:
            order = np.argsort(1.0 - h)
            for idx in order:
                if diff <= 1e-14:
                    break
                add = min(diff, 1.0 - h[idx])
                h[idx] += add
                diff -= add
        else:
            order = np.argsort(-h)
            for idx in order:
                if -diff <= 1e-14:
                    break
                sub = min(-diff, h[idx])
                h[idx] -= sub
                diff += sub
    return h


def _fourier_domain_solve(N, time_budget=40.0):
    """
    Structural change: work in Fourier domain.
    
    Key insight: overlap_k = sum_j h_j * (1-h)_{j+k} (circular)
    In Fourier: overlap_k = IFFT(H * conj(G))_k where G = FFT(1-h) = N - H
    
    So overlap_k = IFFT(H * conj(N - H))_k = IFFT(N*conj(H) - |H|^2)_k
    
    We want min max_k overlap_k subject to:
      0 <= h_j <= 1
      sum(h) = N/2  =>  H[0] = N/2
    
    Strategy: parameterize h via its Fourier coefficients, use projected gradient
    on the max-overlap objective with explicit bounds on h.
    """
    start = time.time()
    
    # Initialize: alternating pattern has only DC and Nyquist freq nonzero
    # h = [a, 1-a, a, 1-a, ...] => H[k] = 0 except k=0 and k=N/2
    h = np.zeros(N)
    h[0::2] = 0.5
    h[1::2] = 0.5  # start at constant, then evolve
    
    # Better init: use a sinusoidal pattern
    # h_j = 0.5 + A * cos(2*pi*m*j/N)
    # This gives H[m] = H[N-m] = N*A/2, others 0
    # Sum = N*0.5 = N/2 always satisfied!
    
    def _evaluate(A, m):
        """Create h from sinusoidal parameterization."""
        h = np.full(N, 0.5)
        h += A * np.cos(2 * np.pi * m * np.arange(N) / N)
        h = np.clip(h, 0, 1)
        h = _fix_sum(h, N)
        return h
    
    # Try multiple sinusoidal modes and amplitudes
    best_h = _evaluate(0.0, 0)  # constant
    best_score = _score(best_h)
    
    for m in [1, 2, 3, 4, 5, 8, 10, 12, 15, 20, 25]:
        if time.time() - start > time_budget - 5:
            break
        for A in [0.1, 0.2, 0.3, 0.4, 0.45, 0.49]:
            if time.time() - start > time_budget - 3:
                break
            h_try = _evaluate(A, m)
            s_try = _score(h_try)
            if s_try < best_score:
                best_score = s_try
                best_h = h_try
    
    # Now do projected gradient descent on the actual h vector
    # Objective: max_k overlap_k where overlap = IFFT(H * conj(N-H))
    # Gradient of max overlap w.r.t. h:
    # overlap_k = sum_j h_j (1-h)_{j+k} = sum_j h_j - sum_j h_j h_{j+k}
    # d(overlap_k)/dh_i = (1-h)_{i+k} - h_i * delta... 
    # Actually: overlap_k = sum_j h_j (1-h_{j+k})
    # d/dh_i = (1-h_{i+k}) for the term where j=i
    # But also h_i appears in (1-h_{j+k}) when j+k=i, i.e., j=i-k
    # So d(overlap_k)/dh_i = (1-h_{i+k}) - h_{i-k}
    
    def _overlaps_and_grad(h):
        """Compute all overlaps and gradient of max overlap."""
        one_minus_h = 1.0 - h
        H = np.fft.fft(h)
        G = np.fft.fft(one_minus_h)
        overlaps = np.real(np.fft.ifft(H * np.conj(G)))
        
        # Gradient: d(overlap_k)/dh_i = (1-h)_{(i+k)%N} - h_{(i-k)%N}
        # For the max, gradient is at the argmax k*
        k_max = np.argmax(overlaps)
        # grad_i = (1-h)_{(i+k_max)%N} - h_{(i-k_max)%N}
        shifted_omh = np.roll(one_minus_h, -k_max)  # (1-h)_{i+k_max}
        shifted_h = np.roll(h, k_max)                # h_{i-k_max}
        grad = shifted_omh - shifted_h
        return overlaps, k_max, grad
    
    current = best_h.copy()
    current_score = _score(current)
    
    # Projected gradient descent with adaptive step
    lr = 0.01
    for it in range(2000):
        if time.time() - start > time_budget - 2:
            break
        
        overlaps, k_max, grad = _overlaps_and_grad(current)
        cur_ov = overlaps[k_max]
        
        # Projected gradient step
        new_h = current - lr * grad
        # Project to [0,1]
        new_h = np.clip(new_h, 0, 1)
        # Fix sum
        new_h = _fix_sum(new_h, N)
        
        new_score = _score(new_h)
        if new_score < current_score - 1e-12:
            current = new_h
            current_score = new_score
            # Increase learning rate slightly
            lr = min(lr * 1.05, 0.05)
        else:
            lr = max(lr * 0.7, 0.001)
    
    return current, current_score


def _try_alternating(N):
    """Try alternating high/low values with optimal a=0.5."""
    a = 0.5
    h = np.zeros(N)
    h[0::2] = a
    h[1::2] = 1 - a
    return _fix_sum(h, N)


def _local_search_fast(h, max_iter=5000, step_size=0.02, time_limit=15.0):
    """Fast local search using FFT-based scoring."""
    N = len(h)
    current = h.copy()
    current_score = _score(current)
    start = time.time()
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        if i == j:
            continue
        delta = random.uniform(-step_size, step_size)
        new_i = current[i] + delta
        new_j = current[j] - delta
        if new_i < 0 or new_i > 1 or new_j < 0 or new_j > 1:
            continue
        
        new_h = current.copy()
        new_h[i] = new_i
        new_h[j] = new_j
        
        new_score = _score(new_h)
        if new_score < current_score - 1e-12:
            current = new_h
            current_score = new_score
    
    return current, current_score


def _refine_with_scipy(h, time_budget=20.0):
    """Refine h using scipy minimize with penalty for constraints."""
    start = time.time()
    N = len(h)
    
    def _objective(x):
        if time.time() - start > time_budget:
            return 1e10
        h = np.clip(x, 0, 1)
        sum_penalty = (np.sum(h) - N/2.0) ** 2 * 1000
        return _score(h) + sum_penalty
    
    best_h = h.copy()
    best_score = _score(h)
    
    for trial in range(3):
        if time.time() - start > time_budget - 2:
            break
        if trial == 0:
            x0 = h.copy()
        elif trial == 1:
            x0 = h + np.random.normal(0, 0.05, N)
            x0 = np.clip(x0, 0, 1)
        else:
            x0 = h + np.random.normal(0, 0.1, N)
            x0 = np.clip(x0, 0, 1)
        
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            h_new = np.clip(res.x, 0, 1)
            h_new = _fix_sum(h_new, N)
            s_new = _score(h_new)
            if s_new < best_score:
                best_score = s_new
                best_h = h_new
        except Exception:
            pass
    
    return best_h, best_score


def _final_polish(h, time_budget=10.0):
    """High-precision polish with very small steps."""
    N = len(h)
    current = h.copy()
    current_score = _score(current)
    start = time.time()
    
    for it in range(3000):
        if time.time() - start > time_budget:
            break
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        if i == j:
            continue
        delta = random.uniform(-0.005, 0.005)
        new_i = current[i] + delta
        new_j = current[j] - delta
        if new_i < 0 or new_i > 1 or new_j < 0 or new_j > 1:
            continue
        
        new_h = current.copy()
        new_h[i] = new_i
        new_h[j] = new_j
        
        new_score = _score(new_h)
        if new_score < current_score - 1e-13:
            current = new_h
            current_score = new_score
    
    return current, current_score


def _soft_max_surrogate(h, beta=50.0):
    """
    Smooth surrogate for max overlap using log-sum-exp.
    """
    N = len(h)
    one_minus_h = 1.0 - h
    H = np.fft.fft(h)
    G = np.fft.fft(one_minus_h)
    overlaps = np.real(np.fft.ifft(H * np.conj(G)))
    max_ov = np.max(overlaps)
    log_sum = np.log(np.sum(np.exp(beta * (overlaps - max_ov)))) + beta * max_ov
    return log_sum / beta


def _refine_with_gradient(h, time_budget=25.0):
    """
    Refine using gradient-based optimization with a smooth surrogate objective.
    """
    start = time.time()
    N = len(h)
    
    def _smooth_objective(x):
        if time.time() - start > time_budget:
            return 1e10
        hx = np.clip(x, 0, 1)
        sum_penalty = (np.sum(hx) - N/2.0) ** 2 * 100
        return _soft_max_surrogate(hx, beta=50.0) + sum_penalty
    
    def _smooth_gradient(x):
        if time.time() - start > time_budget:
            return np.zeros_like(x)
        hx = np.clip(x, 0, 1)
        eps = 1e-7
        grad = np.zeros(N)
        for i in range(N):
            x_plus = hx.copy()
            x_plus[i] += eps
            x_minus = hx.copy()
            x_minus[i] -= eps
            f_plus = _soft_max_surrogate(np.clip(x_plus, 0, 1), beta=50.0) + (np.sum(np.clip(x_plus, 0, 1)) - N/2.0)**2 * 100
            f_minus = _soft_max_surrogate(np.clip(x_minus, 0, 1), beta=50.0) + (np.sum(np.clip(x_minus, 0, 1)) - N/2.0)**2 * 100
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad
    
    best_h = h.copy()
    best_score = _score(h)
    
    for trial in range(2):
        if time.time() - start > time_budget - 5:
            break
        
        if trial == 0:
            x0 = h.copy()
        else:
            x0 = h + np.random.normal(0, 0.03, N)
            x0 = np.clip(x0, 0, 1)
        
        try:
            res = minimize(_smooth_objective, x0, method='L-BFGS-B',
                           jac=_smooth_gradient,
                           bounds=[(0, 1)] * N,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            h_new = np.clip(res.x, 0, 1)
            h_new = _fix_sum(h_new, N)
            s_new = _score(h_new)
            if s_new < best_score:
                best_score = s_new
                best_h = h_new
        except Exception:
            pass
    
    return best_h, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin from 150s limit
    
    random.seed(42)
    np.random.seed(42)
    
    # Start with alternating pattern
    best_h = _try_alternating(N)
    best_score = _score(best_h)
    
    # NEW: Fourier-domain structural approach
    if time.time() < deadline - 55:
        h_fourier, s_fourier = _fourier_domain_solve(N, time_budget=40.0)
        if s_fourier < best_score:
            best_score = s_fourier
            best_h = h_fourier
    
    # Local search from best
    if time.time() < deadline - 30:
        h1, s1 = _local_search_fast(best_h, max_iter=5000, step_size=0.02, time_limit=15.0)
        if s1 < best_score:
            best_score = s1
            best_h = h1
    
    # Refine with scipy Nelder-Mead
    if time.time() < deadline - 15:
        h_ref, s_ref = _refine_with_scipy(best_h, time_budget=10.0)
        if s_ref < best_score:
            best_score = s_ref
            best_h = h_ref
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        h3, s3 = _final_polish(best_h, time_budget=8.0)
        if s3 < best_score:
            best_score = s3
            best_h = h3
    
    # Ensure exact sum and bounds
    best_h = _fix_sum(best_h, N)
    
    return best_h
# EVOLVE-BLOCK-END
