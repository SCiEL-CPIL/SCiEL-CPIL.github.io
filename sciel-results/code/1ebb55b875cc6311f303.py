"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_overlap(h):
    """
    Compute the maximum overlap of h with shifted 1-h.
    
    h is a sequence of step heights on [0, 2] (width 2 per step, N steps).
    The overlap at shift k is sum_i h[i] * (1 - h[(i+k) % N]) for k != 0.
    We want to minimize the maximum overlap over all non-zero shifts.
    
    Actually, let's think more carefully about what "overlap" means here.
    h is a function on [0, 2] with step heights h_i on intervals [2i/N, 2(i+1)/N).
    1-h is the complementary function.
    
    The overlap of h with shifted 1-h by amount s is the integral of h(x) * (1-h)(x-s) dx.
    For discrete steps, this becomes a convolution-like sum.
    
    Let me reconsider. The task says "maximum overlap of h with shifted 1-h".
    The overlap integral is ∫ h(x) (1-h)(x-t) dx for shift t.
    
    For step functions with N steps on [0,2], each of width 2/N:
    h(x) = h_i for x in [2i/N, 2(i+1)/N)
    
    The overlap at shift t can be computed. But for the discrete problem, we likely
    consider shifts that are multiples of the step width, i.e., t = 2k/N for k=0,...,N-1.
    
    At shift k (in step units), the overlap is:
    sum_i h_i * (1 - h_{(i+k) mod N}) * (2/N)
    
    But since we're minimizing, and the factor 2/N is constant, we can just minimize
    the sum part.
    
    Wait, let me re-read: "bounded between 0 and 1 with integral 1". 
    Integral of h over [0,2] is 1. If h has N steps each of width 2/N, then
    sum_i h_i * (2/N) = 1, so sum_i h_i = N/2.
    
    The interface says "sum is exactly len/2", which confirms sum_i h_i = N/2.
    
    The overlap: I'll compute max over k != 0 of sum_i h_i * (1 - h_{(i+k) mod N}).
    
    Actually, we should probably also consider k=0? At k=0, overlap = sum_i h_i(1-h_i).
    But typically "shifted" implies non-zero shift. Let me include all k=1,...,N-1.
    """
    h = np.asarray(h)
    n = len(h)
    max_overlap = 0.0
    for k in range(1, n):
        shifted = np.roll(h, -k)
        overlap = np.sum(h * (1.0 - shifted))
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with a good initial guess
    # The uniform solution h_i = 0.5 gives overlap = N * 0.5 * 0.5 = N/4 = 25 for N=100
    # We want to do better.
    
    # Try a few different initializations and pick the best
    best_h = np.full(N, 0.5)
    best_score = _compute_overlap(best_h)
    
    # Try a sinusoidal pattern
    for freq in [1, 2, 3, 5, 7, 10]:
        t = np.arange(N)
        h_trial = 0.5 + 0.3 * np.sin(2 * np.pi * freq * t / N)
        h_trial = np.clip(h_trial, 0.0, 1.0)
        # Normalize to sum to N/2
        s = np.sum(h_trial)
        if s > 0:
            h_trial = h_trial * (N / 2.0) / s
            h_trial = np.clip(h_trial, 0.0, 1.0)
            # Re-normalize after clipping
            s2 = np.sum(h_trial)
            if s2 > 0:
                h_trial = h_trial * (N / 2.0) / s2
            score = _compute_overlap(h_trial)
            if score < best_score:
                best_score = score
                best_h = h_trial.copy()
    
    # Use scipy optimize to refine
    def objective(x):
        h = np.clip(x, 0.0, 1.0)
        # Soft constraint on sum
        sum_penalty = 100.0 * (np.sum(h) - N / 2.0) ** 2
        return _compute_overlap(h) + sum_penalty
    
    def gradient(x):
        h = np.clip(x, 0.0, 1.0)
        # Numerical gradient
        eps = 1e-8
        grad = np.zeros_like(x)
        for i in range(N):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
        return grad
    
    # Try L-BFGS-B optimization
    x0 = best_h.copy()
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=[(0.0, 1.0)] * N,
            options={'maxiter': 200, 'maxfun': 5000}
        )
        if result.success or result.fun < best_score:
            h_opt = np.clip(result.x, 0.0, 1.0)
            # Ensure sum constraint
            s = np.sum(h_opt)
            if s > 0:
                h_opt = h_opt * (N / 2.0) / s
                h_opt = np.clip(h_opt, 0.0, 1.0)
            score = _compute_overlap(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() < deadline - 20.0:
        # Try a second round of optimization from a different starting point
        # Use a pattern that alternates more
        for phase in np.linspace(0, 2*np.pi, 5, endpoint=False):
            t = np.arange(N)
            h_trial = 0.5 + 0.4 * np.sin(2 * np.pi * t / N + phase)
            h_trial = np.clip(h_trial, 0.0, 1.0)
            s = np.sum(h_trial)
            if s > 0:
                h_trial = h_trial * (N / 2.0) / s
                h_trial = np.clip(h_trial, 0.0, 1.0)
                s2 = np.sum(h_trial)
                if s2 > 0:
                    h_trial = h_trial * (N / 2.0) / s2
                score = _compute_overlap(h_trial)
                if score < best_score:
                    best_score = score
                    best_h = h_trial.copy()
                    # Refine
                    try:
                        result2 = minimize(
                            objective,
                            best_h.copy(),
                            method='L-BFGS-B',
                            bounds=[(0.0, 1.0)] * N,
                            options={'maxiter': 100, 'maxfun': 2000}
                        )
                        h_opt2 = np.clip(result2.x, 0.0, 1.0)
                        s3 = np.sum(h_opt2)
                        if s3 > 0:
                            h_opt2 = h_opt2 * (N / 2.0) / s3
                            h_opt2 = np.clip(h_opt2, 0.0, 1.0)
                        score2 = _compute_overlap(h_opt2)
                        if score2 < best_score:
                            best_score = score2
                            best_h = h_opt2.copy()
                    except Exception:
                        pass
            if time.time() > deadline - 10.0:
                break
    
    # Final normalization to ensure exact sum
    s_final = np.sum(best_h)
    if s_final > 0:
        best_h = best_h * (N / 2.0) / s_final
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
# EVOLVE-BLOCK-END
