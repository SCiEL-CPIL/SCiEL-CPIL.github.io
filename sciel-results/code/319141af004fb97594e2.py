"""Minimize the normalized maximum of the autoconvolution of a nonnegative
step function on 600 equal intervals over [-1/4, 1/4].

Approach: Use multiple asymmetric initializations + simulated annealing to
escape the uniform critical point and find shapes with lower autoconvolution
peak by reducing overlap of f(t) and f(-t).
"""

import time
import math
import random

import numpy as np

N = 600
DELTA = 0.5 / N
DEADLINE = 145.0  # safety margin before 150s kill

_t0 = time.time()


def _time_left():
    return DEADLINE - (time.time() - _t0)


def _midpoints():
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + DELTA / 2.0
    return x


def _autoconv_peak(h):
    """Compute max of discrete autoconvolution using FFT for speed."""
    h = np.asarray(h, dtype=float)
    # Use FFT-based convolution for efficiency
    size = 2 * N
    f = np.fft.rfft(h, n=size)
    c = np.fft.irfft(f * f, n=size)
    return float(np.max(c))


def _objective_l1(h):
    """max(autoconv) / (sum h)^2 -- the target normalization."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s <= 0:
        return float("inf")
    return _autoconv_peak(h) / (s * s)


def _objective(h):
    return _objective_l1(h)


def _gradient_objective(h):
    """Compute gradient of max(autoconv) / (sum h)^2."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s <= 0:
        return np.zeros_like(h)
    
    # Compute autoconvolution
    size = 2 * N
    f = np.fft.rfft(h, n=size)
    c = np.fft.irfft(f * f, n=size)
    
    m_star = int(np.argmax(c))
    peak = float(c[m_star])
    
    # Gradient of max(c) w.r.t. h
    # c[k] = sum_i h[i] * h[i + k - (N-1)]
    # d(c[m_star])/d(h[j]) = sum over valid i where i + (m_star - (N-1)) = j
    # i.e., h[j - (m_star - (N-1))]
    offset = m_star - (N - 1)
    g = np.zeros_like(h)
    lo = max(0, offset)
    hi = min(N, N + offset)
    if offset >= 0:
        g[lo:hi] = h[lo - offset:hi - offset]
    else:
        g[lo:hi] = h[lo - offset:hi - offset]
    
    # Quotient rule: d(peak/s^2) = (g*s^2 - peak*2*s*h) / s^4
    grad = (g * s * s - peak * 2.0 * s * h) / (s ** 4)
    return grad


def _profile_asymmetric(seed):
    """Generate an asymmetric profile with random perturbations."""
    rng = np.random.RandomState(seed)
    h = np.ones(N)
    # Add random smooth perturbations
    for freq in [1, 2, 3, 5, 8]:
        amp = rng.uniform(0.1, 0.4)
        phase = rng.uniform(0, 2 * np.pi)
        x = np.linspace(0, 1, N)
        h += amp * np.sin(2 * np.pi * freq * x + phase)
    # Add some random noise
    h += rng.normal(0, 0.05, N)
    h = np.clip(h, 0.01, None)
    # Normalize
    h = h / np.sum(h) * N
    return h


def _simulated_annealing(h, max_iters=200, t0=1.0, t1=0.01):
    """Simulated annealing to fine-tune the profile."""
    h = h.copy()
    best_h = h.copy()
    best_val = _objective(h)
    
    for it in range(max_iters):
        if _time_left() <= 5.0:
            break
        
        T = t0 * (t1 / t0) ** (it / max_iters)
        
        # Generate a random perturbation
        rng = np.random.RandomState(it * 12345 + 6789)
        h_new = h.copy()
        
        # Randomly perturb a few elements
        n_perturb = rng.randint(5, 20)
        indices = rng.choice(N, size=n_perturb, replace=False)
        for idx in indices:
            delta = rng.normal(0, 0.1 * np.mean(h))
            h_new[idx] = max(0.0, h_new[idx] + delta)
        
        # Preserve total mass approximately
        s_old = float(np.sum(h))
        s_new = float(np.sum(h_new))
        if s_new > 1e-9:
            h_new = h_new * (s_old / s_new)
        
        v_new = _objective(h_new)
        
        # Accept if better, or with probability exp(-delta/T)
        if v_new < best_val:
            h = h_new
            best_h = h.copy()
            best_val = v_new
        else:
            delta = v_new - best_val
            if delta > 0 and rng.random() < math.exp(-delta / (T + 1e-10)):
                h = h_new
    
    return best_h, best_val


def construct():
    global _t0
    _t0 = time.time()
    
    uniform = np.ones(N)
    best_h = uniform.copy()
    best_val = _objective(uniform)
    
    # --- 1) Try multiple asymmetric initializations ---
    n_initials = 20
    for seed in range(n_initials):
        if _time_left() <= 10.0:
            break
        
        h = _profile_asymmetric(seed)
        v = _objective(h)
        if v < best_val - 1e-12:
            best_val = v
            best_h = h.copy()
        
        # Simulated annealing from this initialization
        if _time_left() <= 15.0:
            break
        h_sa, v_sa = _simulated_annealing(h, max_iters=100)
        if v_sa < best_val - 1e-12:
            best_val = v_sa
            best_h = h_sa.copy()
    
    # --- 2) Gradient descent from best found so far ---
    if _time_left() > 20.0:
        try:
            h = best_h.copy()
            lr = 0.05
            for it in range(200):
                if _time_left() <= 10.0:
                    break
                
                grad = _gradient_objective(h)
                grad_norm = float(np.max(np.abs(grad))) + 1e-12
                h_new = np.clip(h - lr * grad / grad_norm, 0.0, None)
                
                # Preserve total mass
                s_old = float(np.sum(h))
                s_new = float(np.sum(h_new))
                if s_new > 1e-9:
                    h_new = h_new * (s_old / s_new)
                
                v = _objective(h_new)
                if v < best_val - 1e-12:
                    best_val = v
                    best_h = h_new.copy()
                h = h_new
                lr *= 0.995
        except Exception:
            pass
    
    # --- 3) Final simulated annealing polish ---
    if _time_left() > 20.0:
        h_final, v_final = _simulated_annealing(best_h, max_iters=150)
        if v_final < best_val - 1e-12:
            best_val = v_final
            best_h = h_final.copy()
    
    # --- 4) Safety: never return worse than uniform ---
    if _objective(best_h) > _objective(uniform) + 1e-9:
        best_h = uniform

    best_h = np.clip(np.asarray(best_h, dtype=float), 0.0, None)
    return [float(v) for v in best_h]


_t0 = time.time()
