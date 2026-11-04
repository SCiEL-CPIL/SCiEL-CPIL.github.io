"""Second autocorrelation inequality: maximize ||h||^2_2 / (||h||_1 * ||h*h||_inf).

Discrete formulation on 50 equal intervals of [-1/4, 1/4], spacing dh = 1/20.
The continuous autoconvolution sampled at the grid is approximated by the
discrete convolution  (h*h)(x_i) = dh * sum_j h_j * h_k  over j+k = i + OFFSET,
where OFFSET = 1/2/dh = 10  (because x_j + x_k = x_i  <=>  j + k = i + 10).

The objective is scale-invariant, so we fix ||h||_2 = 1 and minimize
||h*h||_inf  (equivalently maximize  1 / (||h||_1 * ||h*h||_inf)).

Strategy:
1. Use a smooth max (log-sum-exp) surrogate for the autoconvolution peak to
   enable gradient-based optimization.
2. Parameterize h = |g|^2 where g is complex, ensuring non-negativity and
   providing a smooth manifold for optimization.
3. Generate diverse initializations including sparse/Sidon-like patterns,
   Gaussian lobes, and DFT-based structures.
4. Run L-BFGS-B optimization on the complex parameters with a time budget.
5. Final polish with perturbations around the best solution.
"""

import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # 1/2 / DH

# Wall-clock deadline with safety margin (evaluator kills at 150 s).
_DEADLINE = 145.0
_SAFETY = 10.0
_start = time.time()


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()


def _autoconv(h):
    """Discrete autoconvolution sampled on the same grid (length N)."""
    h = np.asarray(h, dtype=float)
    full = np.convolve(h, h, mode='full')  # length 2N-1, index j+k
    # (h*h)(x_i) uses j+k = i + OFFSET  ->  full[i + OFFSET]
    out = full[OFFSET:OFFSET + N]
    return DH * out


def _ratio(h):
    """Return the objective value R = ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 0.0
    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 0.0:
        return 0.0
    return l2sq / (l1 * acmax)


def _smooth_max(x, sigma=0.005):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max == 0:
        return 0.0
    x_shifted = x - x_max
    log_sum_exp = x_max + np.log(np.sum(np.exp(x_shifted / sigma))) * sigma
    return log_sum_exp


def _objective_complex_flat(g_flat):
    """
    Objective to minimize: negative of the ratio.
    g_flat is a real vector of length 2*N representing complex g.
    h = |g|^2.
    """
    g = g_flat.reshape(N, 2)
    h = g[:, 0]**2 + g[:, 1]**2
    
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 1e12, np.zeros_like(g_flat)
    
    h_norm = h / l1  # normalize to unit L1 for stability
    
    l2sq = np.sum(h_norm**2)
    if l2sq <= 1e-15:
        return 1e12, np.zeros_like(g_flat)
    
    ac = _autoconv(h_norm)
    peak = _smooth_max(ac, sigma=0.005)
    
    if peak <= 1e-15:
        return 1e12, np.zeros_like(g_flat)
    
    # R = l2sq / (l1 * peak) but h_norm has l1=1, so R = l2sq / peak
    # But we need the actual ratio with original h.
    # Since R is scale-invariant, R(h) = R(h_norm).
    R = l2sq / peak
    if R <= 1e-15:
        return 1e12, np.zeros_like(g_flat)
    
    val = -math.log(R)
    
    # Compute gradient using numerical differentiation for robustness
    # (analytic gradient is complex; numerical is safer here)
    eps = 1e-7
    grad = np.zeros_like(g_flat)
    for i in range(len(g_flat)):
        g_plus = g_flat.copy()
        g_minus = g_flat.copy()
        g_plus[i] += eps
        g_minus[i] -= eps
        f_plus = _objective_complex_flat(g_plus)[0]
        f_minus = _objective_complex_flat(g_minus)[0]
        grad[i] = (f_plus - f_minus) / (2 * eps)
    
    return val, grad


def _optimize_complex(g0, time_budget):
    """Run L-BFGS-B optimization on complex parameters g."""
    best_g = g0.copy()
    best_h = g0.reshape(N, 2)[:, 0]**2 + g0.reshape(N, 2)[:, 1]**2
    best_v = _ratio(best_h)
    
    t0 = time.time()
    
    try:
        res = minimize(
            _objective_complex_flat, g0,
            method='L-BFGS-B',
            options={'maxiter': 300, 'maxfun': 1500, 'ftol': 1e-12,
                     'gtol': 1e-8, 'disp': False},
            callback=lambda x: (np.nan if (time.time() - t0) > time_budget else x),
        )
        if res.x is not None:
            g_new = res.x
            h_new = g_new.reshape(N, 2)[:, 0]**2 + g_new.reshape(N, 2)[:, 1]**2
            v_new = _ratio(h_new)
            if v_new > best_v:
                best_v = v_new
                best_g = g_new
                best_h = h_new
    except Exception:
        pass
        
    return best_h, best_v


def _make_gaussian_lobes(centers, widths, heights):
    """Create a signal from Gaussian lobes at given centers with given widths and heights."""
    x = np.linspace(-0.25, 0.25, N)
    h = np.zeros(N)
    for c, w, ht in zip(centers, widths, heights):
        if w <= 0: continue
        h += ht * np.exp(-((x - c) ** 2) / (2 * w ** 2))
    return h


def _find_sidon_set(n, target_size):
    """Find a Sidon set of size target_size in {0, ..., n-1} using greedy construction."""
    sidon = []
    for i in range(n):
        can_add = True
        # Check if i can be added to maintain Sidon property
        # All pairwise sums must be distinct
        existing_sums = set()
        for a in range(len(sidon)):
            for b in range(a, len(sidon)):
                existing_sums.add(sidon[a] + sidon[b])
        
        new_sums = set()
        for s in sidon:
            new_sums.add(i + s)
        new_sums.add(2 * i)
        
        # Check for collisions
        if new_sums & existing_sums:
            can_add = False
        
        if can_add:
            sidon.append(i)
            if len(sidon) == target_size:
                break
    
    return sidon


def construct():
    global _start
    _start = time.time()

    best_h = None
    best_v = 0.0
    
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # 1. Generate diverse initial h profiles
    
    # 1a. Gaussian lobes
    for n_lobe in [1, 2, 3, 4]:
        for trial in range(10):
            if n_lobe == 1:
                centers = [0.0]
                widths = [0.02 + 0.01 * (trial % 5)]
                heights = [1.0]
            else:
                if trial == 0:
                    centers = np.linspace(-0.2, 0.2, n_lobe)
                else:
                    centers = np.sort(rng.uniform(-0.22, 0.22, n_lobe))
                widths = rng.uniform(0.01, 0.04, n_lobe)
                heights = rng.uniform(0.5, 1.5, n_lobe)
            
            h = _make_gaussian_lobes(centers, widths, heights)
            if np.sum(h) > 0:
                h = h / np.linalg.norm(h)
                v = _ratio(h)
                if v > 0:
                    candidates.append((v, h))
    
    # 1b. Random smooth signals
    for trial in range(50):
        h_raw = rng.random(N)
        kernel = np.array([1, 2, 3, 2, 1]) / 9.0
        h_smooth = np.convolve(h_raw, kernel, mode='same')
        h = np.maximum(h_smooth, 0.0)
        if np.sum(h) > 0:
            h = h / np.linalg.norm(h)
            v = _ratio(h)
            if v > 0:
                candidates.append((v, h))
    
    # 1c. Sparse spikes (Sidon-like)
    for target_size in [3, 5, 7, 10]:
        sidon = _find_sidon_set(N, target_size)
        if len(sidon) > 0:
            s = np.zeros(N)
            for idx in sidon:
                s[idx] = 1.0
            if np.sum(s) > 0:
                s = s / np.linalg.norm(s)
                v = _ratio(s)
                if v > 0:
                    candidates.append((v, s))
    
    # 1d. DFT-based structures
    for seed in [10, 20, 30, 40, 50]:
        rng_local = np.random.RandomState(seed)
        z = np.zeros(N, dtype=complex)
        K = rng_local.randint(5, 15)
        active_bins = rng_local.choice(N, size=K, replace=False)
        z[active_bins] = rng_local.randn(K) + 1j * rng_local.randn(K)
        H = np.fft.fft(z)
        h_dft = np.abs(H) ** 2
        if np.sum(h_dft) > 0:
            h_dft = h_dft / np.linalg.norm(h_dft)
            v = _ratio(h_dft)
            if v > 0:
                candidates.append((v, h_dft))
    
    # 1e. Structured patterns
    x = np.linspace(-0.25, 0.25, N)
    patterns = [
        np.ones(N),
        np.exp(-x**2 / 0.05),
        np.exp(-x**2 / 0.2),
        np.exp(-((x - 0.15)**2) / 0.03) + np.exp(-((x + 0.15)**2) / 0.03),
        0.5 + 0.5 * np.cos(4 * np.pi * x),
        np.abs(np.sin(2 * np.pi * x * 3)) ** 2,
    ]
    for p in patterns:
        if np.sum(p) > 0:
            p = p / np.linalg.norm(p)
            v = _ratio(p)
            if v > 0:
                candidates.append((v, p))
    
    # Sort candidates
    candidates.sort(key=lambda x: -x[0])
    
    top_k = 20
    if len(candidates) > top_k:
        candidates = candidates[:top_k]
    
    if candidates:
        best_v = candidates[0][0]
        best_h = candidates[0][1].copy()
    
    # Phase 2: Optimize using complex coefficient representation
    total_time_budget = 100.0
    time_per_candidate = total_time_budget / max(len(candidates), 1)
    
    for v0, h0 in candidates:
        if _time_left() < time_per_candidate + 5.0:
            break
            
        t_budget = min(time_per_candidate, _time_left() - 10.0)
        if t_budget < 2.0:
            break
        
        # Initialize complex coefficients from h0
        g0_real = np.sqrt(h0)
        g0_imag = np.zeros(N)
        g0_flat = np.concatenate([g0_real, g0_imag])
        
        h_opt, v_opt = _optimize_complex(g0_flat, time_budget=t_budget)
        
        if v_opt > best_v:
            best_v = v_opt
            best_h = h_opt.copy()
    
    # Phase 3: Final Polish on the global best using perturbations
    if best_h is not None and _time_left() > 15.0:
        for pert_seed in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            if _time_left() < 8.0:
                break
            rng_local = np.random.RandomState(pert_seed)
            g_best_real = np.sqrt(best_h)
            g_best_imag = np.zeros(N)
            
            pert_real = g_best_real + rng_local.randn(N) * 0.03
            pert_imag = rng_local.randn(N) * 0.03
            
            g_pert_flat = np.concatenate([pert_real, pert_imag])
            
            h_pol, v_pol = _optimize_complex(g_pert_flat, time_budget=5.0)
            if v_pol > best_v:
                best_v = v_pol
                best_h = h_pol.copy()

    if best_h is None:
        best_h = np.ones(N)
    best_h = np.maximum(best_h, 0.0)
    
    return [float(x) for x in best_h]


if __name__ == '__main__':
    # quick self-test / timing
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio(h))
    print("time =", time.time() - t0)
