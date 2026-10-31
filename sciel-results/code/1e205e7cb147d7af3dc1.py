"""
Strategy: Structured parametric search + basis-expansion local refinement.

The objective R(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf) is scale invariant.
We:
  1. Do a coarse parametric search over several shape families (Gaussian, Cauchy,
     exponential, triangular, bimodal, U-shape, etc.) to find a good starting point.
  2. Then perform a local search using a small basis (e.g., first few cosine modes
     on the interval) to refine the height vector, which captures smooth deviations
     from the parametric family without needing to optimize all 50 coordinates directly.
"""
import numpy as np
import time
from itertools import product

N = 50
# Equal intervals on [-1/4, 1/4]: 50 points
# x_i = -1/4 + (i + 0.5) * (1/2) / 50 = -0.25 + (i+0.5)*0.01
# Or simply: x_i = -0.25 + i * (0.5/49) for i in 0..49? 
# The problem says "50 nonnegative heights on equal intervals of [-1/4,1/4]".
# This typically means 50 bins of width 1/100, centered at -1/4 + (i+0.5)/100.
# But for the discrete convolution, the actual x values don't matter, only the sequence h[0..49].
# The convolution is over indices. So x only matters for shaping the sequence.

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-15):
        h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    l2_sq = np.sum(h * h)
    hh = np.convolve(h, h, mode='full')
    hh_max = np.max(hh)
    if hh_max < 1e-300:
        return 0.0
    return l2_sq / (l1 * hh_max)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Index grid for shaping
    idx = np.arange(N, dtype=np.float64)
    center = (N - 1) / 2.0
    # Normalize x to [-1, 1]
    x = (idx - center) / center if center > 0 else np.zeros(N)

    best_h = np.ones(N, dtype=np.float64)
    best_score = compute_ratio(best_h)

    # --- Phase 1: Coarse parametric search ---
    a_values = np.logspace(-2, 3, 300)  # Width parameter from 0.01 to 1000

    def make_shape(name, a, **kwargs):
        if name == 'gauss':
            return np.exp(-a * x**2)
        elif name == 'cauchy':
            return 1.0 / (1.0 + a * x**2)
        elif name == 'exp':
            return np.exp(-a * np.abs(x))
        elif name == 'tri':
            return np.maximum(0.0, 1.0 - a * np.abs(x))
        elif name == 'uniform':
            return np.ones(N)
        elif name == 'ushape':
            return 1.0 + a * x**2
        elif name == 'bimodal':
            c = kwargs.get('c', 0.4)
            return np.exp(-a * (x - c)**2) + np.exp(-a * (x + c)**2)
        elif name == 'ramp':
            return np.linspace(0.1, 1.0, N) ** a
        else:
            return np.ones(N)

    shape_names = ['gauss', 'cauchy', 'exp', 'tri', 'uniform', 'ushape', 'bimodal', 'ramp']
    # For bimodal, also vary the center
    bimodal_centers = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    for name in shape_names:
        if time.time() > deadline - 20.0:
            break
        for a in a_values:
            if name == 'bimodal':
                for c in bimodal_centers:
                    h_raw = make_shape(name, a, c=c)
                    h_raw = np.maximum(h_raw, 0.0)
                    s = np.sum(h_raw)
                    if s < 1e-300:
                        continue
                    h_norm = h_raw / s
                    score = compute_ratio(h_norm)
                    if score > best_score:
                        best_score = score
                        best_h = h_norm.copy()
            else:
                h_raw = make_shape(name, a)
                h_raw = np.maximum(h_raw, 0.0)
                s = np.sum(h_raw)
                if s < 1e-300:
                    continue
                h_norm = h_raw / s
                score = compute_ratio(h_norm)
                if score > best_score:
                    best_score = score
                    best_h = h_norm.copy()

    # --- Phase 2: Local refinement using a cosine basis ---
    # We represent h as a non-negative linear combination of a small set of basis functions.
    # This reduces the search to optimizing a few coefficients.
    
    # Build basis: constant + first M cosine modes
    M = 12  # Number of cosine modes (plus constant = M+1 basis vectors)
    basis = np.zeros((N, M + 1))
    basis[:, 0] = 1.0  # Constant mode
    for m in range(1, M + 1):
        # Cosine modes on the interval, normalized
        basis[:, m] = np.cos(m * np.pi * (idx + 0.5) / N)
    
    # We'll search for coefficients c such that h = basis @ c >= 0, and sum(h) = 1.
    # This is a constrained optimization. We'll use a simple projected gradient or
    # just do a random search over c with clipping.
    
    # Start from the best_h found in Phase 1.
    # Project best_h onto the basis to get initial coefficients.
    # Use least squares: c0 = (B^T B)^{-1} B^T h
    B = basis
    try:
        c0 = np.linalg.lstsq(B, best_h, rcond=None)[0]
    except:
        c0 = np.zeros(M + 1)
        c0[0] = 1.0 / N

    # Ensure initial h is non-negative
    h_init = B @ c0
    h_init = np.maximum(h_init, 0.0)
    s_init = np.sum(h_init)
    if s_init > 1e-300:
        h_init = h_init / s_init
    c_init = np.linalg.lstsq(B, h_init, rcond=None)[0]

    # Local search: perturb c, clip h to non-negative, normalize, evaluate.
    rng = np.random.default_rng(12345)
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002]
    current_c = c_init.copy()
    current_h = B @ current_c
    current_h = np.maximum(current_h, 0.0)
    s_cur = np.sum(current_h)
    if s_cur > 1e-300:
        current_h = current_h / s_cur
    current_score = compute_ratio(current_h)

    for it in range(5000):
        if time.time() > deadline - 5.0:
            break
        ss = step_sizes[min(4, it // 1000)]
        delta_c = rng.normal(0, ss, M + 1)
        new_c = current_c + delta_c
        new_h = B @ new_c
        new_h = np.maximum(new_h, 0.0)
        s_new = np.sum(new_h)
        if s_new < 1e-300:
            continue
        new_h = new_h / s_new
        score = compute_ratio(new_h)
        if score > current_score:
            current_score = score
            current_c = new_c.copy()
            current_h = new_h.copy()
            if score > best_score:
                best_score = score
                best_h = new_h.copy()

    # --- Phase 3: Coordinate ascent on the full 50-dim vector ---
    # Simple local search: for each coordinate, try small perturbations.
    # This is slow but can find fine-grained improvements.
    # We'll do a few rounds of this with small steps.
    
    h_work = best_h.copy()
    score_work = best_score
    
    for round_num in range(3):
        if time.time() > deadline - 3.0:
            break
        # Try perturbations on each coordinate
        step = 0.01 / (round_num + 1)
        for i in range(N):
            if time.time() > deadline - 3.0:
                break
            # Try increasing
            h_trial = h_work.copy()
            h_trial[i] += step
            s_t = np.sum(h_trial)
            if s_t > 0:
                h_trial = h_trial / s_t
                score_t = compute_ratio(h_trial)
                if score_t > score_work:
                    score_work = score_t
                    h_work = h_trial.copy()
            # Try decreasing
            h_trial = h_work.copy()
            h_trial[i] = max(0.0, h_trial[i] - step)
            s_t = np.sum(h_trial)
            if s_t > 0:
                h_trial = h_trial / s_t
                score_t = compute_ratio(h_trial)
                if score_t > score_work:
                    score_work = score_t
                    h_work = h_trial.copy()
        
        if score_work > best_score:
            best_score = score_work
            best_h = h_work.copy()

    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
