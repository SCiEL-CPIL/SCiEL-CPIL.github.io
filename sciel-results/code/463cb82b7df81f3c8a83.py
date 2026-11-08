import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 145.0     # Safety margin from 150s
RNG = np.random.RandomState(42)

def compute_value(h):
    """
    Exact evaluator metric for given heights.
    Returns the value to be maximized.
    """
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)):
        return 0.0
    h = np.maximum(0.0, h)
    if np.all(h == 0):
        return 0.0
    
    g = np.convolve(h, h)
    # g is non-negative
    norm_inf = np.max(g)
    if norm_inf <= 0:
        return 0.0
    
    sum_h = np.sum(h)
    norm_1 = (sum_h * sum_h) / (M + 1)
    if norm_1 <= 0:
        return 0.0
    
    # y = [0, g[0], ..., g[M-1], 0]
    # l2 = (W/3) * sum_{i=0}^{M} (y_i^2 + y_i*y_{i+1} + y_{i+1}^2)
    # Sum y_i^2 = sum(g^2)
    # Sum y_{i+1}^2 = sum(g^2)
    # Sum y_i*y_{i+1} = sum(g[:-1]*g[1:])
    # Total sum = 2*sum(g^2) + sum(g[:-1]*g[1:])
    
    sum_g_sq = np.sum(g * g)
    sum_g_gnext = np.sum(g[:-1] * g[1:])
    l2 = (W / 3.0) * (2.0 * sum_g_sq + sum_g_gnext)
    
    return l2 / (norm_1 * norm_inf)

def _smooth_value(h, beta=50.0):
    """
    Smooth surrogate for L-BFGS-B.
    Minimizes -value.
    Uses Log-Sum-Exp for norm_inf to provide differentiability.
    """
    h = np.maximum(0.0, h)
    if np.all(h == 0):
        return 0.0
    
    g = np.convolve(h, h)
    sum_h = np.sum(h)
    norm_1 = (sum_h * sum_h) / (M + 1)
    if norm_1 <= 0:
        return 0.0
    
    sum_g_sq = np.sum(g * g)
    sum_g_gnext = np.sum(g[:-1] * g[1:])
    l2 = (W / 3.0) * (2.0 * sum_g_sq + sum_g_gnext)
    
    # Log-sum-exp for smooth max
    g_max = np.max(g)
    if g_max <= 0:
        return 0.0
    lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
    
    if lse <= 0:
        return 0.0
        
    val = l2 / (norm_1 * lse)
    return -val

def _neg_exact_value(h):
    """
    Negative of the exact value for minimization.
    """
    return -compute_value(h)

def _generate_starts():
    x = np.linspace(0, N - 1, N)
    starts = []
    
    # 1. Uniform (Seed)
    starts.append(np.ones(N))
    
    # 2. Gaussian centered
    starts.append(np.exp(-0.5 * ((x - N / 2) / (N / 4)) ** 2))
    
    # 3. Two Peaks
    p1 = np.exp(-0.5 * ((x - N * 0.25) / (N / 8)) ** 2)
    p2 = np.exp(-0.5 * ((x - N * 0.75) / (N / 8)) ** 2)
    starts.append(p1 + p2)
    
    # 4. Semicircle
    starts.append(np.sqrt(np.maximum(0.0, 1 - ((2 * x / (N - 1) - 1)) ** 2)))
    
    # 5. Cosine bump
    starts.append(0.5 * (1 + np.cos(2 * np.pi * x / N)))
    
    # 6. Triangle
    starts.append(1 - 2 * np.abs(x - (N - 1) / 2) / (N - 1))
    
    # 7. Comb patterns
    for k in [2, 3, 5, 8, 10]:
        comb = 1.0 + 0.8 * np.cos(2 * np.pi * k * x / N)
        starts.append(np.maximum(0.1, comb))
        
    # 8. Random smooth profiles
    for _ in range(20):
        freqs = RNG.randint(1, 10, size=6)
        phases = RNG.uniform(0, 2 * np.pi, size=6)
        amps = RNG.uniform(0.5, 1.0, size=6)
        prof = np.zeros(N)
        for f_, p_, a_ in zip(freqs, phases, amps):
            prof += a_ * np.cos(2 * np.pi * f_ * x / N + p_)
        starts.append(np.maximum(0.1, prof))
        
    # 9. Random binary-like sequences (smoothed)
    for _ in range(15):
        binary = (RNG.rand(N) > 0.5).astype(float)
        kernel = np.ones(3) / 3.0
        smooth_binary = np.convolve(binary, kernel, mode='same')
        starts.append(np.maximum(0.1, smooth_binary))
        
    # 10. Symmetric random
    for _ in range(10):
        half = RNG.rand(N // 2)
        full = np.concatenate([half, half[::-1]])
        starts.append(np.maximum(0.1, full))

    return starts

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_val = 0.0
    best_h = np.ones(N)

    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        if not np.all(np.isfinite(h)):
            return
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    starts = _generate_starts()
    
    # Initial evaluation of starts
    for s in starts:
        track(s)

    bounds = [(0, None)] * N
    
    # Optimization Phase: Hybrid L-BFGS-B + Nelder-Mead
    # We try multiple betas to balance smoothness and accuracy
    betas = [20.0, 50.0, 100.0]
    
    for i, s in enumerate(starts):
        if time.time() > deadline - 10:  # Reserve time for final checks
            break
            
        x0 = np.maximum(0.0, np.asarray(s, dtype=float))
        
        # Try different betas for smooth optimization
        x_curr = x0
        for beta in betas:
            if time.time() > deadline - 10:
                break
            try:
                res_smooth = opt.minimize(
                    lambda x, b=beta: _smooth_value(x, b),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                # Track the result of smooth optimization
                track(res_smooth.x)
                # Use the best result so far as starting point for next beta or exact
                if -res_smooth.fun > best_val:
                    x_curr = res_smooth.x
            except Exception:
                pass

        # Step 2: Exact optimization with Nelder-Mead to handle the kink in max
        # Nelder-Mead is derivative-free and robust to non-smoothness
        try:
            res_exact = opt.minimize(
                _neg_exact_value,
                x_curr,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            x_final = res_exact.x
        except Exception:
            x_final = x_curr

        track(x_final)

    # If time remains, do a few more random restarts or perturbations
    # This helps escape local optima if the first pass didn't find the global one
    while time.time() < deadline - 5:
        # Perturb the current best
        sigma = 0.1
        pert = best_h + RNG.normal(0.0, sigma, N)
        pert = np.maximum(0.0, pert)
        
        # Quick local refinement with high beta smooth objective
        try:
            res = opt.minimize(
                lambda x: _smooth_value(x, 200.0),
                pert,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            track(res.x)
        except Exception:
            pass
            
        # And also try exact Nelder-Mead on the perturbation
        try:
            res2 = opt.minimize(
                _neg_exact_value,
                pert,
                method='Nelder-Mead',
                options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            track(res2.x)
        except Exception:
            pass

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
