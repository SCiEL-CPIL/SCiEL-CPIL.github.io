"""
Strategy: Global stochastic search using Differential Evolution on the exact
non-smooth objective function, followed by local Nelder-Mead polishing.
This avoids the pitfalls of smooth surrogates (like log-sum-exp) which can
mislead gradient-based optimizers near the non-differentiable peak, and
exploits DE's robustness to non-convexity and lack of gradient requirements.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 150.0
RNG = np.random.RandomState(50)

def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y

def compute_value(h):
    """Exact evaluator metric for given heights (defensive)."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _neg_value(h):
    """Negative of the value for minimization algorithms."""
    return -compute_value(h)

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 10.0) # 10s safety margin

    best_val = 0.0
    best_h = np.ones(N)
    
    # Seed with a few diverse initial guesses to ensure DE has a good starting population
    # and to establish a baseline.
    initial_candidates = []
    initial_candidates.append(np.ones(N))
    # Gaussian bump
    x = np.linspace(0, N - 1, N)
    initial_candidates.append(np.exp(-0.5 * ((x - N / 2) / (N / 4)) ** 2))
    # Two peaks
    p1 = np.exp(-0.5 * ((x - N * 0.25) / (N / 8)) ** 2)
    p2 = np.exp(-0.5 * ((x - N * 0.75) / (N / 8)) ** 2)
    initial_candidates.append(p1 + p2)
    # Semicircle
    initial_candidates.append(np.sqrt(np.maximum(0.0, 1 - ((2 * x / (N - 1) - 1)) ** 2)))
    
    for h in initial_candidates:
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    # Differential Evolution Phase
    # DE is a population-based method that works well for non-smooth, non-convex problems.
    # We use a relatively small population to fit within the time limit, but DE's
    # global exploration is often more effective than local gradient methods for this problem.
    
    bounds = [(0.0, 10.0)] * N # Heights are non-negative; 10 is a generous upper bound
    
    def callback(xk, convergence):
        nonlocal best_val, best_h
        v = compute_value(xk)
        if v > best_val:
            best_val = v
            best_h = xk.copy()
        return time.time() > deadline

    # We will run DE in stages if time permits, but typically one good run is enough.
    # maxiter is set high, but the callback will stop it if time runs out.
    # popsize=10 means 500 individuals evaluated per generation.
    # With N=50, each evaluation is fast (convolution).
    
    try:
        res = opt.differential_evolution(
            _neg_value,
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.9,
            seed=50,
            polish=False, # We will do our own polish to control time precisely
            callback=callback,
            init='latinhypercube'
        )
        if res.fun < 0: # If we found a valid solution
            v = -res.fun
            if v > best_val:
                best_val = v
                best_h = res.x.copy()
    except Exception:
        pass

    # Polish Phase: Nelder-Mead on the best candidate
    # NM is derivative-free and good for final refinement of non-smooth functions.
    if time.time() < deadline - 5.0:
        try:
            # Use a tight initial simplex around the best found
            # We scale the perturbation based on the magnitude of best_h
            scale = np.max(np.abs(best_h)) * 0.1 + 1e-6
            x0 = best_h.copy()
            
            # Nelder-Mead can be sensitive to initial simplex size.
            # We'll run it for a limited number of iterations.
            res_nm = opt.minimize(
                _neg_value,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if res_nm.fun < 0:
                v = -res_nm.fun
                if v > best_val:
                    best_val = v
                    best_h = res_nm.x.copy()
        except Exception:
            pass

    # Final safety check and return
    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    
    # One last check to ensure we return the absolute best we've seen
    final_val = compute_value(best_h)
    if final_val > best_val:
        best_val = final_val

    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
