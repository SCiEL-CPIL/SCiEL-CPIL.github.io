import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sequence of 50 nonnegative heights to maximize
    R = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    using FFT-based autocorrelation, differential evolution, and coordinate descent.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    def compute_ratio(h):
        """Compute R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) using FFT for speed."""
        h = np.asarray(h, dtype=float)
        h = np.maximum(h, 0.0)  # enforce nonnegativity
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        if l1 < 1e-300:
            return 0.0
        # FFT-based autocorrelation
        n_fft = 2 * N
        H = np.fft.rfft(h, n=n_fft)
        autocorr = np.fft.irfft(H * np.conj(H), n=n_fft)
        conv_inf = np.max(autocorr)
        if conv_inf < 1e-300:
            return 0.0
        return l2_sq / (l1 * conv_inf)

    def neg_ratio(h):
        """Negative ratio for minimization."""
        return -compute_ratio(h)

    best_h = np.zeros(N)
    best_score = 0.0

    def update_best(h):
        nonlocal best_h, best_score
        h = np.asarray(h, dtype=float)
        h = np.maximum(h, 0.0)
        if np.sum(h) < 1e-300:
            return
        h = h / np.sum(h)
        s = compute_ratio(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()

    # --- Phase 1: Differential Evolution over structured starting points ---
    # We'll optimize over the simplex (sum h = 1, h >= 0) by parameterizing
    # with N-1 free variables and the last determined, but DE works better
    # in the full space with bounds. We'll use bounds [0, 1] and normalize.
    
    # Actually, let's use a different parameterization: optimize log-spaced
    # or just use the raw values with bounds [0,1] and normalize after.
    # But DE needs a fixed dimension. Let's use N-1 variables for the first N-1
    # and compute the last as max(0, 1 - sum). This is tricky for DE.
    
    # Better: use N variables with bounds [0,1], normalize in the objective.
    # This wastes some computation but is simpler.
    
    # Actually, for DE, let's use a simpler approach: optimize N-1 variables
    # representing the first N-1 heights, and the Nth is 1 - sum(first N-1).
    # But this can go negative. Let's just use N variables in [0,1] and normalize.
    
    bounds = [(0.0, 1.0)] * N
    
    # Generate diverse structured starting points
    def make_starts(n_starts=20):
        starts = []
        # Single spike
        for i in range(0, N, max(1, N // 10)):
            h = np.zeros(N)
            h[i] = 1.0
            starts.append(h)
        
        # 2-point clusters
        for i in range(0, N - 1, max(1, (N-1) // 10)):
            for ratio in [0.2, 0.5, 0.8]:
                h = np.zeros(N)
                h[i] = ratio
                h[i+1] = 1.0 - ratio
                starts.append(h)
        
        # Sparse random
        rng = np.random.RandomState(42)
        for _ in range(5):
            k = rng.randint(2, 8)
            idx = rng.choice(N, size=k, replace=False)
            h = np.zeros(N)
            weights = rng.dirichlet(np.ones(k))
            h[idx] = weights
            starts.append(h)
        
        # Ramp
        for alpha in [0.5, 1.0, 2.0]:
            h = np.exp(-alpha * np.arange(N))
            h = h / np.sum(h)
            starts.append(h)
        
        # Gaussian
        for sigma in [2.0, 5.0, 10.0]:
            x = np.arange(N) - N / 2
            h = np.exp(-x**2 / (2 * sigma**2))
            h = h / np.sum(h)
            starts.append(h)
        
        return starts[:n_starts]
    
    starts = make_starts(20)
    
    # Run DE from a few diverse starting points
    # Use small population and few generations to stay within time budget
    n_de_runs = 5
    for run_idx in range(n_de_runs):
        if time.time() > deadline - 30:
            break
        
        # Pick a starting point
        start_h = starts[run_idx % len(starts)].copy()
        
        # DE with small population
        try:
            result = differential_evolution(
                neg_ratio,
                bounds=bounds,
                seed=run_idx * 100 + 42,
                maxiter=50,
                popsize=8,
                tol=1e-6,
                init='sobol',
                polish=False,
                updating='deferred',
                mutation=(0.3, 0.8),
                recombination=0.8,
            )
            h_de = result.x
            h_de = np.maximum(h_de, 0.0)
            if np.sum(h_de) > 0:
                h_de = h_de / np.sum(h_de)
            update_best(h_de)
        except Exception:
            pass
    
    # --- Phase 2: Coordinate Descent Polish ---
    def coordinate_descent(h, max_iter=2000):
        """Refine h via coordinate descent on the ratio."""
        h = np.asarray(h, dtype=float).copy()
        h = np.maximum(h, 0.0)
        if np.sum(h) < 1e-300:
            return h
        h = h / np.sum(h)
        
        best_s = compute_ratio(h)
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                if h[i] < 1e-12:
                    continue
                # Try small perturbations of h[i]
                for delta in [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]:
                    h_new = h.copy()
                    h_new[i] = max(0.0, h[i] + delta)
                    # Redistribute the difference to other indices
                    diff = h[i] - h_new[i]
                    if abs(diff) < 1e-15:
                        continue
                    # Add diff to a random other index
                    j = (i + 1 + it) % N
                    h_new[j] = max(0.0, h_new[j] + diff)
                    s_new = compute_ratio(h_new)
                    if s_new > best_s + 1e-10:
                        h = h_new
                        best_s = s_new
                        improved = True
                        break
            if not improved:
                break
        return h
    
    # Polish the best solution found
    h_polished = coordinate_descent(best_h, max_iter=3000)
    update_best(h_polished)
    
    # --- Phase 3: More DE runs if time permits ---
    remaining_time = deadline - 30 - time.time()
    if remaining_time > 10:
        for run_idx in range(n_de_runs, n_de_runs + 5):
            if time.time() > deadline - 10:
                break
            start_h = starts[run_idx % len(starts)].copy()
            try:
                result = differential_evolution(
                    neg_ratio,
                    bounds=bounds,
                    seed=run_idx * 100 + 42,
                    maxiter=30,
                    popsize=6,
                    tol=1e-6,
                    init='sobol',
                    polish=False,
                    updating='deferred',
                    mutation=(0.3, 0.8),
                    recombination=0.8,
                )
                h_de = result.x
                h_de = np.maximum(h_de, 0.0)
                if np.sum(h_de) > 0:
                    h_de = h_de / np.sum(h_de)
                update_best(h_de)
            except Exception:
                pass
        
        # Final polish
        h_polished2 = coordinate_descent(best_h, max_iter=2000)
        update_best(h_polished2)
    
    # --- Phase 4: Nelder-Mead polish ---
    if time.time() < deadline - 5:
        try:
            from scipy.optimize import minimize as sp_minimize
            res = sp_minimize(
                neg_ratio,
                x0=best_h,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10},
            )
            h_nm = np.maximum(res.x, 0.0)
            if np.sum(h_nm) > 0:
                h_nm = h_nm / np.sum(h_nm)
            update_best(h_nm)
        except Exception:
            pass
    
    # Ensure nonnegativity and return
    best_h = np.maximum(best_h, 0.0)
    if np.sum(best_h) > 0:
        best_h = best_h / np.sum(best_h)
    
    return list(best_h)
# EVOLVE-BLOCK-END
