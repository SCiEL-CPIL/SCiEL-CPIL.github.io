"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Strategy: Global search via Differential Evolution (DE) to escape local minima,
followed by local refinement using Nelder-Mead or coordinate descent.
This avoids reliance on gradient information which can be misleading for non-convex
quadratic max-min problems."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
_DEADLINE = time.time() + 145  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    # Zero-pad to avoid circular convolution issues
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    # Distribute the difference evenly, but clip
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    # If clipping changed the sum, do a second pass
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    # Final tiny correction on one element to ensure exact sum in floating point
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h


def _objective_for_de(x):
    """
    Objective function for Differential Evolution.
    Takes raw x, projects to valid space, returns score.
    """
    h = _fix_sum(x)
    return _score(h)


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            # print(f"New best: {s:.6f} at t={time.time()-t0:.2f}s")

    # Initial structured guesses
    i = np.arange(N)
    candidates = []
    
    # 1. Constant
    candidates.append(np.full(N, 0.5))
    
    # 2. Cosine
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    candidates.append(_fix_sum(h_cos))
    
    # 3. Sine
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    candidates.append(_fix_sum(h_sin))
    
    # 4. Random
    for _ in range(5):
        h_rand = np.random.uniform(0, 1, N)
        candidates.append(_fix_sum(h_rand))
        
    for h in candidates:
        _update_best(h)
        
    # Phase 1: Differential Evolution (Global Search)
    # DE is robust for non-convex problems.
    # Bounds: [0, 1] for each h_i.
    # Constraint: sum(h) = N/2. DE doesn't handle equality constraints well directly,
    # but we project inside the objective. This works but might be inefficient.
    # Alternative: Use scipy.optimize.minimize with method='Nelder-Mead' on the best DE result.
    
    if time.time() - t0 < _DEADLINE - 60:
        try:
            # Use a small population for speed, but enough for diversity
            # maxiter is limited by time
            result = differential_evolution(
                _objective_for_de,
                bounds=[(0.0, 1.0)] * N,
                seed=42,
                maxiter=50,  # Keep low, we have time for local polish
                popsize=10,
                tol=1e-8,
                polish=False,  # We will do our own polish
                init='latinhypercube'
            )
            if result.fun < best_score:
                _update_best(result.x)
        except Exception as e:
            pass

    # Phase 2: Local Refinement with Nelder-Mead
    # Nelder-Mead is derivative-free and works well for low-dimensional non-smooth problems.
    # It respects bounds via projection or careful step sizes.
    
    if best_h is not None and time.time() - t0 < _DEADLINE - 20:
        # Define objective for Nelder-Mead that projects
        def _nm_objective(x):
            h = _fix_sum(x)
            return _score(h)
            
        # Start from best found so far
        x0 = best_h.copy()
        
        # Try a few random restarts for local optimization if time permits
        local_starts = [x0]
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 30:
                break
            # Perturb best solution
            pert = best_h + np.random.normal(0, 0.05, N)
            local_starts.append(_fix_sum(pert))
            
        for start in local_starts:
            if time.time() - t0 > _DEADLINE - 25:
                break
            try:
                res = minimize(
                    _nm_objective,
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                if res.fun < best_score:
                    _update_best(res.x)
            except Exception:
                pass

    # Phase 3: Coordinate Descent Polish
    # Very fast, can run many iterations.
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h = best_h.copy()
        # Simple coordinate descent: update one variable at a time to minimize score
        # For each j, find h_j in [0,1] that minimizes score, keeping others fixed.
        # This is a 1D optimization.
        
        for _ in range(500):  # Max passes
            if time.time() - t0 > _DEADLINE - 10:
                break
            improved = False
            for j in range(N):
                if time.time() - t0 > _DEADLINE - 10:
                    break
                # Save original
                orig_hj = h[j]
                
                # Binary search or golden section on h_j
                # Since N=100, evaluating score is ~0.1ms. 
                # We can do a fine grid or binary search.
                # The function is not convex in h_j, so use a small grid.
                # Or use scipy minimize_scalar
                try:
                    def _obj_1d(x):
                        h[j] = x
                        return _score(h)
                    
                    res = minimize(_obj_1d, orig_hj, method='Nelder-Mead', 
                                   options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 100})
                    if res.fun < _score(h):
                        h[j] = res.x[0]
                        improved = True
                except Exception:
                    pass
                    
            if not improved:
                break
                
        _update_best(h)

    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _fix_sum(best_h)
    return best_h
