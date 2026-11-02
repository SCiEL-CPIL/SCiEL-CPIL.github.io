import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _autoconvolution(h):
    """Compute autoconvolution of step heights. Returns 2*N-1 values."""
    n = len(h)
    m = 2 * n - 1
    ac = np.zeros(m)
    # ac[i] corresponds to offset (i - (n-1))
    for j in range(-(n-1), n):
        if j >= 0:
            s = np.dot(h[:n-j], h[j:])
        else:
            s = np.dot(h[-j:], h[:n+j])
        ac[j + (n-1)] = s
    return ac

def _objective_smooth(h, alpha=50.0):
    """Negative of the ratio, with smooth max for sup norm."""
    h = np.maximum(h, 0.0)
    if np.all(h == 0):
        return 1e10
    l2_sq = np.dot(h, h)
    l1 = np.sum(h)
    ac = _autoconvolution(h)
    sup_ac = np.max(ac)
    if sup_ac <= 0:
        return 1e10
    # ratio = l2_sq / (l1 * sup_ac)
    # We want to maximize ratio, so minimize -ratio
    # Use smooth max for differentiability
    # Actually, let's just use the raw max since L-BFGS-B handles it
    ratio = l2_sq / (l1 * sup_ac)
    return -ratio

def _construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    best_h = None
    best_score = -1e10
    
    def evaluate(h):
        h = np.maximum(h, 0.0)
        if np.all(h == 0):
            return -1e10
        l2_sq = np.dot(h, h)
        l1 = np.sum(h)
        ac = _autoconvolution(h)
        sup_ac = np.max(ac)
        if sup_ac <= 0:
            return -1e10
        return l2_sq / (l1 * sup_ac)
    
    def try_optimize(h0, max_iter=200):
        nonlocal best_h, best_score
        h0 = np.maximum(h0, 0.0)
        try:
            res = minimize(
                _objective_smooth, h0, method='L-BFGS-B',
                bounds=[(0.0, None)] * N,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_res = np.maximum(res.x, 0.0)
            score = evaluate(h_res)
            if score > best_score:
                best_score = score
                best_h = h_res.copy()
        except Exception:
            pass
    
    # Diverse structured initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N))
    
    # 2. Triangular (tent)
    t = np.linspace(-1, 1, N)
    inits.append(np.maximum(0, 1 - np.abs(t)))
    
    # 3. Bump (Gaussian centered)
    x = np.linspace(-1, 1, N)
    inits.append(np.exp(-2 * x**2))
    
    # 4. Power law (decay from center)
    inits.append(np.maximum(0, 1 - np.abs(x))**3)
    
    # 5. Two bumps
    inits.append(np.exp(-8*(x-0.4)**2) + np.exp(-8*(x+0.4)**2))
    
    # 6. Periodic
    inits.append(0.5 + 0.5 * np.cos(4 * np.pi * x))
    
    # 7. Power law from edge
    inits.append(np.linspace(1, 0.1, N))
    
    # 8. Inverted power law
    inits.append(np.linspace(0.1, 1, N))
    
    # 9. Flat with small center bump
    inits.append(np.ones(N) * 0.5 + 0.5 * np.exp(-20 * x**2))
    
    # 10. Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        inits.append(rng.random(N) + 0.1)
    
    # 11. Cosine raised
    inits.append(0.5 + 0.5 * np.cos(np.pi * x))
    
    # 12. Sinc-like
    inits.append(np.maximum(0.01, np.sinc(0.5 * x)))
    
    # 13. Plateau
    inits.append(np.where(np.abs(x) < 0.3, 1.0, 0.3))
    
    # 14. Sharp center
    inits.append(np.where(np.abs(x) < 0.1, 1.0, 0.05))
    
    # 15. Quadratic
    inits.append(1 - x**2)
    
    for i, h0 in enumerate(inits):
        if time.time() > deadline:
            break
        try_optimize(h0, max_iter=150)
    
    # Coordinate descent refinement on best
    if best_h is not None:
        if time.time() < deadline:
            # Local coordinate descent
            h = best_h.copy()
            for coord_iter in range(2000):
                if time.time() > deadline:
                    break
                improved = False
                for j in range(N):
                    old_h = h[j]
                    # Try a few perturbations
                    for delta in [0.01, 0.05, 0.1, 0.5, -0.01, -0.05, -0.1, -0.5]:
                        h_new = h.copy()
                        h_new[j] = max(0, old_h + delta)
                        score_new = evaluate(h_new)
                        if score_new > best_score:
                            best_score = score_new
                            best_h = h_new.copy()
                            h = h_new.copy()
                            improved = True
                            break
                if not improved:
                    break
    
    # Final: also try scaling to normalize (ratio is scale-invariant, so this shouldn't matter,
    # but numerical issues might)
    if best_h is not None:
        # Normalize so max is 1
        mx = np.max(best_h)
        if mx > 0:
            best_h = best_h / mx
    
    if best_h is None:
        best_h = np.ones(N)
    
    return list(best_h)

def construct():
    return _construct()
