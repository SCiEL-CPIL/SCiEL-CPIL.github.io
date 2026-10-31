"""Combined program: structured initializations + L-BFGS-B refinement with deadline guard."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 135.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    
    def elapsed():
        return time.time() - start
    
    def compute_ratio(h):
        """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
        h = np.asarray(h, dtype=np.float64)
        l1 = np.sum(h)
        if l1 < 1e-300:
            return 0.0
        l2_sq = np.sum(h * h)
        hh = np.convolve(h, h, mode='full')
        hh_max = np.max(hh)
        if hh_max < 1e-300:
            return 0.0
        return l2_sq / (l1 * hh_max)
    
    def neg_ratio(params):
        """Negative ratio for minimization."""
        h = np.exp(params)  # ensure nonnegativity via log-space
        return -compute_ratio(h)
    
    # Generate structured initial candidates
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Triangular
    t = np.linspace(0, 1, N)
    candidates.append(np.maximum(0, 1 - 2 * np.abs(t - 0.5)))
    
    # Gaussian-like
    x = np.linspace(-3, 3, N)
    candidates.append(np.exp(-x**2 / 2))
    
    # Sine bump
    candidates.append(np.maximum(0, np.sin(np.pi * np.linspace(0, 1, N))))
    
    # Half-sine
    candidates.append(np.maximum(0, np.sin(np.pi / 2 * np.linspace(0, 1, N))))
    
    # Cosine bump
    candidates.append(np.cos(np.pi * (np.linspace(0, 1, N) - 0.5)) ** 2)
    
    # Plateau
    plateau = np.zeros(N)
    plateau[N//4:3*N//4] = 1.0
    candidates.append(plateau)
    
    # Two peaks
    two_peaks = np.zeros(N)
    two_peaks[N//8:3*N//8] = 1.0
    two_peaks[5*N//8:7*N//8] = 1.0
    candidates.append(two_peaks)
    
    # Random sparse
    rng = np.random.RandomState(42)
    for _ in range(5):
        h = np.zeros(N)
        idx = rng.choice(N, size=N//5, replace=False)
        h[idx] = 1.0
        candidates.append(h)
    
    # Evaluate initial candidates
    best_h = None
    best_val = -np.inf
    
    for h in candidates:
        h = np.asarray(h, dtype=np.float64)
        if np.max(h) < 1e-10:
            continue
        val = compute_ratio(h)
        if val > best_val:
            best_val = val
            best_h = h.copy()
    
    if best_h is None:
        return [1.0] * N
    
    # Convert to log-space for optimization
    best_h_log = np.log(np.maximum(best_h, 1e-10))
    
    # L-BFGS-B refinement from best initial
    best_params = best_h_log.copy()
    best_score = best_val
    
    # Try multiple random restarts if time permits
    rng = np.random.RandomState(123)
    n_restarts = 10
    
    for i in range(n_restarts):
        if elapsed() > DEADLINE:
            break
        
        # Generate random starting point
        if i == 0:
            params = best_h_log.copy()
        else:
            # Mix of structured and random
            if i % 2 == 0:
                # Perturb best
                params = best_h_log + 0.5 * rng.randn(N)
            else:
                # Random log-uniform
                params = rng.uniform(-2, 2, size=N)
        
        try:
            result = minimize(
                neg_ratio,
                params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 10000}
            )
            
            if result.success or result.fun < -best_score:
                h_opt = np.exp(result.x)
                val_opt = -result.fun
                if val_opt > best_score:
                    best_score = val_opt
                    best_params = result.x.copy()
        except Exception:
            continue
    
    # Final evaluation
    final_h = np.exp(best_params)
    final_val = compute_ratio(final_h)
    
    # Ensure nonnegativity and return
    final_h = np.maximum(final_h, 0.0)
    if np.sum(final_h) < 1e-10:
        return [1.0] * N
    
    return final_h.tolist()
# EVOLVE-BLOCK-END
