import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 600
# Use symmetry: h is even, so we only need to optimize N//2 = 300 variables
# h[i] = h[N-1-i] for i in range(N)
# Actually, let's think about this more carefully.
# The step function has 600 equal intervals on [-1/4, 1/4].
# For the autocorrelation to be well-defined and for symmetry, 
# we can assume h is symmetric: h[i] = h[N-1-i].
# This reduces the optimization to 300 variables.

def compute_score(h):
    """Compute max_k S_k / S_0 where S_k is autocorrelation."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-15):
        h = np.clip(h, 0, None)
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    S = np.fft.irfft(H * np.conj(H), n=n_fft)[:N]
    S0 = np.dot(h, h)
    if S0 < 1e-30:
        return np.inf
    return np.max(S) / S0

def symmetrize(h_half):
    """Expand a half-symmetric vector to full symmetric vector."""
    h_half = np.asarray(h_half, dtype=np.float64)
    h_half = np.maximum(h_half, 0)
    # h is symmetric: h[i] = h[N-1-i]
    # If N is even, h[0..N/2-1] determines h[N/2..N-1]
    # h[N-1-i] = h[i], so h[N-1] = h[0], h[N-2] = h[1], etc.
    # h_half has N/2 elements: h_half[i] = h[i] for i in 0..N/2-1
    # Then h[i] for i in N/2..N-1 is h[N-1-i]
    h_full = np.zeros(N)
    h_full[:N//2] = h_half
    h_full[N//2:] = h_half[::-1]
    return h_full

def objective(h_half):
    """Objective function for optimization."""
    h_full = symmetrize(h_half)
    return compute_score(h_full)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    best_h = np.ones(N)
    best_score = compute_score(best_h)
    
    # Try several initializations
    # 1. Uniform
    # 2. Gaussian-like
    # 3. Cosine
    # 4. Triangle
    # 5. Random
    
    inits = []
    
    # Uniform
    inits.append(np.ones(N//2))
    
    # Gaussian-like
    x = np.linspace(-1, 1, N//2)
    inits.append(np.exp(-4 * x**2))
    
    # Cosine
    inits.append(0.5 * (1 + np.cos(np.pi * x)))
    
    # Triangle
    inits.append(1 - np.abs(x))
    
    # Random
    np.random.seed(42)
    for _ in range(5):
        inits.append(np.random.uniform(0.1, 1.0, N//2))
    
    # Try L-BFGS-B on each initialization
    for i, h0 in enumerate(inits):
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                objective,
                h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * (N//2),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_h = symmetrize(result.x)
        except Exception:
            pass
    
    # Try differential evolution for global search
    if time.time() < deadline - 30:
        try:
            bounds = [(0, 1)] * (N//2)
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=True
            )
            
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_h = symmetrize(de_result.x)
        except Exception:
            pass
    
    # Final refinement with L-BFGS-B from the best found
    if time.time() < deadline - 10:
        try:
            h0_best = best_h[:N//2].copy()
            result = minimize(
                objective,
                h0_best,
                method='L-BFGS-B',
                bounds=[(0, None)] * (N//2),
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_h = symmetrize(result.x)
        except Exception:
            pass
    
    # Ensure nonnegativity
    best_h = np.maximum(best_h, 0)
    
    return list(best_h)

construct()
