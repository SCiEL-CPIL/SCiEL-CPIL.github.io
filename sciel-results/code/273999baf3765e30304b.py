"""
Strategy: Low-dimensional DCT representation optimization.

Instead of optimizing 50 individual heights, we represent the step heights h
as a linear combination of the first K Discrete Cosine Basis functions.
h = DCT_matrix @ coeffs

This structural constraint:
1. Reduces dimensionality from 50 to K (e.g., K=10 or 12).
2. Enforces global smoothness, avoiding high-frequency spikes that can
   increase norm_inf relative to norm_1 or l2 in detrimental ways.
3. Allows for faster convergence and potentially better global exploration
   of the "smooth" manifold of solutions.

We use L-BFGS-B to optimize the coefficients. We also include a "projection"
step to ensure non-negativity by clipping the resulting h, and we use a
smoothed objective for the max-norm term.
"""
import numpy as np
import time
import scipy.optimize as opt
from scipy.fftpack import dct, idct

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 149.0     # safety margin
RNG = np.random.RandomState(50)

# Precompute DCT matrix for the basis
# We use type 2 DCT for the forward transform and type 3 for inverse.
# scipy.fftpack.dct(x, type=2) gives the coefficients.
# scipy.fftpack.idct(c, type=3) gives the signal.
# We want a matrix B such that h = B @ coeffs.
# Let's construct B explicitly for the first K components.
K_MAX = 15
B = np.zeros((N, K_MAX))
for k in range(K_MAX):
    # The k-th DCT basis function (0-indexed)
    # For type 2 DCT, the basis vector v_k has components:
    # v_k[j] = sqrt(2/N) * cos(pi * k * (j + 0.5) / N) for k>0
    # v_0[j] = sqrt(1/N)
    # However, scipy's idct(type=3) is the inverse of dct(type=2).
    # Let's just use idct to generate the basis vectors to be safe and consistent.
    c = np.zeros(K_MAX)
    c[k] = 1.0
    B[:, k] = idct(c, type=3)
    # Note: idct might include scaling factors. We don't care about absolute scale
    # of basis vectors as long as they are linearly independent and span the space.
    # We will optimize the coefficients, so the scale is absorbed.

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

def _smooth_value_from_coeffs(coeffs, beta=100.0, K=10):
    """
    Smooth surrogate for the value function using log-sum-exp for norm_inf.
    Input: coeffs (vector of length K)
    Returns negative value for minimization.
    """
    # Reconstruct h from coefficients
    # We only use the first K coefficients
    c_full = np.zeros(K_MAX)
    c_full[:K] = coeffs
    h = idct(c_full, type=3)
    
    # Enforce non-negativity
    h = np.maximum(0.0, h)
    
    if np.all(h == 0):
        return 0.0
        
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)  # g >= 0 since h >= 0
    
    # Smooth max via log-sum-exp
    g_max = np.max(g)
    if g_max <= 0:
        return 0.0
    lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
    
    if norm_1 <= 0 or lse <= 0:
        return 0.0
        
    val = l2 / (norm_1 * lse)
    return -val

def _generate_coeff_starts(K):
    starts = []
    # Uniform coefficients (flat-ish)
    starts.append(np.ones(K))
    
    # Only DC component (constant)
    c_dc = np.zeros(K)
    c_dc[0] = 1.0
    starts.append(c_dc)
    
    # DC + AC1 (cosine bump)
    c_bump = np.zeros(K)
    c_bump[0] = 1.0
    c_bump[1] = 0.5
    starts.append(c_bump)
    
    # Random smooth profiles
    for _ in range(50):
        # Low frequency emphasis
        coeffs = RNG.normal(0, 1, K)
        # Decay amplitude with frequency to ensure smoothness
        coeffs *= np.exp(-np.arange(K) / 3.0)
        # Ensure non-negative DC component to keep mean positive
        coeffs[0] = abs(coeffs[0]) + 0.1
        starts.append(coeffs)
        
    # Specific structured starts
    # Semicircle-like: mix of low freqs
    c_semi = np.zeros(K)
    c_semi[0] = 1.0
    c_semi[1] = -0.2
    c_semi[2] = 0.1
    starts.append(c_semi)
    
    # Two peaks: higher frequency
    c_two = np.zeros(K)
    c_two[0] = 1.0
    c_two[2] = 0.8
    starts.append(c_two)
    
    return starts

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = 0.0
    best_h = np.ones(N)
    
    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
    
    # We will optimize K coefficients. Let's try a few K values.
    # Smaller K is smoother, larger K is more flexible.
    K_values = [5, 8, 10, 12]
    
    for K in K_values:
        if time.time() > deadline - 30.0:
            break
            
        starts = _generate_coeff_starts(K)
        
        # Initial evaluation of starts
        for s in starts:
            # Reconstruct h to evaluate
            c_full = np.zeros(K_MAX)
            c_full[:K] = s
            h = idct(c_full, type=3)
            h = np.maximum(0.0, h)
            track(h)
            
        # Optimization
        bounds = [(-10, 10)] * K  # Generous bounds for coefficients
        
        # Phase 1: Exploration with moderate beta
        for s in starts:
            if time.time() > deadline - 20.0:
                break
            x0 = np.asarray(s, dtype=float)
            
            for beta in [50.0, 100.0]:
                if time.time() > deadline - 10.0:
                    break
                try:
                    res = opt.minimize(
                        lambda x: _smooth_value_from_coeffs(x, beta, K),
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if res.fun < 0:
                        # Reconstruct h from res.x
                        c_full = np.zeros(K_MAX)
                        c_full[:K] = res.x
                        h = idct(c_full, type=3)
                        track(h)
                except Exception:
                    pass
        
        # Phase 2: Refinement from best found so far (projected back to coeffs)
        # We need to find coefficients that represent best_h.
        # Use least squares to project best_h onto the DCT basis.
        if best_val > 0:
            # Solve B[:, :K] @ coeffs = best_h
            try:
                coeffs_best, residuals, rank, sv = np.linalg.lstsq(B[:, :K], best_h, rcond=None)
            except:
                coeffs_best = np.zeros(K)
            
            for beta in [100.0, 200.0, 500.0]:
                if time.time() > deadline - 5.0:
                    break
                try:
                    res = opt.minimize(
                        lambda x: _smooth_value_from_coeffs(x, beta, K),
                        coeffs_best,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    if res.fun < 0:
                        c_full = np.zeros(K_MAX)
                        c_full[:K] = res.x
                        h = idct(c_full, type=3)
                        track(h)
                except Exception:
                    pass

    # Final polish: Use the best_h directly in a high-dimensional local search?
    # The prompt asks for a structural change. The DCT approach IS the structural change.
    # However, mixing in a final high-dim polish might be beneficial if the DCT
    # approximation is slightly off. But to stay true to the "low-dimensional"
    # strategy, we will stick to the DCT results.
    
    # One more check: ensure best_h is valid
    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
