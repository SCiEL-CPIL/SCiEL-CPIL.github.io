# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize

K = 10
N_THETA = 70
N_PHI = 130

# Precompute quadrature weights and nodes
_x, _w = np.polynomial.legendre.leggauss(N_THETA)
_cos_theta = _x
_theta_weights = _w * 0.5

# phi grid
_phi = np.arange(N_PHI) * (2 * np.pi / N_PHI)

# Precompute associated Legendre functions P_l^m(cos theta) with Condon-Shortley phase
_P = {}
for m in range(K + 1):
    _P[m] = lpmv(m, K, _cos_theta)  # shape (N_THETA,)

# Normalization constants N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
def _norm_const(m):
    m = abs(m)
    val = 21.0 / (4.0 * np.pi) * float(factorial(10 - m) / factorial(10 + m))
    return np.sqrt(val)

# Build the full coefficient matrix: shape (2K+1, N_THETA, N_PHI)
_basis = np.zeros((2 * K + 1, N_THETA, N_PHI), dtype=np.float64)
for idx, m in enumerate(range(-K, K + 1)):
    am = abs(m)
    norm = _norm_const(am)
    Pm = _P[am]  # shape (N_THETA,)
    if m == 0:
        _basis[idx, :, :] = norm * Pm[:, None]
    elif m > 0:
        cos_term = np.cos(m * _phi)  # shape (N_PHI,)
        _basis[idx, :, :] = np.sqrt(2.0) * norm * Pm[:, None] * cos_term[None, :]
    else:
        sin_term = np.sin(am * _phi)  # shape (N_PHI,)
        _basis[idx, :, :] = np.sqrt(2.0) * norm * Pm[:, None] * sin_term[None, :]

# Quadrature weights on the sphere
_w_phi = 2.0 * np.pi / N_PHI
_sphere_weights = np.outer(_theta_weights, np.full(N_PHI, _w_phi))  # shape (N_THETA, N_PHI)
_sphere_weights_flat = _sphere_weights.ravel()  # shape (N_THETA * N_PHI,)

# Precompute basis as flat matrix: shape (N_POINTS, 2K+1)
# f(point) = B(point, :) @ c
_B_flat = _basis.transpose(1, 2, 0).reshape(-1, 2 * K + 1)  # shape (N_POINTS, 21)
N_POINTS = N_THETA * N_PHI

def _compute_R(coeffs):
    """Compute ||f||_4 / ||f||_2 for given coefficients."""
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.shape != (2 * K + 1,):
        return np.inf
    if not np.all(np.isfinite(coeffs)):
        return np.inf
    max_abs = np.max(np.abs(coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        return np.inf
    
    f = _B_flat @ coeffs  # shape (N_POINTS,)
    
    l2_sq = np.sum(f ** 2 * _sphere_weights_flat)
    if l2_sq <= 0:
        return np.inf
    l2_norm = np.sqrt(l2_sq)
    
    l4_pow4 = np.sum(np.abs(f) ** 4 * _sphere_weights_flat)
    if l4_pow4 <= 0:
        return np.inf
    l4_norm = l4_pow4 ** 0.25
    
    return l4_norm / l2_norm

def _objective(coeffs):
    """
    Objective function for L-BFGS-B.
    We minimize R = ||f||_4 / ||f||_2.
    To improve numerical stability and avoid division by zero issues during optimization,
    we can minimize R^4 = (sum w f^4) / (sum w f^2)^2.
    However, L-BFGS-B handles smooth functions well. Let's stick to R for simplicity,
    but scale the coefficients internally if needed? No, R is scale invariant.
    We just need to ensure the optimizer doesn't drift to zero.
    """
    if not np.all(np.isfinite(coeffs)):
        return 1e10
    max_abs = np.max(np.abs(coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        # Penalty for invalid scale
        return 1e10
        
    f = _B_flat @ coeffs
    l2_sq = np.sum(f ** 2 * _sphere_weights_flat)
    if l2_sq <= 0:
        return 1e10
    l4_pow4 = np.sum(np.abs(f) ** 4 * _sphere_weights_flat)
    if l4_pow4 <= 0:
        return 1e10
        
    R = (l4_pow4 ** 0.25) / (l2_sq ** 0.5)
    return R

def _generate_initial_candidates():
    """
    Generate a diverse set of initial candidates based on structural insights.
    1. Uniform random.
    2. Single m values (pure harmonics).
    3. Pairs of m values (interference).
    4. Symmetric subsets (even m, odd m).
    5. Random sparse vectors.
    """
    candidates = []
    rng = np.random.default_rng(42)
    
    # 1. Pure m values (21 candidates)
    for m in range(-K, K + 1):
        c = np.zeros(2 * K + 1)
        c[m + K] = 1.0
        candidates.append(c)
    
    # 2. Pairs of m values (choose 2 from 21)
    # This is C(21, 2) = 210 candidates. We'll sample a subset to keep time reasonable.
    # Let's take all pairs where |m1 - m2| is small (nearby modes interfere strongly)
    # and some random pairs.
    indices = list(range(2 * K + 1))
    # Nearby pairs
    for i in range(2 * K + 1):
        for j in range(i + 1, min(i + 3, 2 * K + 1)):
            c = np.zeros(2 * K + 1)
            c[i] = 1.0
            c[j] = 1.0
            candidates.append(c)
    
    # Random pairs
    for _ in range(50):
        i, j = rng.choice(2 * K + 1, size=2, replace=False)
        c = np.zeros(2 * K + 1)
        c[i] = 1.0
        c[j] = 1.0
        candidates.append(c)
        
    # 3. Even m only (m = 0, 2, 4, 6, 8, 10 and -2, -4, -6, -8, -10)
    # 11 dimensions. Random in this subspace.
    even_indices = [m + K for m in range(-K, K + 1, 2)]
    for _ in range(20):
        c = np.zeros(2 * K + 1)
        vals = rng.uniform(-1, 1, size=len(even_indices))
        c[even_indices] = vals
        candidates.append(c)
        
    # 4. Odd m only
    odd_indices = [m + K for m in range(-K + 1, K, 2)] # -9, -7, ..., 9
    for _ in range(20):
        c = np.zeros(2 * K + 1)
        vals = rng.uniform(-1, 1, size=len(odd_indices))
        c[odd_indices] = vals
        candidates.append(c)
        
    # 5. Random sparse vectors (e.g., 3 non-zero elements)
    for _ in range(50):
        c = np.zeros(2 * K + 1)
        non_zero_idx = rng.choice(2 * K + 1, size=3, replace=False)
        c[non_zero_idx] = rng.uniform(-1, 1, size=3)
        candidates.append(c)
        
    # 6. Fully random
    for _ in range(50):
        c = rng.uniform(-1, 1, size=2 * K + 1)
        candidates.append(c)
        
    return candidates

def rudin_problem10() -> np.ndarray:
    deadline = time.time() + 590  # Safety margin
    start_time = time.time()
    
    best_coeffs = None
    best_R = np.inf
    
    # Generate initial candidates
    candidates = _generate_initial_candidates()
    
    # Optimize each candidate with L-BFGS-B
    # We limit the number of candidates to optimize to ensure we finish in time.
    # Each L-BFGS-B run is fast (ms), but we have ~200 candidates.
    # 200 * 10ms = 2s. This is very safe.
    
    for c_init in candidates:
        if time.time() > deadline - 10:
            break
            
        # Ensure initial candidate is valid
        if not np.all(np.isfinite(c_init)):
            continue
        max_abs = np.max(np.abs(c_init))
        if max_abs < 1e-3 or max_abs > 1e5:
            if max_abs > 0:
                c_init = c_init / max_abs * 0.1
            else:
                continue
                
        try:
            res = minimize(
                _objective, 
                c_init, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            if res.fun < best_R and np.all(np.isfinite(res.x)):
                best_R = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
            
    # If no valid result found, fallback to a single random search
    if best_coeffs is None:
        rng = np.random.default_rng(42)
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            c = rng.uniform(-1, 1, size=2 * K + 1)
            R = _compute_R(c)
            if R < best_R:
                best_R = R
                best_coeffs = c.copy()
                
    if best_coeffs is None:
        best_coeffs = np.ones(2 * K + 1)
        best_R = _compute_R(best_coeffs)
        
    best_coeffs = np.asarray(best_coeffs, dtype=np.float64)
    
    # Ensure finite
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(2 * K + 1)
        
    # Ensure max|c| in [1e-3, 1e5]
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        if max_abs > 0:
            best_coeffs = best_coeffs / max_abs * 1e-2
        else:
            best_coeffs = np.ones(2 * K + 1)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs / max_abs * 1e5
        
    return best_coeffs
# EVOLVE-BLOCK-END
