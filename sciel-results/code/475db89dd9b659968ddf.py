# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize, basin_hopping

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

def _compute_R_normalized(c_norm):
    """
    Compute R for a unit-norm coefficient vector.
    Since R is scale invariant, we can compute it for any scale.
    Using unit norm ensures stability.
    """
    f = _B_flat @ c_norm
    l2_sq = np.sum(f ** 2 * _sphere_weights_flat)
    if l2_sq <= 0:
        return np.inf
    l2_norm = np.sqrt(l2_sq)
    
    l4_pow4 = np.sum(np.abs(f) ** 4 * _sphere_weights_flat)
    if l4_pow4 <= 0:
        return np.inf
    l4_norm = l4_pow4 ** 0.25
    
    return l4_norm / l2_norm

def _objective_unit_sphere(c):
    """
    Objective function for optimizers.
    We assume c is already on the unit sphere or we normalize it?
    Standard optimizers don't handle constraints well.
    Instead, we will use a penalty or just rely on the fact that 
    basin_hopping with a local minimizer might drift off scale.
    
    Better approach: 
    The objective is scale invariant. 
    If we minimize R(c), the gradient with respect to scale is 0.
    This causes issues for L-BFGS-B.
    
    Solution: Fix the scale by adding a penalty term 
    ||c||_2^2 - 1 squared, or simply normalize c inside the objective 
    and return the value. But if we normalize inside, the gradient 
    of the objective w.r.t c is not simply the gradient of R.
    
    Alternative: Use the fact that we can pick ANY scale.
    Let's just minimize R(c) directly. L-BFGS-B will find a local min 
    on the cone. The value R is the same for all scales.
    To prevent c from going to 0 or inf, we add a small regularization 
    term to keep ||c|| close to 1.
    """
    c = np.asarray(c, dtype=np.float64)
    if not np.all(np.isfinite(c)):
        return 1e10
    
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-6:
        return 1e10
        
    # Normalize to unit sphere to evaluate R, but add a penalty if we 
    # want to keep the optimizer on the sphere. 
    # Actually, since R is scale invariant, minimizing R(c) is fine.
    # The "flat" direction is the radial direction.
    # L-BFGS-B might struggle if the Hessian is singular in that direction.
    # Let's add a tiny penalty to keep ||c|| near 1.
    penalty = 1e-6 * (norm_c - 1.0)**2
    
    c_norm = c / norm_c
    f = _B_flat @ c_norm
    l2_sq = np.sum(f ** 2 * _sphere_weights_flat)
    if l2_sq <= 0:
        return 1e10
    l2_norm = np.sqrt(l2_sq)
    l4_pow4 = np.sum(np.abs(f) ** 4 * _sphere_weights_flat)
    if l4_pow4 <= 0:
        return 1e10
    l4_norm = l4_pow4 ** 0.25
    
    return l4_norm / l2_norm + penalty

def _random_unit_vector(rng):
    v = rng.normal(size=2 * K + 1)
    v /= np.linalg.norm(v)
    return v

def _generate_initial_candidates(rng, n_candidates):
    """
    Generate diverse initial candidates on the unit sphere.
    """
    candidates = []
    # 1. Pure m modes (normalized)
    for m in range(-K, K + 1):
        c = np.zeros(2 * K + 1)
        c[m + K] = 1.0
        candidates.append(c)
        
    # 2. Random vectors
    for _ in range(n_candidates):
        candidates.append(_random_unit_vector(rng))
        
    # 3. Structured: Even/Odd m
    even_indices = [m + K for m in range(-K, K + 1, 2)]
    for _ in range(10):
        c = np.zeros(2 * K + 1)
        c[even_indices] = rng.normal(size=len(even_indices))
        c /= np.linalg.norm(c)
        candidates.append(c)
        
    odd_indices = [m + K for m in range(-K + 1, K, 2)]
    for _ in range(10):
        c = np.zeros(2 * K + 1)
        c[odd_indices] = rng.normal(size=len(odd_indices))
        c /= np.linalg.norm(c)
        candidates.append(c)
        
    return candidates

def rudin_problem10() -> np.ndarray:
    deadline = time.time() + 590
    start_time = time.time()
    
    best_coeffs = None
    best_R = np.inf
    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = _generate_initial_candidates(rng, 50)
    
    # We will use Basin Hopping for a few of the best candidates
    # and L-BFGS-B for all of them.
    
    # First pass: Local optimization on all candidates
    for c_init in candidates:
        if time.time() > deadline - 30:
            break
            
        try:
            res = minimize(
                _objective_unit_sphere, 
                c_init, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            if res.fun < best_R:
                # Remove penalty for final R calculation
                c_res = res.x
                c_res_norm = c_res / np.linalg.norm(c_res)
                R_val = _compute_R_normalized(c_res_norm)
                if R_val < best_R:
                    best_R = R_val
                    best_coeffs = c_res_norm.copy()
        except Exception:
            pass
            
    # Second pass: Basin Hopping on the best candidate and a few others
    # Basin Hopping is slower but can escape local minima.
    # We run it on the current best and 5 random ones.
    bh_candidates = [best_coeffs] if best_coeffs is not None else []
    bh_candidates.extend(candidates[-5:]) # Last 5 random ones
    
    for c_init in bh_candidates:
        if time.time() > deadline - 10:
            break
        if c_init is None:
            continue
            
        try:
            def step(c):
                # Random perturbation
                noise = rng.normal(scale=0.1, size=c.shape)
                c_new = c + noise
                c_new /= np.linalg.norm(c_new)
                return c_new
                
            res_bh = basin_hopping(
                _objective_unit_sphere, 
                c_init, 
                stepfunc=step, 
                niter=50, 
                T=1.0, 
                accept_test='anneal',
                minstep=1e-4,
                maxfun=1000
            )
            
            if res_bh.fun < best_R:
                c_res = res_bh.x
                c_res_norm = c_res / np.linalg.norm(c_res)
                R_val = _compute_R_normalized(c_res_norm)
                if R_val < best_R:
                    best_R = R_val
                    best_coeffs = c_res_norm.copy()
        except Exception:
            pass

    # Final check and scaling
    if best_coeffs is None:
        best_coeffs = np.ones(2 * K + 1)
        best_coeffs /= np.linalg.norm(best_coeffs)
        
    best_coeffs = np.asarray(best_coeffs, dtype=np.float64)
    
    # Ensure finite
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(2 * K + 1)
        best_coeffs /= np.linalg.norm(best_coeffs)
        
    # The task requires max|c| in [1e-3, 1e5].
    # Our best_coeffs is unit norm, so max|c| is roughly 1/sqrt(21) ~ 0.2.
    # This is within [1e-3, 1e5].
    # We can return it as is.
    
    return best_coeffs
# EVOLVE-BLOCK-END
