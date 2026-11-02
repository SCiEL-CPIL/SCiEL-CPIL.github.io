"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200

def _time_left():
    return _DEADLINE - time.time()

# Precompute constants for the analytic solution
# Indices of Hermite polynomials
K = [0, 4, 8, 12]
# Factorials
FACT = {0: 1, 4: 24, 8: 40320, 12: 479001600}
# Normalization N_k = 2^k k! sqrt(pi)
SQRT_PI = np.sqrt(np.pi)
N = np.array([ (2**k) * FACT[k] * SQRT_PI for k in K ])

# Matrix V where V_ij = int H_i H_j x^2 e^{-x^2} dx
# V_ii = 2^i i! sqrt(pi) (2i+1)
# V_i,i+2 = 2^i (i+2)! sqrt(pi)
V_mat = np.zeros((4, 4))
for idx_i, i in enumerate(K):
    V_mat[idx_i, idx_i] = (2**i) * FACT[i] * SQRT_PI * (2*i + 1)
    if i + 2 in FACT:
        idx_j = K.index(i + 2)
        val = (2**i) * FACT[i+2] * SQRT_PI
        V_mat[idx_i, idx_j] = val
        V_mat[idx_j, idx_i] = val
    if i - 2 in FACT:
        idx_j = K.index(i - 2)
        val = (2**i) * FACT[i-2] * SQRT_PI
        V_mat[idx_i, idx_j] = val
        V_mat[idx_j, idx_i] = val

# Coefficients for a12 in terms of a0, a4, a8
# P(0) = a0 + 3a4 + 105a8 + 945a12 = 0
# H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=945
# a12 = -(a0 + 3a4 + 105a8) / 945
A12_COEF = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
A12_CONST = 0.0

def _compute_bound_analytic(a0, a4, a8):
    a = np.array([a0, a4, a8])
    a12 = np.dot(A12_COEF, a) + A12_CONST
    c = np.array([a0, a4, a8, a12])
    
    # Q0 = sum c_i^2 N_i
    Q0 = np.sum(c**2 * N)
    if Q0 <= 0:
        return 1e10
        
    # Q1 = c^T V c
    Q1 = c @ V_mat @ c
    
    # For Fourier: coefficients c'_k = c_k i^k
    # Q2 = sum c_i c_j i^{i+j} V_ij
    # i^k for k in [0, 4, 8, 12] is [1, 1, 1, 1]
    # So i^{i+j} is 1 if i+j is even, -1 if i+j is odd?
    # i^0=1, i^4=1, i^8=1, i^12=1.
    # So i^{i+j} = 1 for all pairs in our set.
    # Wait, i^4 = 1. i^8 = 1.
    # So c'_k = c_k.
    # Then Q2 = Q1?
    # Let's check.
    # F(H_n) = i^n H_n.
    # |F(P)|^2 involves cross terms c_i c_j (i^i) (i^j)^* = c_i c_j i^{i-j}.
    # The variance formula uses \int \xi^2 |F|^2.
    # |F|^2 = \sum c_i c_j i^{i-j} H_i H_j e^{-\xi^2}.
    # So the matrix for Q2 is V_mat multiplied by i^{i-j} in the (i,j) entry.
    
    Q2_mat = V_mat.copy()
    for idx_i, i in enumerate(K):
        for idx_j, j in enumerate(K):
            # i^i * (i^j)^* = i^{i-j}
            # i is 1, -1, -1, 1, 1...
            # i^0=1, i^4=1, i^8=1, i^12=1.
            # So i^i is always 1.
            # i^j is always 1.
            # So i^{i-j} = 1.
            # Wait. i^4 = 1. i^8 = 1.
            # So the phase is always 1.
            # Thus Q2 = Q1.
            # If Q2 = Q1, then A(f)A(F) = Q1/Q0.
            # This would mean the bound is just Q1/Q0.
            
            pass # No change needed if phase is 1
    
    # However, let's be safe and compute it generally.
    # phase[i][j] = 1j ** (K[i] - K[j])
    Q2 = 0.0
    for idx_i in range(4):
        for idx_j in range(4):
            phase = 1j ** (K[idx_i] - K[idx_j])
            Q2 += c[idx_i] * c[idx_j] * V_mat[idx_i, idx_j] * phase.real
            # Imaginary part should be 0 for real c
    
    if Q1 <= 0 or Q2 <= 0:
        return 1e10
        
    var_f = Q1 / Q0
    var_F = Q2 / Q0
    
    if var_f <= 0 or var_F <= 0:
        return 1e10
        
    return np.sqrt(var_f * var_F)

def _objective_log(log_coeffs):
    a0, a4, a8 = log_coeffs
    try:
        val = _compute_bound_analytic(a0, a4, a8)
        if np.isnan(val) or np.isinf(val):
            return 1e10
        return val
    except Exception:
        return 1e10

def construct():
    t_start = time.time()
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _compute_bound_analytic(*seed)
    best_coeffs = seed.copy()
    
    # Use L-BFGS-B for local optimization from multiple starts
    # or just a global search first.
    
    # Simple grid search to find a good start
    # The seed is already good. Let's just polish it.
    
    try:
        res = minimize(_objective_log, seed, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12})
        if res.fun < best_val:
            best_val = res.fun
            best_coeffs = res.x
    except Exception:
        pass
    
    # Try a few other starts if time permits
    if time.time() - t_start < 180:
        rng = np.random.RandomState(42)
        for _ in range(10):
            if time.time() - t_start > 190:
                break
            trial = rng.uniform(-2, 2, size=3)
            val = _compute_bound_analytic(*trial)
            if val < best_val:
                best_val = val
                best_coeffs = trial.copy()
                # Polish this new best
                try:
                    res2 = minimize(_objective_log, best_coeffs, method='L-BFGS-B', 
                                   options={'maxiter': 500, 'ftol': 1e-12})
                    if res2.fun < best_val:
                        best_val = res2.fun
                        best_coeffs = res2.x
                except Exception:
                    pass
                    
    return best_coeffs
# EVOLVE-BLOCK-END
