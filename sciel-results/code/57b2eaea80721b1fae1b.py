"""
Refined program: Fast moment calculation via matrix precomputation + Multi-start L-BFGS-B.
This approach significantly speeds up the objective function evaluation, allowing for
a more thorough search of the local minima and tighter convergence tolerances.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

# Precompute the moment integral I[n] = int x^(2n) e^(-x^2) dx
# I[n] = (2n-1)!! / 2^n * sqrt(pi)
def _precompute_integrals():
    I = np.zeros(13)
    sp = math.sqrt(math.pi)
    # I[0] = sqrt(pi)
    I[0] = sp
    for n in range(1, 13):
        # Recurrence: I[n] = (2n-1)/2 * I[n-1]
        I[n] = (2 * n - 1) / 2.0 * I[n - 1]
    return I

_I_TABLE = _precompute_integrals()

# Precompute the quadratic form matrices for m0 and m2
# p is a vector of 7 coefficients [p0, p1, p2, p3, p4, p5, p6] corresponding to x^0, x^2, ..., x^12
# m_k = sum_{i,j} p_i p_j I[k+i+j]
# Let A_k be a 7x7 matrix where A_k[i,j] = I[k+i+j]
# Then m_k = p @ A_k @ p

_A0 = np.zeros((7, 7))
_A2 = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        _A0[i, j] = _I_TABLE[i + j]
        if i + j + 1 < 13:
            _A2[i, j] = _I_TABLE[i + j + 1]

# Derivative of p with respect to params [c0, c4, c8]
# c12 = -(c0 + 3c4 + 105c8) / 945
# p[0] = c0 + 3c4 + 105c8 + 945c12 = 0 (Wait, g(0)=0 implies p[0]=0? No.)
# Let's re-verify the Hermite coefficients.
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
# g(x) = (c0 H0 + c4 H4 + c8 H8 + c12 H12) e^(-x^2/2)
# P(x) = c0*1 + c4*(x^4-6x^2+3) + c8*(x^8-28x^6+210x^4-420x^2+105) + c12*(...)
# p[0] (x^0) = c0 + 3c4 + 105c8 + 945c12
# g(0) = P(0) = 0 => c0 + 3c4 + 105c8 + 945c12 = 0 => p[0] = 0 always.
# This simplifies things! p[0] is identically zero.
# We can remove p[0] from the calculation or keep it as 0.
# Let's keep the 7-element vector for consistency but p[0] will be 0.

# dp/dc matrix (7x3)
# dc12/dc = [-1/945, -3/945, -105/945]
dc12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

_dp_dc = np.zeros((7, 3))
# Row 0 (x^0): Coeffs of c0, c4, c8, c12 are 1, 3, 105, 945.
# dp0/dc = [1, 3, 105] + 945*dc12 = [1 - 1, 3 - 3, 105 - 105] = [0, 0, 0]. Correct.
_dp_dc[0, :] = 0.0

# Row 1 (x^2): Coeffs 0, -6, -420, -7560.
_dp_dc[1, :] = np.array([0.0, -6.0, -420.0]) - 7560.0 * dc12

# Row 2 (x^4): Coeffs 0, 1, 210, 13860.
_dp_dc[2, :] = np.array([0.0, 1.0, 210.0]) + 13860.0 * dc12

# Row 3 (x^6): Coeffs 0, 0, -28, -6930.
_dp_dc[3, :] = np.array([0.0, 0.0, -28.0]) - 6930.0 * dc12

# Row 4 (x^8): Coeffs 0, 0, 1, 1155.
_dp_dc[4, :] = np.array([0.0, 0.0, 1.0]) + 1155.0 * dc12

# Row 5 (x^10): Coeffs 0, 0, 0, -66.
_dp_dc[5, :] = np.array([0.0, 0.0, 0.0]) - 66.0 * dc12

# Row 6 (x^12): Coeffs 0, 0, 0, 1.
_dp_dc[6, :] = np.array([0.0, 0.0, 0.0]) + 1.0 * dc12

def _bound_and_grad(params):
    """
    Compute the uncertainty bound and gradient.
    params = [c0, c4, c8]
    """
    c0, c4, c8 = params[0], params[1], params[2]
    
    # Construct p vector (7 elements)
    # p = _dp_dc.T @ params ? No, p is linear in params.
    # p_k = sum_a _dp_dc[k, a] * params[a]
    p = _dp_dc.T @ params
    
    # Compute moments m0 and m2
    # m0 = p @ _A0 @ p
    # m2 = p @ _A2 @ p
    m0 = p @ _A0 @ p
    m2 = p @ _A2 @ p
    
    if m0 <= 1e-15 or not np.isfinite(m0):
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    var_x = m2 / m0
    # Due to symmetry, var_xi = var_x.
    # Bound = sqrt(var_x * var_xi) = var_x.
    bound = var_x
    
    # Gradient calculation
    # dvar_x = (dm2 m0 - m2 dm0) / m0^2
    # dm0/dparams = 2 * p @ (_A0 @ _dp_dc)  -- Wait.
    # m0 = sum_{i,j} p_i p_j A0[i,j].
    # dm0/dp_k = 2 sum_j p_j A0[k,j].
    # dm0/dparams_c = sum_k (dm0/dp_k) * (dp_k/dparams_c) = 2 sum_k (sum_j p_j A0[k,j]) _dp_dc[k,c]
    # Let v0 = _A0 @ p. Then v0[k] = sum_j A0[k,j] p_j.
    # dm0/dparams_c = 2 * sum_k v0[k] _dp_dc[k,c] = 2 * v0 @ _dp_dc[:,c]
    
    v0 = _A0 @ p
    v2 = _A2 @ p
    
    dm0_dp = 2.0 * v0
    dm2_dp = 2.0 * v2
    
    # dm0/dparams = dm0_dp @ _dp_dc
    dm0_dparams = dm0_dp @ _dp_dc
    dm2_dparams = dm2_dp @ _dp_dc
    
    grad = (dm2_dparams * m0 - m2 * dm0_dparams) / (m0 * m0)
    
    return bound, grad

def _objective(params):
    val, _ = _bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 180  # Safety margin before 210s kill

    # Initial guess from seed
    x0 = np.array([-1.13, 0.04, 0.000308642])
    
    best_x = x0.copy()
    best_score = _objective(x0)
    
    # Strategy: Multi-start L-BFGS-B
    # The problem is 3D. We can afford many restarts.
    # We use a combination of the known good point and random perturbations.
    
    # 1. High-precision polish of the seed point
    try:
        res_seed = minimize(_objective_grad, best_x, jac=True, method='L-BFGS-B',
                            options={'maxiter': 10000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 50})
        if res_seed.fun < best_score:
            best_score = res_seed.fun
            best_x = res_seed.x.copy()
    except Exception:
        pass

    if time.time() > deadline - 20:
        return best_x

    # 2. Multi-start from random points in a reasonable box
    # The coefficients are small. c0 is around -1, c4 around 0.04, c8 around 0.0003.
    # Let's try a box: c0 in [-2, 0], c4 in [-0.1, 0.1], c8 in [-0.01, 0.01]
    
    np.random.seed(42)
    num_starts = 100
    for i in range(num_starts):
        if time.time() > deadline - 5:
            break
            
        # Generate a random start point
        # Mix of Gaussian perturbations around the best found so far and uniform randoms
        if i < 20:
            # Local perturbations
            x_start = best_x + np.random.normal(0, 0.1, size=3)
        else:
            # Global-ish restarts
            x_start = np.array([
                np.random.uniform(-2.0, 0.0),
                np.random.uniform(-0.1, 0.1),
                np.random.uniform(-0.01, 0.01)
            ])
            
        try:
            res_rand = minimize(_objective_grad, x_start, jac=True, method='L-BFGS-B',
                                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res_rand.fun < best_score:
                best_score = res_rand.fun
                best_x = res_rand.x.copy()
        except Exception:
            pass

    # 3. Final ultra-high precision polish
    if time.time() < deadline - 5:
        try:
            res_final = minimize(_objective_grad, best_x, jac=True, method='L-BFGS-B',
                                 options={'maxiter': 20000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 100})
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_x = res_final.x.copy()
        except Exception:
            pass

    return best_x

# EVOLVE-BLOCK-END
