# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

# Precompute Gauss-Hermite nodes and weights
_N_GQ = 15
_X_GQ, _W_GQ = np.polynomial.hermite.hermgauss(_N_GQ)

# Precompute powers and Hermite polynomials
_X2 = _X_GQ * _X_GQ
_H0 = np.ones(_N_GQ)
_H4 = _X2**2 - 6 * _X2 + 3
_H8 = _X2**4 - 28 * _X2**3 + 210 * _X2**2 - 420 * _X2 + 105
_H12 = _X2**6 - 84 * _X2**5 + 1260 * _X2**4 - 12600 * _X2**3 + 55440 * _X2**2 - 100800 * _X2 + 10395

# Precompute derivatives
_X3 = _X_GQ**3
_X5 = _X_GQ**5
_X7 = _X_GQ**7
_X9 = _X_GQ**9
_X11 = _X_GQ**11

_DH4 = 4 * _X3 - 12 * _X_GQ
_DH8 = 8 * _X7 - 168 * _X5 + 840 * _X3 - 840 * _X_GQ
_DH12 = 12 * _X11 - 840 * _X9 + 10080 * _X7 - 75600 * _X5 + 221760 * _X3 - 201600 * _X_GQ

# Basis vectors for the subspace defined by P(0)=0
# P = a*H0 + b*H4 + c*H8 + d*H12 with d = -(a+3b+105c)/10395
# P = a*(H0 - H12/10395) + b*(H4 - 3H12/10395) + c*(H8 - 105H12/10395)
# Let phi0, phi1, phi2 be the basis functions
_INV_10395 = 1.0 / 10395.0
_phi0 = _H0 - _INV_10395 * _H12
_phi1 = _H4 - 3 * _INV_10395 * _H12
_phi2 = _H8 - 105 * _INV_10395 * _H12

# Derivatives of basis functions
# H0' = 0
_dphi0 = -_INV_10395 * _DH12
_dphi1 = _DH4 - 3 * _INV_10395 * _DH12
_dphi2 = _DH8 - 105 * _INV_10395 * _DH12

# Compute moment matrices
# M0[i,j] = sum(w * phi_i * phi_j)
# M2[i,j] = sum(w * x^2 * phi_i * phi_j)
# MJ[i,j] = sum(w * (dphi_i - x*phi_i) * (dphi_j - x*phi_j))

_phis = np.array([_phi0, _phi1, _phi2])
_dphis = np.array([_dphi0, _dphi1, _dphi2])

# g' = P' - xP
# g'_i = dphi_i - x * phi_i
_gprimes = _dphis - _X_GQ[:, np.newaxis] * _phis

# M0
_M0 = np.dot(_W_GQ * _phis, _phis.T)
# M2
_M2 = np.dot(_W_GQ * _X2[:, np.newaxis] * _phis, _phis.T)
# MJ
_MJ = np.dot(_W_GQ * _gprimes, _gprimes.T)

# Objective function for v = [a, b, c]
def _objective_vec(v):
    a, b, c = v
    # v is [a, b, c]
    # I0 = v^T M0 v
    I0 = np.dot(v, np.dot(_M0, v))
    I2 = np.dot(v, np.dot(_M2, v))
    J0 = np.dot(v, np.dot(_MJ, v))
    
    if I0 <= 1e-15 or I2 <= 1e-15 or J0 <= 1e-15:
        return 1e10
    
    return np.sqrt(I2 * J0) / I0

def _objective_bc(bc):
    b, c = bc
    v = np.array([1.0, b, c])
    return _objective_vec(v)

def _gradient_bc(bc):
    b, c = bc
    v = np.array([1.0, b, c])
    # Gradient of F(v) w.r.t v
    # F = sqrt(I2 J0) / I0
    # dF/db = dF/dv * dv/db = dF/dv[1]
    # dF/dc = dF/dv[2]
    
    I0 = np.dot(v, np.dot(_M0, v))
    I2 = np.dot(v, np.dot(_M2, v))
    J0 = np.dot(v, np.dot(_MJ, v))
    
    if I0 <= 1e-15 or I2 <= 1e-15 or J0 <= 1e-15:
        return np.array([0.0, 0.0])
    
    F = np.sqrt(I2 * J0) / I0
    
    # dI0/dv = 2 M0 v
    dI0 = 2 * _M0.dot(v)
    dI2 = 2 * _M2.dot(v)
    dJ0 = 2 * _MJ.dot(v)
    
    # dF/dv = F * (0.5 dI2/I2 + 0.5 dJ0/J0 - dI0/I0)
    # Note: dI2/I2 is a vector
    grad_v = F * (0.5 * dI2 / I2 + 0.5 * dJ0 / J0 - dI0 / I0)
    
    return grad_v[1:]

def construct():
    start_time = time.time()
    deadline = 170.0
    
    seed_result = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_params = seed_result.copy()
    best_score = _objective_vec(best_params)
    
    # Phase 1: Global search using differential_evolution
    # Search space for b, c (ratios b/a, c/a)
    # Seed ratios: b/a ~ -0.035, c/a ~ -0.0027
    # Let's search a box around this, and also a larger box.
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    
    try:
        if time.time() < deadline - 10:
            res_de = differential_evolution(
                _objective_bc,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-12,
                polish=True,
                init='sobol'
            )
            if res_de.fun < best_score:
                best_score = res_de.fun
                best_params = np.array([1.0, res_de.x[0], res_de.x[1]])
    except Exception:
        pass
    
    # Phase 2: Local refinement from best point and seed
    points_to_refine = [best_params.copy(), seed_result.copy()]
    
    # Add seed ratios
    seed_ratios = [seed_result[1]/seed_result[0], seed_result[2]/seed_result[0]]
    points_to_refine.append(np.array([1.0] + seed_ratios))
    
    for p in points_to_refine:
        if time.time() > deadline - 5:
            break
        try:
            b, c = p[1], p[2]
            res_min = minimize(
                _objective_bc,
                np.array([b, c]),
                jac=_gradient_bc,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res_min.fun < best_score:
                best_score = res_min.fun
                best_params = np.array([1.0, res_min.x[0], res_min.x[1]])
        except Exception:
            pass
            
    # Phase 3: Final polish
    if time.time() < deadline - 2:
        try:
            b, c = best_params[1], best_params[2]
            res_polish = minimize(
                _objective_bc,
                np.array([b, c]),
                jac=_gradient_bc,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            if res_polish.fun < best_score:
                best_score = res_polish.fun
                best_params = np.array([1.0, res_polish.x[0], res_polish.x[1]])
        except Exception:
            pass
            
    # Ensure we don't return worse than seed
    seed_score = _objective_vec(seed_result)
    if best_score > seed_score:
        best_params = seed_result.copy()
        
    return best_params
# EVOLVE-BLOCK-END
