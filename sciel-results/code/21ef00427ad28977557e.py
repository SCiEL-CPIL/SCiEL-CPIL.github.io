import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, root
from scipy.linalg import eigh

_START_TIME = time.time()
DEADLINE = _START_TIME + 190  # 20s safety margin

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
# I[0] = sqrt(pi)
# I[n] = I[n-1] * (2n-1)/(2n)
I = np.zeros(25)
I[0] = np.sqrt(np.pi)
for n in range(1, 25):
    I[n] = I[n-1] * (2.0*n - 1.0) / (2.0*n)

# Coefficients for Hermite polynomials in power basis x^{2k}
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

# p_k is the coefficient of x^{2k} in P(x)
# P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
# a12 = -(a0 + 3*a4 + 105*a8) / 945

# Matrix A such that p = A * a, where a = [a0, a4, a8]
# p has 7 components (k=0..6)
# Row k of A corresponds to coefficient of x^{2k}

H0_c = np.array([1, 0, 0, 0, 0, 0, 0])
H4_c = np.array([3, -6, 1, 0, 0, 0, 0])
H8_c = np.array([105, -420, 210, -28, 1, 0, 0])
H12_c = np.array([945, -7560, 13860, -6930, 1155, -66, 1])

c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c

A_mat = np.column_stack((col0, col1, col2)) # Shape (7, 3)

def _get_p(a):
    return A_mat @ a

def _compute_bound_and_grad(a):
    """
    Compute the uncertainty bound and its analytic gradient.
    a: array of shape (3,)
    Returns: (bound, grad)
    """
    p = _get_p(a)
    
    # q_k = p_k * (-1)^k
    q = p * np.array([1, -1, 1, -1, 1, -1, 1])
    
    # Compute moments M_x[n] and M_xi[n] for n=0,1
    # M[n] = sum_{i,j} p_i p_j I[i+j+n]
    Mx0 = 0.0
    Mx1 = 0.0
    Mxi0 = 0.0
    Mxi1 = 0.0
    
    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0.0 and qi == 0.0:
            continue
        for j in range(7):
            pj = p[j]
            qj = q[j]
            idx = i + j
            Mx0 += pi * pj * I[idx]
            Mx1 += pi * pj * I[idx + 1]
            Mxi0 += qi * qj * I[idx]
            Mxi1 += qi * qj * I[idx + 1]
    
    if Mx0 <= 1e-14 or Mxi0 <= 1e-14:
        return 1e18, np.zeros(3)
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-14 or vxi <= 1e-14:
        return 1e18, np.zeros(3)
        
    prod = vx * vxi
    if prod <= 0:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(prod)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
        
    # Gradient computation
    Sx0 = np.zeros(7)
    Sx1 = np.zeros(7)
    Sxi0 = np.zeros(7)
    Sxi1 = np.zeros(7)
    
    for i in range(7):
        Sx0[i] = np.dot(p, I[i:i+7])
        Sx1[i] = np.dot(p, I[i+1:i+8])
        Sxi0[i] = np.dot(q, I[i:i+7])
        Sxi1[i] = np.dot(q, I[i+1:i+8])
            
    dMx0 = 2.0 * A_mat.T @ Sx0
    dMx1 = 2.0 * A_mat.T @ Sx1
    dMxi0 = 2.0 * A_mat.T @ Sxi0
    dMxi1 = 2.0 * A_mat.T @ Sxi1
    
    inv_Mx0_sq = 1.0 / (Mx0 * Mx0)
    inv_Mxi0_sq = 1.0 / (Mxi0 * Mxi0)
    
    dvx = (dMx1 * Mx0 - Mx1 * dMx0) * inv_Mx0_sq
    dvxi = (dMxi1 * Mxi0 - Mxi1 * dMxi0) * inv_Mxi0_sq
        
    factor = 0.5 / bound
    grad = factor * (vxi * dvx + vx * dvxi)
    
    if not np.all(np.isfinite(grad)):
        return bound, np.zeros(3)
        
    return bound, grad

def _stationary_equations(a):
    """
    Returns the gradient of F = Vx * Vxi.
    We want to find roots of this system (stationary points).
    """
    p = _get_p(a)
    q = p * np.array([1, -1, 1, -1, 1, -1, 1])
    
    Mx0 = 0.0
    Mx1 = 0.0
    Mxi0 = 0.0
    Mxi1 = 0.0
    
    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0.0 and qi == 0.0:
            continue
        for j in range(7):
            pj = p[j]
            qj = q[j]
            idx = i + j
            Mx0 += pi * pj * I[idx]
            Mx1 += pi * pj * I[idx + 1]
            Mxi0 += qi * qj * I[idx]
            Mxi1 += qi * qj * I[idx + 1]
            
    if Mx0 <= 1e-14 or Mxi0 <= 1e-14:
        return np.array([1e18, 1e18, 1e18])
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-14 or vxi <= 1e-14:
        return np.array([1e18, 1e18, 1e18])
        
    Sx0 = np.zeros(7)
    Sx1 = np.zeros(7)
    Sxi0 = np.zeros(7)
    Sxi1 = np.zeros(7)
    
    for i in range(7):
        Sx0[i] = np.dot(p, I[i:i+7])
        Sx1[i] = np.dot(p, I[i+1:i+8])
        Sxi0[i] = np.dot(q, I[i:i+7])
        Sxi1[i] = np.dot(q, I[i+1:i+8])
            
    dMx0 = 2.0 * A_mat.T @ Sx0
    dMx1 = 2.0 * A_mat.T @ Sx1
    dMxi0 = 2.0 * A_mat.T @ Sxi0
    dMxi1 = 2.0 * A_mat.T @ Sxi1
    
    inv_Mx0_sq = 1.0 / (Mx0 * Mx0)
    inv_Mxi0_sq = 1.0 / (Mxi0 * Mxi0)
    
    dvx = (dMx1 * Mx0 - Mx1 * dMx0) * inv_Mx0_sq
    dvxi = (dMxi1 * Mxi0 - Mxi1 * dMxi0) * inv_Mxi0_sq
        
    # Gradient of F = Vx * Vxi
    # dF = Vxi * dVx + Vx * dVxi
    grad_F = vxi * dvx + vx * dvxi
    
    return grad_F

def _is_local_min(a):
    """
    Check if a is a local minimum by checking Hessian positive definiteness.
    Approximate Hessian using finite differences of the gradient.
    """
    eps = 1e-6
    grad0 = _stationary_equations(a)
    if np.any(np.abs(grad0) > 1e-4):
        return False
        
    hess = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            a_plus = a.copy()
            a_minus = a.copy()
            a_plus[i] += eps
            a_minus[i] -= eps
            if i == j:
                hess[i, j] = (_stationary_equations(a_plus)[i] - _stationary_equations(a_minus)[i]) / (2*eps)
            else:
                # Cross derivatives
                a_pp = a.copy(); a_pm = a.copy(); a_mp = a.copy(); a_mm = a.copy()
                a_pp[i] += eps; a_pp[j] += eps
                a_pm[i] += eps; a_pm[j] -= eps
                a_mp[i] -= eps; a_mp[j] += eps
                a_mm[i] -= eps; a_mm[j] -= eps
                hess[i, j] = (_stationary_equations(a_pp)[i] - _stationary_equations(a_pm)[i] - _stationary_equations(a_mp)[i] + _stationary_equations(a_mm)[i]) / (4*eps*eps)
                
    try:
        eigvals = np.linalg.eigvalsh(hess)
        return np.all(eigvals > -1e-6)
    except:
        return False

def construct():
    """
    Find three coefficients (a0, a4, a8) that minimize the uncertainty product.
    Returns array of shape (3,).
    """
    
    # Seed value
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val, _ = _compute_bound_and_grad(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    deadline_margin = 10  # seconds
    
    # Strategy:
    # 1. Global search using Differential Evolution
    # 2. Local polish using L-BFGS-B with analytic gradient
    # 3. Structural refinement: Find stationary points via root finding and verify local minima
    
    # Define bounds for the search space
    # Based on the seed and typical scales
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]
    
    # Stage 1: Global Search
    if time.time() < DEADLINE - deadline_margin:
        try:
            # Differential Evolution is robust for small dimensions
            res_de = differential_evolution(
                _compute_bound_and_grad, 
                bounds, 
                maxiter=100, 
                popsize=20, 
                seed=42,
                tol=1e-10,
                polish=False,
                args=()
            )
            
            if res_de.success and res_de.fun < best_val:
                best_val = res_de.fun
                best_params = res_de.x.copy()
                
        except Exception:
            pass

    # Stage 2: Multi-start Local Optimization from Grid + DE result
    starts = [best_params.copy()]
    
    # Add seed
    starts.append(seed_params.copy())
    
    # Add some grid points around the best found
    if best_val < 1e18:
        eps = 0.1
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    start = best_params + np.array([i*eps, j*eps*0.1, k*eps*0.01])
                    # Only add if within bounds
                    if np.all(start >= np.array(bounds).T[0]) and np.all(start <= np.array(bounds).T[1]):
                        starts.append(start)
    
    # Random starts
    rng = np.random.default_rng(123)
    for _ in range(20):
        start = np.array([
            rng.uniform(bounds[0][0], bounds[0][1]),
            rng.uniform(bounds[1][0], bounds[1][1]),
            rng.uniform(bounds[2][0], bounds[2][1])
        ])
        starts.append(start)

    for start in starts:
        if time.time() > DEADLINE - deadline_margin:
            break
            
        try:
            res_min = minimize(
                _compute_bound_and_grad, 
                start, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res_min.success and res_min.fun < best_val:
                best_val = res_min.fun
                best_params = res_min.x.copy()
                
        except Exception:
            pass

    # Stage 3: Structural Refinement - Find stationary points
    # We look for roots of the gradient of the uncertainty product
    # This is a structural change: instead of just minimizing, we explicitly find stationary points
    # and verify they are local minima
    
    if time.time() < DEADLINE - deadline_margin:
        # Generate multiple starting points for root finding
        root_starts = [best_params.copy(), seed_params.copy()]
        
        # Add points from the grid
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    start = best_params + np.array([i*0.1, j*0.01, k*0.001])
                    if np.all(start >= np.array(bounds).T[0]) and np.all(start <= np.array(bounds).T[1]):
                        root_starts.append(start)
        
        # Random starts
        for _ in range(30):
            start = np.array([
                rng.uniform(bounds[0][0], bounds[0][1]),
                rng.uniform(bounds[1][0], bounds[1][1]),
                rng.uniform(bounds[2][0], bounds[2][1])
            ])
            root_starts.append(start)
        
        for start in root_starts:
            if time.time() > DEADLINE - deadline_margin:
                break
                
            try:
                # Find root of stationary equations
                res_root = root(
                    _stationary_equations, 
                    start, 
                    method='hybr',
                    tol=1e-12,
                    maxfev=1000
                )
                
                if res_root.success:
                    candidate = res_root.x
                    # Check if it's a local minimum
                    if _is_local_min(candidate):
                        val, _ = _compute_bound_and_grad(candidate)
                        if val < best_val:
                            best_val = val
                            best_params = candidate.copy()
                            
            except Exception:
                pass

    # Stage 4: High-precision Polish
    if time.time() < DEADLINE - deadline_margin:
        try:
            res_polish = minimize(
                _compute_bound_and_grad, 
                best_params, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            
            if res_polish.success and res_polish.fun < best_val:
                best_val = res_polish.fun
                best_params = res_polish.x.copy()
                
        except Exception:
            pass

    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
