import numpy as np
import time
from scipy.optimize import minimize_scalar

_DEADLINE = time.time() + 195

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
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

H0_c = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
H4_c = np.array([3.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0])
H8_c = np.array([105.0, -420.0, 210.0, -28.0, 1.0, 0.0, 0.0])
H12_c = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])

c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c

A_mat = np.column_stack((col0, col1, col2))  # Shape (7, 3)

def _get_p(a):
    return A_mat @ a

def _compute_ft_matrix(max_k=6):
    """Compute the Fourier transform matrix T where F[x^{2k} e^{-x^2/2}] = sum_j T[k,j] xi^{2j} e^{-xi^2/2}."""
    T = np.zeros((max_k+1, max_k+1))
    T[0, 0] = 1.0
    
    # Q_0(xi) = 1
    coeffs = [1.0] + [0.0] * max_k
    T[0, :] = coeffs
    
    for k in range(1, max_k+1):
        new_coeffs = [0.0] * (max_k+1)
        for m in range(max_k+1):
            val = 0.0
            if m+1 <= max_k:
                val += -2.0*(m+1)*(2.0*m+1.0) * coeffs[m+1]
            val += (4.0*m+1.0) * coeffs[m]
            if m >= 1:
                val += -coeffs[m-1]
            new_coeffs[m] = val
        coeffs = new_coeffs
        T[k, :] = coeffs
    
    return T

FT_mat = _compute_ft_matrix(6)
B_mat = FT_mat.T @ A_mat  # Shape (7, 3)

# Precompute Moment Matrices in a-space (3x3)
# M0 = A^T G0 A, M1 = A^T G1 A
# N0 = B^T G0 B, N1 = B^T G1 B

G0 = np.zeros((7, 7))
G1 = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        G0[i, j] = I[i + j]
        G1[i, j] = I[i + j + 1]

M0 = A_mat.T @ G0 @ A_mat
M1 = A_mat.T @ G1 @ A_mat
N0 = B_mat.T @ G0 @ B_mat
N1 = B_mat.T @ G1 @ B_mat

# Ensure symmetry
M0 = (M0 + M0.T) / 2.0
M1 = (M1 + M1.T) / 2.0
N0 = (N0 + N0.T) / 2.0
N1 = (N1 + N1.T) / 2.0

def _objective(a):
    """
    Compute the uncertainty bound for coefficients a = [a0, a4, a8].
    """
    if np.any(np.abs(a) < 1e-15):
        # Avoid division by zero if a is zero vector
        pass
        
    # Mx0 = a^T M0 a
    Mx0 = a @ M0 @ a
    Mx1 = a @ M1 @ a
    Mxi0 = a @ N0 @ a
    Mxi1 = a @ N1 @ a
    
    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return 1e18
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18
        
    return bound

def _inner_optimize_s(r):
    """
    For a fixed ratio r = a4/a0, optimize s = a8/a0.
    We set a0 = 1, a4 = r, a8 = s.
    Returns the optimal s and the minimum value.
    """
    def f(s):
        a = np.array([1.0, r, s])
        return _objective(a)
    
    # Initial guess for s based on seed or previous best
    # Seed: a0=-1.13, a4=0.04, a8=0.000308642
    # r_seed = 0.04 / -1.13 approx -0.035
    # s_seed = 0.000308642 / -1.13 approx -0.00027
    
    # We will use a bounded search for s to ensure robustness
    # s is expected to be small
    try:
        res = minimize_scalar(f, bounds=(-0.01, 0.01), method='bounded', options={'xatol': 1e-10})
        if res.success:
            return res.x, res.fun
        else:
            # Fallback to grid
            s_grid = np.linspace(-0.01, 0.01, 100)
            vals = [f(s) for s in s_grid]
            idx = np.argmin(vals)
            return s_grid[idx], vals[idx]
    except Exception:
        return 0.0, 1e18

def _outer_optimize_r(r_range=(-5.0, 5.0), n_grid=200):
    """
    Sweep r = a4/a0 and find the global minimum.
    """
    r_grid = np.linspace(r_range[0], r_range[1], n_grid)
    best_val = 1e18
    best_r = 0.0
    best_s = 0.0
    
    for r in r_grid:
        if time.time() > _DEADLINE - 10:
            break
        s_opt, val = _inner_optimize_s(r)
        if val < best_val:
            best_val = val
            best_r = r
            best_s = s_opt
            
    # Refine around best_r
    if best_val < 1e18:
        # Determine local interval
        idx = np.argmin(np.abs(r_grid - best_r))
        if idx > 0 and idx < len(r_grid) - 1:
            r_min = r_grid[idx-1]
            r_max = r_grid[idx+1]
        else:
            r_min = best_r - 0.1
            r_max = best_r + 0.1
            
        def f_r(r):
            s_opt, val = _inner_optimize_s(r)
            return val
        
        try:
            res = minimize_scalar(f_r, bounds=(r_min, r_max), method='bounded', options={'xatol': 1e-8})
            if res.success and res.fun < best_val:
                best_r = res.x
                best_s, best_val = _inner_optimize_s(best_r)
        except Exception:
            pass
            
    # Construct final a vector
    # We can scale a0 to anything, let's set a0 = 1 for the direction
    # But the evaluator might expect specific magnitudes? 
    # The problem statement says "score is the upper bound", which is scale invariant.
    # However, the seed returns specific values. Let's return the direction scaled to match seed magnitude or just [1, r, s].
    # The interface says "three coefficients".
    # Let's return [1.0, best_r, best_s]
    
    return np.array([1.0, best_r, best_s]), best_val

def construct():
    """
    Find three coefficients (a0, a4, a8) that minimize the uncertainty product.
    Returns array of shape (3,).
    """
    
    # Seed value
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val = _objective(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    deadline_margin = 10  # seconds
    
    # 1. Try the 1D global search strategy
    if time.time() < _DEADLINE - deadline_margin:
        try:
            # Broad search
            params_1d, val_1d = _outer_optimize_r(r_range=(-10.0, 10.0), n_grid=100)
            if np.isfinite(val_1d) and val_1d < best_val:
                best_val = val_1d
                best_params = params_1d.copy()
                
                # Refined search around the best found
                r_best = params_1d[1]
                # Local refinement
                params_ref, val_ref = _outer_optimize_r(r_range=(r_best - 1.0, r_best + 1.0), n_grid=50)
                if np.isfinite(val_ref) and val_ref < best_val:
                    best_val = val_ref
                    best_params = params_ref.copy()
                    
        except Exception:
            pass

    # 2. If time remains, do a local polish from the best found using 3D L-BFGS
    if time.time() < _DEADLINE - deadline_margin:
        try:
            from scipy.optimize import minimize
            
            # We need a gradient for L-BFGS. We can use finite differences or analytic.
            # Let's use analytic gradient derived from the matrix forms.
            
            def _obj_and_grad(a):
                val = _objective(a)
                if val > 1e17:
                    return val, np.zeros(3)
                
                Mx0 = a @ M0 @ a
                Mx1 = a @ M1 @ a
                Mxi0 = a @ N0 @ a
                Mxi1 = a @ N1 @ a
                
                vx = Mx1 / Mx0
                vxi = Mxi1 / Mxi0
                bound = np.sqrt(vx * vxi)
                
                # d(bound) = 1/(2*bound) * (vxi * d(vx) + vx * d(vxi))
                # d(vx) = (d(Mx1/Mx0)) = (dMx1*Mx0 - Mx1*dMx0) / Mx0^2
                # dMx0 = 2 * (M0 a) . da
                # dMx1 = 2 * (M1 a) . da
                
                dMx0 = 2.0 * (M0 @ a)
                dMx1 = 2.0 * (M1 @ a)
                dMxi0 = 2.0 * (N0 @ a)
                dMxi1 = 2.0 * (N1 @ a)
                
                dvx_da = (dMx1 * Mx0 - Mx1 * dMx0) / (Mx0 * Mx0)
                dvxi_da = (dMxi1 * Mxi0 - Mxi1 * dMxi0) / (Mxi0 * Mxi0)
                
                grad = (1.0 / (2.0 * bound)) * (vxi * dvx_da + vx * dvxi_da)
                
                return bound, grad

            res = minimize(_obj_and_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if np.isfinite(res.fun) and res.fun < best_val:
                candidate_a = res.x
                val_check = _objective(candidate_a)
                if val_check < best_val:
                    best_val = val_check
                    best_params = candidate_a.copy()
        except Exception:
            pass

    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
