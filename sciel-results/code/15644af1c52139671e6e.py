"""Uncertainty inequality optimization using exact polynomial-moment integrals.
The objective E2/E0 is a ratio of quadratic forms in the coefficients.
We solve the generalized eigenvalue problem M c = lambda G c subject to the
linear constraint g(0)=0, which reduces to a small constrained eigenvalue
problem. We then polish with a few Newton steps on the constrained manifold."""
import time
import numpy as np
import math

_T0 = time.time()
_DEADLINE = 210.0
_MARGIN = 15.0

def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _T0)

# Exact integrals for Hermite-Gaussian basis h_n(x) = H_n(x) exp(-x^2/2)
# <h_n, h_m> = delta_nm * sqrt(pi) * 2^n * n!
# <x^2 h_n, h_m> = (n+1/2) delta_nm N_n + (1/4) delta_{n+2,m} N_{n+2} + (n(n-1)/4) delta_{n-2,m} N_{n-2}

_sqrt_pi = math.sqrt(math.pi)
N = {}
for n in [0, 4, 8, 12]:
    N[n] = _sqrt_pi * (2 ** n) * math.factorial(n)

degrees = [0, 4, 8, 12]

# G is diagonal
G = np.diag([N[d] for d in degrees])

# M matrix
M = np.zeros((4, 4))
for i, ni in enumerate(degrees):
    M[i, i] = (ni + 0.5) * N[ni]
    if ni + 2 in degrees:
        j = degrees.index(ni + 2)
        M[i, j] += 0.25 * N[ni + 2]
    if ni >= 2 and ni - 2 in degrees:
        j = degrees.index(ni - 2)
        M[i, j] += (ni * (ni - 1) / 4.0) * N[ni - 2]

# Constraint: g(0) = 0 => c0 + 12*c4 + 1344*c8 + 1032192*c12 = 0
# H_0(0)=1, H_4(0)=12, H_8(0)=1344, H_12(0)=1032192
H_vals_at_0 = np.array([1.0, 12.0, 1344.0, 1032192.0])

def _compute_bound(c0, c4, c8):
    """Compute E2/E0 given free coefficients c0, c4, c8."""
    c12 = -(c0 * 1.0 + c4 * 12.0 + c8 * 1344.0) / 1032192.0
    c = np.array([c0, c4, c8, c12])
    E0 = c @ G @ c
    E2 = c @ M @ c
    if E0 < 1e-30:
        return np.inf
    return E2 / E0

def construct():
    try:
        if _time_left() <= 0:
            return np.array([-113.0/100, 1.0/25, 1.0/3240])
        
        # The problem is to minimize (c^T M c) / (c^T G c) subject to a^T c = 0
        # where a = H_vals_at_0.
        # This is a constrained generalized eigenvalue problem.
        # The minimum is the smallest eigenvalue of the pencil (M - lambda G)
        # on the subspace a^T c = 0.
        
        # We can solve this by projecting onto the null space of a^T.
        # Let Q be a matrix whose columns form an orthonormal basis for {c : a^T c = 0}.
        # Then we solve the 3x3 eigenvalue problem Q^T M Q y = lambda Q^T G Q y.
        
        # Construct Q: null space of a^T (a is 1x4)
        # Use SVD of a
        U, s, Vt = np.linalg.svd(H_vals_at_0.reshape(1, -1))
        # The null space is spanned by the last 3 rows of Vt
        Q = Vt[1:].T  # 4x3 matrix
        
        # Form the projected matrices
        MQ = Q.T @ M @ Q  # 3x3
        GQ = Q.T @ G @ Q  # 3x3
        
        # Solve generalized eigenvalue problem MQ y = lambda GQ y
        # Use scipy.linalg.eigh
        from scipy.linalg import eigh
        eigenvalues, eigenvectors = eigh(MQ, GQ)
        
        # The minimum eigenvalue gives the optimal bound
        min_idx = np.argmin(eigenvalues)
        best_eigenvalue = eigenvalues[min_idx]
        best_y = eigenvectors[:, min_idx]
        
        # Convert back to original coordinates
        best_c = Q @ best_y
        
        # Extract free coefficients c0, c4, c8
        best_params = best_c[:3]
        
        # Verify
        val = _compute_bound(best_params[0], best_params[1], best_params[2])
        
        # If the eigenvalue approach didn't give a finite result, fall back
        if not np.isfinite(val) or val > 1.0:
            # Try the seed program's answer
            seed_params = np.array([-113.0/100, 1.0/25, 1.0/3240])
            seed_val = _compute_bound(seed_params[0], seed_params[1], seed_params[2])
            if np.isfinite(seed_val) and seed_val < val:
                best_params = seed_params
                val = seed_val
        
        # Optional: polish with a few Newton steps on the constrained manifold
        # The gradient of the objective with respect to the free parameters is:
        # grad_k = 2 * (M c - lambda G c) @ dc/dc_k / (c^T G c)
        # where lambda = E2/E0.
        # But since we're already at the eigenvalue solution, the gradient should be ~0.
        
        # Let's do a few L-BFGS-B steps to ensure we're at a local minimum
        if _time_left() > 20:
            from scipy.optimize import minimize
            
            def obj(x):
                if _time_left() <= 5:
                    return np.inf
                return _compute_bound(x[0], x[1], x[2])
            
            def grad(x):
                if _time_left() <= 5:
                    return np.zeros(3)
                c12 = -(x[0] * 1.0 + x[1] * 12.0 + x[2] * 1344.0) / 1032192.0
                c = np.array([x[0], x[1], x[2], c12])
                E0 = c @ G @ c
                E2 = c @ M @ c
                if E0 < 1e-30:
                    return np.zeros(3)
                lam = E2 / E0
                # dE0/dx_k = 2 * (G c) @ dc/dx_k
                # dE2/dx_k = 2 * (M c) @ dc/dx_k
                # grad_k = (dE2_k * E0 - E2 * dE0_k) / E0^2
                Gc = G @ c
                Mc = M @ c
                dc12_dx = np.array([-1.0/1032192.0, -12.0/1032192.0, -1344.0/1032192.0])
                g = np.zeros(3)
                for k in range(3):
                    dc_k = np.zeros(4)
                    dc_k[k] = 1.0
                    dc_k[3] = dc12_dx[k]
                    dE0_k = 2.0 * (Gc @ dc_k)
                    dE2_k = 2.0 * (Mc @ dc_k)
                    g[k] = (dE2_k * E0 - E2 * dE0_k) / (E0 ** 2)
                return g
            
            res = minimize(obj, best_params, jac=grad, method='L-BFGS-B',
                          options={'maxiter': 100, 'gtol': 1e-12})
            
            if np.all(np.isfinite(res.x)) and res.fun < val:
                best_params = res.x
                val = res.fun
        
        # Final sanity check
        if not np.all(np.isfinite(best_params)):
            return np.array([-113.0/100, 1.0/25, 1.0/3240])
        
        return best_params
    except Exception:
        return np.array([-113.0/100, 1.0/25, 1.0/3240])
