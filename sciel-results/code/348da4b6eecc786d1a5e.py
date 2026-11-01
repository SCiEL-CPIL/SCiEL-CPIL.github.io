"""
Exact uncertainty bound via generalized eigenvalue problem.
After substituting c12 to enforce g(0)=0, the objective m2/m0 is a ratio of
quadratic forms in the 3 free parameters. The global minimum is the smallest
generalized eigenvalue of (M2, M0), giving an exact closed-form solution.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _precompute_integrals():
    I = np.zeros(13)
    sp = math.sqrt(math.pi)
    for n in range(13):
        num = math.factorial(2 * n)
        den = (4.0 ** n) * math.factorial(n)
        I[n] = (num / den) * sp
    return I

_I_TABLE = _precompute_integrals()

def _build_matrices():
    """
    Build the 3x3 matrices M0 and M2 such that for params = [c0,c4,c8]:
      p = A @ params (7-vector of polynomial coefficients)
      m0 = p^T @ I0 @ p = params^T @ (A^T I0 A) @ params
      m2 = p^T @ I2 @ p = params^T @ (A^T I2 A) @ params
    
    where I_k[i,j] = _I_TABLE[k+i+j] (for indices in range).
    """
    # A maps [c0,c4,c8] -> p (7-vector)
    # c12 = -(c0 + 3c4 + 105c8)/945
    # p[0] = c0 + 3c4 + 105c8 + 945*c12
    # p[1] = -6c4 - 420c8 - 7560*c12
    # p[2] = c4 + 210c8 + 13860*c12
    # p[3] = -28c8 - 6930*c12
    # p[4] = c8 + 1155*c12
    # p[5] = -66*c12
    # p[6] = c12
    
    # dc12/dc = [-1/945, -3/945, -105/945]
    dc12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # Coefficients for each polynomial term from each Hermite basis
    # H0: [1, 0, 0, 0, 0, 0, 0]
    # H4: [3, -6, 1, 0, 0, 0, 0]
    # H8: [105, -420, 210, -28, 1, 0, 0]
    # H12: [945, -7560, 13860, -6930, 1155, -66, 1]
    
    H_coeffs = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [105.0, -420.0, 210.0, -28.0, 1.0, 0.0, 0.0],
        [945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0]
    ])
    
    # p = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # A[i,j] = coefficient of p[i] from param[j]
    A = np.zeros((7, 3))
    for j in range(3):
        # contribution from c_j (j=0->c0, j=1->c4, j=2->c8)
        A[:, j] = H_coeffs[j, :]  # H0 for c0, H4 for c4, H8 for c8
        # contribution from c12
        A[:, j] += dc12[j] * H_coeffs[3, :]
    
    # Build I0 and I2 matrices (7x7)
    I0 = np.zeros((7, 7))
    I2 = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            idx0 = i + j
            idx2 = i + j + 1
            if idx0 < 13:
                I0[i, j] = _I_TABLE[idx0]
            if idx2 < 13:
                I2[i, j] = _I_TABLE[idx2]
    
    M0 = A.T @ I0 @ A
    M2 = A.T @ I2 @ A
    
    return M0, M2

def _objective(params):
    """Compute the uncertainty bound for given params."""
    c0, c4, c8 = params
    c12 = -(c0 + 3.0 * c4 + 105.0 * c8) / 945.0
    
    p = np.zeros(7)
    p[0] = c0 + 3.0*c4 + 105.0*c8 + 945.0*c12
    p[1] = -6.0*c4 - 420.0*c8 - 7560.0*c12
    p[2] = c4 + 210.0*c8 + 13860.0*c12
    p[3] = -28.0*c8 - 6930.0*c12
    p[4] = c8 + 1155.0*c12
    p[5] = -66.0*c12
    p[6] = c12
    
    m0 = 0.0
    m2 = 0.0
    for i in range(7):
        if p[i] == 0.0:
            continue
        for j in range(7):
            if p[j] == 0.0:
                continue
            idx0 = i + j
            idx2 = i + j + 1
            if idx0 < 13:
                m0 += p[i] * p[j] * _I_TABLE[idx0]
            if idx2 < 13:
                m2 += p[i] * p[j] * _I_TABLE[idx2]
    
    if m0 <= 0 or not np.isfinite(m0):
        return 1e18
    return m2 / m0

def construct():
    start_time = time.time()
    deadline = start_time + 200  # Safety margin before 210s kill
    
    # Build the quadratic form matrices
    M0, M2 = _build_matrices()
    
    # Solve generalized eigenvalue problem: M2 @ v = lambda * M0 @ v
    # The minimum of m2/m0 = v^T M2 v / v^T M0 v is the smallest eigenvalue.
    # Use scipy.linalg.eig for generalized eigenvalues.
    from scipy.linalg import eig
    
    try:
        # eig(M2, M0) solves M2 @ v = lambda * M0 @ v
        eigenvalues, eigenvectors = eig(M2, M0)
        
        # Find the smallest real positive eigenvalue
        best_eig = float('inf')
        best_vec = None
        for i in range(len(eigenvalues)):
            ev = eigenvalues[i]
            if abs(ev.imag) < 1e-10:
                ev_real = ev.real
                if ev_real > 0 and ev_real < best_eig:
                    best_eig = ev_real
                    best_vec = eigenvectors[:, i]
        
        if best_vec is not None:
            # Normalize the eigenvector (scale doesn't matter for the ratio,
            # but we need a specific representative; use the first component as reference)
            # The eigenvector gives the direction [c0, c4, c8]
            best_x = best_vec.real.copy()
            # Scale so that the first non-zero component has a reasonable magnitude
            # Actually, the ratio is scale-invariant, so any scaling works.
            # But we should return a specific point. Let's normalize so ||x||_2 = 1
            norm = np.linalg.norm(best_x)
            if norm > 1e-15:
                best_x = best_x / norm
            # Actually, let's keep a natural scale. The seed used ~1.13 for c0.
            # Let's scale so |c0| ~ 1 if possible
            if abs(best_x[0]) > 1e-10:
                best_x = best_x / abs(best_x[0])
        else:
            # Fallback: use the seed initial guess
            best_x = np.array([-1.13, 0.04, 0.000308642])
    except Exception:
        best_x = np.array([-1.13, 0.04, 0.000308642])
    
    best_score = _objective(best_x)
    
    # Verify with L-BFGS-B polish from the eigenvalue solution
    from scipy.optimize import minimize
    
    def _obj_and_grad(params):
        val = _objective(params)
        # Numerical gradient for robustness
        eps = 1e-8
        grad = np.zeros(3)
        for i in range(3):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            grad[i] = (_objective(params_plus) - _objective(params_minus)) / (2 * eps)
        return val, grad
    
    try:
        res = minimize(_obj_and_grad, best_x, jac=True, method='L-BFGS-B',
                      options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14})
        if res.fun < best_score:
            best_score = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    # Multi-start verification: try many random directions to confirm global minimum
    # Since the problem is a ratio of quadratic forms, the global min is the smallest eigenvalue.
    # But let's verify with random starts to be safe.
    np.random.seed(42)
    for trial in range(200):
        if time.time() > deadline:
            break
        x_rand = np.array([
            np.random.uniform(-10, 10),
            np.random.uniform(-1, 1),
            np.random.uniform(-0.1, 0.1)
        ])
        try:
            res_r = minimize(_obj_and_grad, x_rand, jac=True, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-13})
            if res_r.fun < best_score:
                best_score = res_r.fun
                best_x = res_r.x.copy()
        except Exception:
            pass
    
    # Also try structured starts based on the eigenvector with perturbations
    if best_vec is not None:
        for trial in range(100):
            if time.time() > deadline:
                break
            # Perturb the eigenvector direction
            noise = np.random.normal(0, 0.1, 3)
            x_pert = best_vec.real + noise
            try:
                res_p = minimize(_obj_and_grad, x_pert, jac=True, method='L-BFGS-B',
                               options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-13})
                if res_p.fun < best_score:
                    best_score = res_p.fun
                    best_x = res_p.x.copy()
            except Exception:
                pass
    
    return best_x

# EVOLVE-BLOCK-END
