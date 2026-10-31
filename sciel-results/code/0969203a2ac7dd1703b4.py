import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import eigh

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

H0_c = np.array([1.0, 0, 0, 0, 0, 0, 0])
H4_c = np.array([3.0, -6, 1, 0, 0, 0, 0])
H8_c = np.array([105.0, -420, 210, -28, 1, 0, 0])
H12_c = np.array([945.0, -7560, 13860, -6930, 1155, -66, 1])

# a12 is linear in a: a12 = c12_0*a0 + c12_4*a4 + c12_8*a8
c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c

A_mat = np.column_stack((col0, col1, col2)) # Shape (7, 3)

sign_vec = np.array([1.0, -1, 1, -1, 1, -1, 1])

def _get_p(a):
    return A_mat @ a

def _compute_variances(a):
    """
    Compute Vx and Vxi.
    Returns (Vx, Vxi, p, q)
    """
    p = _get_p(a)
    q = p * sign_vec
    
    Mx0 = 0.0
    Mx1 = 0.0
    Mxi0 = 0.0
    Mxi1 = 0.0
    
    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0 and qi == 0:
            continue
        for j in range(7):
            idx0 = i + j
            idx1 = i + j + 1
            I0 = I[idx0]
            I1 = I[idx1]
            
            Mx0 += pi * p[j] * I0
            Mx1 += pi * p[j] * I1
            
            Mxi0 += qi * q[j] * I0
            Mxi1 += qi * q[j] * I1
            
    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return 1e18, 1e18, p, q
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18, 1e18, p, q
        
    return vx, vxi, p, q

def _compute_bound_and_grad(a):
    """
    Compute the uncertainty bound J = sqrt(Vx * Vxi) and its gradient.
    """
    vx, vxi, p, q = _compute_variances(a)
    
    if vx >= 1e17 or vxi >= 1e17:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
        
    # Gradient computation
    Sx = np.zeros((2, 7))
    Sxi = np.zeros((2, 7))
    
    for n in range(2):
        for i in range(7):
            Sx[n, i] = np.dot(p, I[i+n:i+n+7])
            Sxi[n, i] = np.dot(q, I[i+n:i+n+7])
            
    Mx0_calc = np.dot(p, Sx[0])
    Mx1_calc = np.dot(p, Sx[1])
    Mxi0_calc = np.dot(q, Sxi[0])
    Mxi1_calc = np.dot(q, Sxi[1])
    
    if Mx0_calc <= 1e-15 or Mxi0_calc <= 1e-15:
        return 1e18, np.zeros(3)
        
    vx_calc = Mx1_calc / Mx0_calc
    vxi_calc = Mxi1_calc / Mxi0_calc
    
    dMx = np.zeros((2, 3))
    dMxi = np.zeros((2, 3))
    
    for m in range(3):
        for n in range(2):
            dMx[n, m] = 2.0 * np.dot(A_mat[:, m], Sx[n, :])
            dMxi[n, m] = 2.0 * np.dot(A_mat[:, m], Sxi[n, :])
            
    dvx = np.zeros(3)
    dvxi = np.zeros(3)
    
    for m in range(3):
        dvx[m] = (dMx[1, m]*Mx0_calc - Mx1_calc*dMx[0, m]) / (Mx0_calc*Mx0_calc)
        dvxi[m] = (dMxi[1, m]*Mxi0_calc - Mxi1_calc*dMxi[0, m]) / (Mxi0_calc*Mxi0_calc)
        
    grad = np.zeros(3)
    factor = 1.0 / (2.0 * bound)
    for m in range(3):
        grad[m] = factor * (vxi_calc*dvx[m] + vx_calc*dvxi[m])
        
    return bound, grad

def _build_quadratic_forms():
    """
    Build the quadratic form matrices for Vx and Vxi.
    
    Vx = (p^T M1 p) / (p^T M0 p) where M0[n,m] = I[n+m], M1[n,m] = I[n+m+1]
    Vxi = (q^T M1 q) / (q^T M0 q) where q = D*p, D=diag(sign_vec)
    
    Since p = A_mat * a, we have:
    Vx = (a^T A^T M1 A a) / (a^T A^T M0 A a)
    Vxi = (a^T A^T D M1 D A a) / (a^T A^T D M0 D A a)
    
    So the uncertainty product squared is:
    (Vx * Vxi) = [(a^T B1 a)(a^T C1 a)] / [(a^T B0 a)(a^T C0 a)]
    
    where B1 = A^T M1 A, B0 = A^T M0 A, C1 = A^T D M1 D A, C0 = A^T D M0 D A
    """
    # Build M0 and M1 matrices (7x7)
    M0 = np.zeros((7, 7))
    M1 = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            M0[i, j] = I[i + j]
            M1[i, j] = I[i + j + 1]
    
    # D = diag(sign_vec)
    D = np.diag(sign_vec)
    
    # B0 = A^T M0 A, B1 = A^T M1 A
    B0 = A_mat.T @ M0 @ A_mat
    B1 = A_mat.T @ M1 @ A_mat
    
    # C0 = A^T D M0 D A, C1 = A^T D M1 D A
    C0 = A_mat.T @ (D @ M0 @ D) @ A_mat
    C1 = A_mat.T @ (D @ M1 @ D) @ A_mat
    
    return B0, B1, C0, C1

def _solve_secular_equation(B0, B1, C0, C1):
    """
    Solve for the optimal Lagrange multiplier lambda that minimizes Vx * Vxi.
    
    At the optimum, the gradient of log(Vx) + log(Vxi) = 0, which gives:
    B1 a / (a^T B0 a) + C1 a / (a^T C0 a) = lambda * (B0 a / (a^T B0 a) + C0 a / (a^T C0 a))
    
    This is a generalized eigenvalue problem. We can reformulate as:
    Find lambda such that det(B1 - lambda*B0 + C1 - lambda*C0) = 0
    with the constraint that the resulting eigenvector gives positive denominators.
    
    Actually, the correct approach: minimize f(a) = (a^T B1 a)(a^T C1 a) / ((a^T B0 a)(a^T C0 a))
    subject to some normalization. The stationary condition leads to a secular equation.
    
    A simpler approach: use the fact that at the optimum, there exists lambda such that:
    (B1 - lambda B0) a = -mu (C1 - lambda C0) a
    for some mu > 0. This is a coupled eigenvalue problem.
    
    Let's use a different approach: parameterize by the ratio and solve a 3x3 secular equation.
    """
    # The objective is f(a) = (a^T B1 a)(a^T C1 a) / ((a^T B0 a)(a^T C0 a))
    # By homogeneity, we can normalize a^T B0 a = 1.
    # Then minimize g(a) = (a^T B1 a)(a^T C1 a) / (a^T C0 a) subject to a^T B0 a = 1.
    
    # Using Lagrange multipliers:
    # grad((a^T B1 a)(a^T C1 a) / (a^T C0 a)) = lambda * grad(a^T B0 a)
    
    # This leads to: 2*(C1 a / (a^T C0 a)) * (B1 a) + 2*(B1 a / (a^T C0 a)) * (C1 a) - 
    #                 2*(a^T B1 a)(a^T C1 a) * (C0 a) / (a^T C0 a)^2 = 2*lambda * B0 a
    
    # This is complex. Let's use a numerical approach with the quadratic forms.
    
    # Alternative: use scipy to solve the constrained optimization directly with the quadratic forms
    # This should be much faster than the original approach since we're working with 3x3 matrices.
    
    def _obj_and_grad(a):
        a = np.asarray(a)
        b0_val = a @ B0 @ a
        b1_val = a @ B1 @ a
        c0_val = a @ C0 @ a
        c1_val = a @ C1 @ a
        
        if b0_val <= 1e-15 or c0_val <= 1e-15:
            return 1e18, np.zeros(3)
        
        vx = b1_val / b0_val
        vxi = c1_val / c0_val
        
        if vx <= 1e-15 or vxi <= 1e-15:
            return 1e18, np.zeros(3)
        
        bound = np.sqrt(vx * vxi)
        
        # Gradient: d/dx [sqrt((x^T B1 x)(x^T C1 x) / ((x^T B0 x)(x^T C0 x)))]
        # Let F = (x^T B1 x)(x^T C1 x), G = (x^T B0 x)(x^T C0 x)
        # f = sqrt(F/G)
        # df/dx = (1/(2*sqrt(F/G))) * (F'/G - F*G'/G^2)
        #        = (1/(2*bound)) * (F'/G - F*G'/G^2)
        
        # F' = 2*B1*x * (x^T C1 x) + 2*C1*x * (x^T B1 x)
        # G' = 2*B0*x * (x^T C0 x) + 2*C0*x * (x^T B0 x)
        
        F = b1_val * c1_val
        G = b0_val * c0_val
        
        F_grad = 2.0 * (B1 @ a) * c1_val + 2.0 * (C1 @ a) * b1_val
        G_grad = 2.0 * (B0 @ a) * c0_val + 2.0 * (C0 @ a) * b0_val
        
        factor = 1.0 / (2.0 * bound)
        grad = factor * (F_grad / G - F * G_grad / (G * G))
        
        return bound, grad
    
    return _obj_and_grad

def construct():
    """
    Find three coefficients (a0, a4, a8) that minimize the uncertainty product.
    Returns array of shape (3,).
    """
    
    # Build the quadratic form matrices
    B0, B1, C0, C1 = _build_quadratic_forms()
    
    # Get the objective function and gradient using the quadratic forms
    _qf_obj_grad = _solve_secular_equation(B0, B1, C0, C1)
    
    # Seed value
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val, _ = _qf_obj_grad(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    deadline_margin = 10  # seconds
    
    # Generate diverse starting points
    np.random.seed(42)
    starts = [seed_params.copy()]
    
    # Random starts
    for _ in range(30):
        start = np.random.uniform(-5, 5, 3)
        starts.append(start)
        
    # Structured starts around the seed
    for a0 in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
        for a4 in [-0.1, -0.05, 0, 0.05, 0.1]:
            for a8 in [-0.01, -0.005, 0, 0.005, 0.01]:
                starts.append(np.array([a0, a4, a8]))
    
    # Sort starts by initial objective value
    def _init_score(s):
        v, _ = _qf_obj_grad(s)
        return v
    
    starts.sort(key=_init_score)
    
    # Run local minimization from each start
    for start in starts:
        if time.time() > _DEADLINE - deadline_margin:
            break
            
        try:
            res_local = minimize(_qf_obj_grad, start, 
                                 jac=lambda x: _qf_obj_grad(x)[1], 
                                 method='L-BFGS-B', 
                                 options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-15})
            if np.isfinite(res_local.fun) and res_local.fun < best_val:
                best_val = res_local.fun
                best_params = res_local.x.copy()
        except Exception:
            pass
    
    # High-precision polish
    if time.time() < _DEADLINE - deadline_margin and best_val < seed_val:
        try:
            res_polish = minimize(_qf_obj_grad, best_params, 
                                  jac=lambda x: _qf_obj_grad(x)[1], 
                                  method='L-BFGS-B', 
                                  options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-17})
            if np.isfinite(res_polish.fun) and res_polish.fun < best_val:
                best_val = res_polish.fun
                best_params = res_polish.x.copy()
        except Exception:
            pass
    
    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
