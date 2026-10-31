import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195

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

# H0 coeffs: [1, 0, 0, 0, 0, 0, 0]
# H4 coeffs: [3, -6, 1, 0, 0, 0, 0] (order x^0, x^2, x^4)
# H8 coeffs: [105, -420, 210, -28, 1, 0, 0]
# H12 coeffs: [945, -7560, 13860, -6930, 1155, -66, 1]

H0_c = np.array([1, 0, 0, 0, 0, 0, 0])
H4_c = np.array([3, -6, 1, 0, 0, 0, 0])
H8_c = np.array([105, -420, 210, -28, 1, 0, 0])
H12_c = np.array([945, -7560, 13860, -6930, 1155, -66, 1])

# a12 is linear in a: a12 = c12_0*a0 + c12_4*a4 + c12_8*a8
c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

# p = a0*H0_c + a4*H4_c + a8*H8_c + a12*H12_c
# p = a0*(H0_c + c12_0*H12_c) + a4*(H4_c + c12_4*H12_c) + a8*(H8_c + c12_8*H12_c)

col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c

A_mat = np.column_stack((col0, col1, col2)) # Shape (7, 3)

def _get_p(a):
    return A_mat @ a

def _build_moment_matrices():
    """
    Build the moment matrices for position and momentum in the Hermite basis.
    
    Mx[n] = sum_{i,j} p_i p_j I[i+j+n] = a^T Bx[n] a
    Mxi[n] = sum_{i,j} q_i q_j I[i+j+n] = a^T Bxi[n] a
    
    where q_k = p_k * (-1)^k
    """
    sign_vec = np.array([1, -1, 1, -1, 1, -1, 1])
    
    # Bx[n] is a 3x3 matrix such that a^T Bx[n] a = sum_{i,j} (A a)_i (A a)_j I[i+j+n]
    # = a^T A^T diag-ish A a where the "diag" involves I[i+j+n]
    # Bx[n] = A^T @ K_n @ A where K_n[i,j] = I[i+j+n]
    
    Bx = np.zeros((3, 3, 2))  # Bx[:, :, n]
    Bxi = np.zeros((3, 3, 2))  # Bxi[:, :, n]
    
    # For momentum, q_i = p_i * sign_vec[i], so:
    # Mxi[n] = sum_{i,j} q_i q_j I[i+j+n] = sum_{i,j} p_i p_j sign_vec[i] sign_vec[j] I[i+j+n]
    # = a^T A^T K_n' A a where K_n'[i,j] = sign_vec[i] sign_vec[j] I[i+j+n]
    
    for n in range(2):
        # Build K_n matrix (7x7)
        K_n = np.zeros((7, 7))
        K_n_prime = np.zeros((7, 7))
        for i in range(7):
            for j in range(7):
                K_n[i, j] = I[i + j + n]
                K_n_prime[i, j] = sign_vec[i] * sign_vec[j] * I[i + j + n]
        
        # Bx[:, :, n] = A^T @ K_n @ A
        Bx[:, :, n] = A_mat.T @ K_n @ A_mat
        Bxi[:, :, n] = A_mat.T @ K_n_prime @ A_mat
    
    return Bx, Bxi

def _compute_bound_from_matrices(a, Bx, Bxi):
    """
    Compute the uncertainty bound given coefficients a and precomputed matrices.
    """
    Mx0 = a @ Bx[:, :, 0] @ a
    Mx1 = a @ Bx[:, :, 1] @ a
    Mxi0 = a @ Bxi[:, :, 0] @ a
    Mxi1 = a @ Bxi[:, :, 1] @ a
    
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

def _solve_via_eigenvalue():
    """
    Solve the optimization as a generalized eigenvalue problem.
    
    We want to minimize sqrt((a^T Bx1 a / a^T Bx0 a) * (a^T Bxi1 a / a^T Bxi0 a))
    
    This is equivalent to minimizing the product of two Rayleigh quotients.
    We can use an iterative approach or try to find a closed form.
    
    Alternative: parameterize a = r * v where v is on the unit sphere and optimize over v.
    But the product structure makes this non-trivial for exact eigenvalue solution.
    
    Let's use a different approach: fix one ratio and optimize the other, or use
    a sequential quadratic programming approach.
    
    Actually, let's use the fact that we have only 3 parameters. We can:
    1. Normalize a to have ||a|| = 1 (since the bound is homogeneous of degree 0)
    2. Use spherical coordinates or direct 3D optimization
    3. Try multiple initial points with L-BFGS-B
    """
    Bx, Bxi = _build_moment_matrices()
    
    # The bound is homogeneous: f(λa) = f(a) for any λ ≠ 0
    # So we can fix ||a||_2 = 1 and optimize over the unit sphere in R^3
    
    # Use spherical coordinates: a = [sin(θ)cos(φ), sin(θ)sin(φ), cos(θ)]
    # Or just use 2 parameters with a constraint
    
    # Let's use direct optimization with normalization
    def _obj_and_grad(params):
        """
        params: (θ, φ) spherical angles
        a = [sin(θ)cos(φ), sin(θ)sin(φ), cos(θ)]
        """
        theta, phi = params
        a = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        val = _compute_bound_from_matrices(a, Bx, Bxi)
        
        # Compute gradient via finite differences or analytically
        # For simplicity, use finite differences
        eps = 1e-8
        grad = np.zeros(2)
        
        # d/dθ
        params_plus = [theta + eps, phi]
        a_plus = np.array([
            np.sin(params_plus[0]) * np.cos(params_plus[1]),
            np.sin(params_plus[0]) * np.sin(params_plus[1]),
            np.cos(params_plus[0])
        ])
        val_plus = _compute_bound_from_matrices(a_plus, Bx, Bxi)
        grad[0] = (val_plus - val) / eps
        
        # d/dφ
        params_plus = [theta, phi + eps]
        a_plus = np.array([
            np.sin(params_plus[0]) * np.cos(params_plus[1]),
            np.sin(params_plus[0]) * np.sin(params_plus[1]),
            np.cos(params_plus[0])
        ])
        val_plus = _compute_bound_from_matrices(a_plus, Bx, Bxi)
        grad[1] = (val_plus - val) / eps
        
        return val, grad
    
    # Multiple random starts on the sphere
    rng = np.random.default_rng(123)
    best_val = 1e18
    best_params = None
    
    # Seed-based start
    seed_a = np.array([-1.13, 0.04, 0.000308642])
    seed_a_norm = seed_a / np.linalg.norm(seed_a)
    
    # Convert to spherical
    theta_seed = np.arccos(np.clip(seed_a_norm[2], -1, 1))
    phi_seed = np.arctan2(seed_a_norm[1], seed_a_norm[0])
    
    starts = [(theta_seed, phi_seed)]
    
    # Random starts
    for _ in range(200):
        theta = rng.uniform(0, np.pi)
        phi = rng.uniform(-np.pi, np.pi)
        starts.append((theta, phi))
    
    # Structured starts: try different sign patterns
    for s0 in [-1, 1]:
        for s4 in [-1, 1]:
            for s8 in [-1, 1]:
                a_test = np.array([s0, s4 * 0.1, s8 * 0.01])
                a_test = a_test / np.linalg.norm(a_test)
                theta = np.arccos(np.clip(a_test[2], -1, 1))
                phi = np.arctan2(a_test[1], a_test[0])
                starts.append((theta, phi))
    
    # Optimize from each start
    for start in starts:
        if time.time() > _DEADLINE - 10:
            break
            
        try:
            res = minimize(_obj_and_grad, start, method='L-BFGS-B',
                          jac=True, bounds=[(0, np.pi), (-np.pi, np.pi)],
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if np.isfinite(res.fun) and res.fun < best_val:
                theta, phi = res.x
                a = np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])
                best_val = res.fun
                best_params = a.copy()
        except Exception:
            pass
    
    # Convert back to original scale (the bound is scale-invariant, so any scale works)
    # But we need to return coefficients that give the same bound
    # The original seed had specific magnitudes, so let's scale to match typical magnitudes
    if best_params is not None:
        # Scale to have similar magnitude to seed
        seed_mag = np.linalg.norm(np.array([-1.13, 0.04, 0.000308642]))
        current_mag = np.linalg.norm(best_params)
        if current_mag > 1e-15:
            scale = seed_mag / current_mag
            best_params = best_params * scale
    
    return best_params, best_val

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
    # We can compute this efficiently using the I array
    
    Mx = np.zeros(2)
    Mxi = np.zeros(2)
    
    for i in range(7):
        pi = p[i]
        if pi == 0: continue
        for j in range(7):
            idx0 = i + j
            idx1 = i + j + 1
            Mx[0] += pi * p[j] * I[idx0]
            Mx[1] += pi * p[j] * I[idx1]
            
            idx0_q = i + j
            idx1_q = i + j + 1
            Mxi[0] += q[i] * q[j] * I[idx0_q]
            Mxi[1] += q[i] * q[j] * I[idx1_q]
            
    if Mx[0] <= 1e-15 or Mxi[0] <= 1e-15:
        return 1e18, np.zeros(3)
        
    vx = Mx[1] / Mx[0]
    vxi = Mxi[1] / Mxi[0]
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
        
    # Gradient computation
    # J = 0.5 * (vx * vxi)^0.5
    # dJ = 0.5 * (vx * vxi)^-0.5 * (d(vx*vxi))
    # d(vx*vxi) = vxi * dvx + vx * dvxi
    # dvx = d(Mx1/Mx0) = (dMx1*Mx0 - Mx1*dMx0) / Mx0^2
    # dvxi = d(Mxi1/Mxi0) = (dMxi1*Mxi0 - Mxi1*dMxi0) / Mxi0^2
    
    # dM[n] = 2 * sum_{i,j} (dp_i * p_j) I[i+j+n]
    # dp_i = sum_m A[i,m] * da_m
    # So dM[n]_m = 2 * sum_i A[i,m] sum_j p_j I[i+j+n]
    
    # Let S[n]_i = sum_j p_j I[i+j+n]
    Sx = np.zeros((2, 7))
    Sxi = np.zeros((2, 7))
    
    for n in range(2):
        for i in range(7):
            Sx[n, i] = np.dot(p, I[i+n:i+n+7])
            Sxi[n, i] = np.dot(q, I[i+n:i+n+7])
            
    # dM[n] is a vector of 3 components (for each a_m)
    # dM[n]_m = 2 * sum_i A[i,m] S[n, i]
    dMx = np.zeros((2, 3))
    dMxi = np.zeros((2, 3))
    
    for m in range(3):
        for n in range(2):
            dMx[n, m] = 2.0 * np.dot(A_mat[:, m], Sx[n, :])
            dMxi[n, m] = 2.0 * np.dot(A_mat[:, m], Sxi[n, :])
            
    # dvx_m = (dMx[1,m]*Mx[0] - Mx[1]*dMx[0,m]) / Mx[0]^2
    dvx = np.zeros(3)
    dvxi = np.zeros(3)
    
    for m in range(3):
        dvx[m] = (dMx[1, m]*Mx[0] - Mx[1]*dMx[0, m]) / (Mx[0]*Mx[0])
        dvxi[m] = (dMxi[1, m]*Mxi[0] - Mxi[1]*dMxi[0, m]) / (Mxi[0]*Mxi[0])
        
    # d(bound) = (1/(2*bound)) * (vxi*dvx + vx*dvxi)
    grad = np.zeros(3)
    factor = 1.0 / (2.0 * bound)
    for m in range(3):
        grad[m] = factor * (vxi*dvx[m] + vx*dvxi[m])
        
    return bound, grad

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
    
    deadline_margin = 5  # seconds
    
    # Try the eigenvalue/spherical approach first
    if time.time() < _DEADLINE - deadline_margin:
        try:
            eig_params, eig_val = _solve_via_eigenvalue()
            if eig_params is not None and np.isfinite(eig_val) and eig_val < best_val:
                # Verify
                check_val, _ = _compute_bound_and_grad(eig_params)
                if check_val < best_val:
                    best_val = check_val
                    best_params = eig_params.copy()
        except Exception:
            pass
    
    # Keep the original log-sign optimization as backup
    def _log_sign_obj(ts):
        m0, m4, m8 = ts[0], ts[1], ts[2]
        t0, t4, t8 = ts[3], ts[4], ts[5]
        
        s0 = np.tanh(t0)
        s4 = np.tanh(t4)
        s8 = np.tanh(t8)
        
        a0 = s0 * np.exp(m0)
        a4 = s4 * np.exp(m4)
        a8 = s8 * np.exp(m8)
        
        a = np.array([a0, a4, a8])
        val, _ = _compute_bound_and_grad(a)
        return val
    
    def _log_sign_grad(ts):
        m0, m4, m8 = ts[0], ts[1], ts[2]
        t0, t4, t8 = ts[3], ts[4], ts[5]
        
        s0 = np.tanh(t0)
        s4 = np.tanh(t4)
        s8 = np.tanh(t8)
        
        a0 = s0 * np.exp(m0)
        a4 = s4 * np.exp(m4)
        a8 = s8 * np.exp(m8)
        
        a = np.array([a0, a4, a8])
        val, grad_a = _compute_bound_and_grad(a)
        
        grad_ts = np.zeros(6)
        grad_ts[0] = grad_a[0] * a0
        grad_ts[1] = grad_a[1] * a4
        grad_ts[2] = grad_a[2] * a8
        
        ds0 = 1.0 - s0 * s0
        ds4 = 1.0 - s4 * s4
        ds8 = 1.0 - s8 * s8
        
        grad_ts[3] = grad_a[0] * s0 * np.exp(m0) * ds0
        grad_ts[4] = grad_a[1] * s4 * np.exp(m4) * ds4
        grad_ts[5] = grad_a[2] * s8 * np.exp(m8) * ds8
        
        return val, grad_ts
    
    def _to_log_sign(a):
        ts = np.zeros(6)
        for i, ai in enumerate(a):
            if abs(ai) < 1e-15:
                ts[i] = -10.0
                ts[i+3] = 0.0
            else:
                ts[i] = np.log(abs(ai))
                ts[i+3] = np.arctanh(np.sign(ai))
        return ts
    
    seed_ts = _to_log_sign(seed_params)
    
    starts = [seed_ts.copy()]
    
    rng = np.random.default_rng(42)
    
    for _ in range(100):
        ts = np.zeros(6)
        ts[0] = rng.uniform(-8, 2)
        ts[1] = rng.uniform(-8, 0)
        ts[2] = rng.uniform(-8, -2)
        ts[3] = rng.uniform(-3, 3)
        ts[4] = rng.uniform(-3, 3)
        ts[5] = rng.uniform(-3, 3)
        starts.append(ts)
    
    for sign0 in [-1, 1]:
        for sign4 in [-1, 1]:
            for sign8 in [-1, 1]:
                ts = np.array([
                    np.log(0.5),
                    np.log(0.01),
                    np.log(0.001),
                    np.arctanh(sign0 * 0.9),
                    np.arctanh(sign4 * 0.9),
                    np.arctanh(sign8 * 0.9)
                ])
                starts.append(ts)
    
    rng2 = np.random.default_rng(99)
    perm = np.arange(len(starts))
    perm[1:] = rng2.permutation(perm[1:])
    starts = [starts[i] for i in perm]
    
    for start in starts:
        if time.time() > _DEADLINE - deadline_margin:
            break
            
        try:
            res = minimize(_log_sign_obj, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if np.isfinite(res.fun) and res.fun < best_val:
                m0, m4, m8 = res.x[0], res.x[1], res.x[2]
                t0, t4, t8 = res.x[3], res.x[4], res.x[5]
                
                s0 = np.tanh(t0)
                s4 = np.tanh(t4)
                s8 = np.tanh(t8)
                
                a0 = s0 * np.exp(m0)
                a4 = s4 * np.exp(m4)
                a8 = s8 * np.exp(m8)
                
                candidate_a = np.array([a0, a4, a8])
                val_check, _ = _compute_bound_and_grad(candidate_a)
                if val_check < best_val:
                    best_val = val_check
                    best_params = candidate_a.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < _DEADLINE - deadline_margin:
        try:
            best_ts = _to_log_sign(best_params)
            res = minimize(_log_sign_obj, best_ts, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if np.isfinite(res.fun) and res.fun < best_val:
                m0, m4, m8 = res.x[0], res.x[1], res.x[2]
                t0, t4, t8 = res.x[3], res.x[4], res.x[5]
                
                s0 = np.tanh(t0)
                s4 = np.tanh(t4)
                s8 = np.tanh(t8)
                
                a0 = s0 * np.exp(m0)
                a4 = s4 * np.exp(m4)
                a8 = s8 * np.exp(m8)
                
                candidate_a = np.array([a0, a4, a8])
                val_check, _ = _compute_bound_and_grad(candidate_a)
                if val_check < best_val:
                    best_val = val_check
                    best_params = candidate_a.copy()
        except Exception:
            pass

    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
