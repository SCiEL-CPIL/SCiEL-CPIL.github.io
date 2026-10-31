import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import factorial

_START = time.time()
_DEADLINE = _START + 180.0
_MARGIN = 5.0

def _time_left():
    return _DEADLINE - time.time() - _MARGIN

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

H0_c = np.array([1, 0, 0, 0, 0, 0, 0])
H4_c = np.array([3, -6, 1, 0, 0, 0, 0])
H8_c = np.array([105, -420, 210, -28, 1, 0, 0])
H12_c = np.array([945, -7560, 13860, -6930, 1155, -66, 1])

# a12 = -(a0 + 3*a4 + 105*a8) / 945 to ensure P(0)=0
# P(0) = a0*1 + a4*3 + a8*105 + a12*945 = 0
c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c

A_mat = np.column_stack((col0, col1, col2))  # Shape (7, 3)

# D = diag(1, -1, 1, -1, 1, -1, 1)
D = np.diag(np.array([1, -1, 1, -1, 1, -1, 1]))
A_xi = D @ A_mat

# W_0[i,j] = I[i+j]
# W_1[i,j] = I[i+j+1]
W_0 = np.zeros((7, 7))
W_1 = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        W_0[i, j] = I[i+j]
        W_1[i, j] = I[i+j+1]

N_x = A_mat.T @ W_0 @ A_mat
M_x = A_mat.T @ W_1 @ A_mat
N_xi = A_xi.T @ W_0 @ A_xi
M_xi = A_xi.T @ W_1 @ A_xi

def _compute_bound(a):
    """
    Compute the uncertainty bound for given coefficients a = [a0, a4, a8].
    """
    # Quadratic forms
    Nx = np.dot(a, N_x @ a)
    Mx = np.dot(a, M_x @ a)
    Nxi = np.dot(a, N_xi @ a)
    Mxi = np.dot(a, M_xi @ a)
    
    if Nx <= 1e-15 or Nxi <= 1e-15:
        return 1e18
        
    vx = Mx / Nx
    vxi = Mxi / Nxi
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18
        
    return bound

def _compute_bound_and_grad(a):
    """
    Compute the uncertainty bound and its analytic gradient.
    """
    Nx = np.dot(a, N_x @ a)
    Mx = np.dot(a, M_x @ a)
    Nxi = np.dot(a, N_xi @ a)
    Mxi = np.dot(a, M_xi @ a)
    
    if Nx <= 1e-15 or Nxi <= 1e-15:
        return 1e18, np.zeros(3)
        
    vx = Mx / Nx
    vxi = Mxi / Nxi
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
        
    # Gradient of f(a) = sqrt( (a^T M_x a / a^T N_x a) * (a^T M_xi a / a^T N_xi a) )
    # Let R_x = a^T M_x a / a^T N_x a
    # Let R_xi = a^T M_xi a / a^T N_xi a
    # f = sqrt(R_x R_xi)
    # grad f = 1/(2 sqrt(R_x R_xi)) * ( R_xi grad R_x + R_x grad R_xi )
    
    # grad (a^T M a / a^T N a) = (2 a^T M / (a^T N a) - 2 (a^T M a) (a^T N) / (a^T N a)^2)
    # = 2/(a^T N a) * (a^T M - R (a^T N))
    
    gMx = 2.0 * (M_x @ a)
    gNx = 2.0 * (N_x @ a)
    grad_Rx = (gMx / Nx) - (Mx / (Nx * Nx)) * gNx
    
    gMxi = 2.0 * (M_xi @ a)
    gNxi = 2.0 * (N_xi @ a)
    grad_Rxi = (gMxi / Nxi) - (Mxi / (Nxi * Nxi)) * gNxi
    
    grad = (vxi * grad_Rx + vx * grad_Rxi) / (2.0 * bound)
    
    return bound, grad

def construct():
    """
    Find three coefficients (a0, a4, a8) that minimize the uncertainty product.
    Returns array of shape (3,).
    """
    
    # Seed value
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val = _compute_bound(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    # 1. Log-magnitude and sign parameterization
    # We parameterize a = [a0, a4, a8] as:
    # a_i = sign_i * exp(log_mag_i)
    # where sign_i is in {-1, 1} or handled via tanh for smoothness?
    # The finding says "tanh-based sign representation provides smooth transitions".
    # Let's use a vector u of length 3.
    # a_i = exp(u_i) * tanh(u_i + 10) ? No, that forces positive.
    # Better: a_i = exp(u_i) * s_i where s_i is derived from another variable?
    # Or simpler: just optimize in log-space for magnitudes and treat signs as discrete?
    # But the finding says "tanh-based sign".
    # Let's try: a_i = exp(m_i) * tanh(s_i)
    # If s_i is large positive, tanh(s_i) ~ 1. If s_i is large negative, tanh(s_i) ~ -1.
    # This allows smooth sign changes.
    
    def _unparam(u):
        m = u[:3]
        s = u[3:6]
        return np.exp(m) * np.tanh(s)
    
    def _log_obj_and_grad(u):
        a = _unparam(u)
        val, grad_a = _compute_bound_and_grad(a)
        if not np.isfinite(val):
            return val, np.zeros(6)
        
        # Chain rule: grad_u = (grad_a / a) * [exp(m) * tanh(s) * d(m)/d(m) ...]
        # a_i = exp(m_i) * tanh(s_i)
        # d(a_i)/d(m_i) = exp(m_i) * tanh(s_i) = a_i
        # d(a_i)/d(s_i) = exp(m_i) * (1 - tanh(s_i)^2) = a_i * (1 - tanh(s_i)^2)
        
        grad_u = np.zeros(6)
        for i in range(3):
            ai = a[i]
            if abs(ai) < 1e-12:
                continue
            grad_u[i] = grad_a[i] * ai
            grad_u[i+3] = grad_a[i] * ai * (1 - np.tanh(u[i+3])**2)
            
        return val, grad_u
    
    # Initialize u from seed_params
    u_seed = np.zeros(6)
    for i in range(3):
        ai = seed_params[i]
        if abs(ai) < 1e-12:
            u_seed[i] = np.log(1e-3)
            u_seed[i+3] = 0.0
        else:
            u_seed[i] = np.log(abs(ai))
            u_seed[i+3] = np.arctanh(np.sign(ai) * min(1 - 1e-6, 1.0)) # arctanh(1) is inf, so cap it
    
    # 2. Global Search with Differential Evolution in Log-Space
    # Bounds for u:
    # m_i (log magnitude): [-15, 5] (covers 1e-6 to 150)
    # s_i (sign): [-10, 10] (tanh is nearly +/-1)
    bounds_u = [(-15, 5), (-15, 5), (-15, 5), (-10, 10), (-10, 10), (-10, 10)]
    
    if _time_left() > 10:
        try:
            res_de = differential_evolution(
                lambda u: _log_obj_and_grad(u)[0], 
                bounds_u, 
                seed=42, 
                maxiter=200, 
                tol=1e-8,
                popsize=15,
                polish=True,  # Uses L-BFGS-B on the best result
                init='latinhypercube'
            )
            if np.isfinite(res_de.fun):
                u_candidate = res_de.x
                a_candidate = _unparam(u_candidate)
                val_candidate = _compute_bound(a_candidate)
                if val_candidate < best_val:
                    best_val = val_candidate
                    best_params = a_candidate.copy()
        except Exception:
            pass

    # 3. Local Polish with L-BFGS-B in Log-Space from multiple starts
    if _time_left() > 5:
        rng = np.random.default_rng(123)
        # Starts: seed, DE result, and random perturbations
        starts = [u_seed.copy()]
        if 'res_de' in locals() and res_de.success:
            starts.append(res_de.x.copy())
        
        # Add random starts
        for _ in range(10):
            if _time_left() < 2:
                break
            u_rand = np.array([
                rng.uniform(-10, 0), # log mag
                rng.uniform(-10, 0),
                rng.uniform(-10, 0),
                rng.uniform(-5, 5),  # sign
                rng.uniform(-5, 5),
                rng.uniform(-5, 5)
            ])
            starts.append(u_rand)
            
        for u0 in starts:
            if _time_left() < 1:
                break
            try:
                res_lbfgs = minimize(
                    _log_obj_and_grad, 
                    u0, 
                    method='L-BFGS-B', 
                    jac=True, 
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if np.isfinite(res_lbfgs.fun):
                    u_candidate = res_lbfgs.x
                    a_candidate = _unparam(u_candidate)
                    val_candidate = _compute_bound(a_candidate)
                    if val_candidate < best_val:
                        best_val = val_candidate
                        best_params = a_candidate.copy()
            except Exception:
                pass

    # 4. Final Polish in Linear Space (just in case log-space got stuck near boundary)
    if _time_left() > 2:
        try:
            res_final = minimize(
                _compute_bound_and_grad, 
                best_params, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if np.isfinite(res_final.fun):
                a_candidate = res_final.x
                val_candidate = _compute_bound(a_candidate)
                if val_candidate < best_val:
                    best_val = val_candidate
                    best_params = a_candidate.copy()
        except Exception:
            pass

    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
