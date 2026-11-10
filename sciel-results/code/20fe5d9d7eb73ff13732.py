import numpy as np
import time
from scipy.special import eval_legendre, gammaln
from scipy.optimize import minimize

def _precompute():
    """Precompute quadrature nodes, weights, and spherical harmonic basis."""
    k = 10
    
    # Gauss-Legendre quadrature for cos(theta) in [-1, 1]
    n_theta = 70
    x, w = np.polynomial.legendre.leggauss(n_theta)
    
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    # Precompute P_10^m(cos(theta)) for m = 0..10
    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = eval_legendre(10, m, x)
    
    # Normalization constants
    N = np.zeros(11)
    for m in range(11):
        log_N = 0.5 * (np.log(21.0 / (4 * np.pi)) + gammaln(11 - m) - gammaln(11 + m))
        N[m] = np.exp(log_N)
    
    # Build basis matrix B of shape (n_theta * n_phi, 21)
    # We flatten the grid to make matrix multiplication efficient
    total_points = n_theta * n_phi
    B = np.zeros((total_points, 21))
    
    idx = 0
    for i in range(n_theta):
        for j in range(n_phi):
            # m = -10..10
            for m_idx in range(21):
                m = m_idx - 10
                if m == 0:
                    B[idx, m_idx] = N[0] * P[0, i]
                elif m > 0:
                    B[idx, m_idx] = np.sqrt(2) * N[m] * P[m, i] * np.cos(m * phi[j])
                else:
                    am = abs(m)
                    B[idx, m_idx] = np.sqrt(2) * N[am] * P[am, i] * np.sin(am * phi[j])
            idx += 1
    
    # Weights vector
    W = np.zeros(total_points)
    idx = 0
    for i in range(n_theta):
        for j in range(n_phi):
            W[idx] = w[i] * dphi
            idx += 1
    
    return B, W

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # safety margin
    
    B, W = _precompute()
    dim = 21
    n_points = B.shape[0]
    
    # Objective: Minimize ||f||_4^4 subject to ||c||_2^2 = 1
    # f = B @ c
    # J(c) = sum(W * (B @ c)^4)
    
    def objective(c):
        f = B @ c
        f2 = f * f
        return np.sum(W * f2 * f2)
    
    def gradient(c):
        f = B @ c
        f2 = f * f
        # d/dc sum W f^4 = 4 * sum W f^3 * B = 4 * B.T @ (W * f^3)
        return 4.0 * (B.T @ (W * f * f2))
    
    # Constraint: ||c||^2 - 1 = 0
    constraint = {'type': 'eq', 'fun': lambda c: np.sum(c*c) - 1.0, 'jac': lambda c: 2.0 * c}
    
    best_c = None
    best_val = np.inf
    
    # Multi-start optimization
    # Use SLSQP for robustness with equality constraint
    np.random.seed(42)
    
    # Generate diverse starting points
    # 1. Random
    # 2. Sparse (only a few non-zero)
    # 3. Symmetric patterns
    
    starts = []
    
    # Random starts
    for _ in range(50):
        c = np.random.randn(dim)
        c /= np.linalg.norm(c)
        starts.append(c)
        
    # Sparse starts: pick 1, 2, or 3 random indices
    for _ in range(30):
        c = np.zeros(dim)
        k = np.random.randint(1, 4)
        idxs = np.random.choice(dim, k, replace=False)
        c[idxs] = np.random.randn(k)
        c /= np.linalg.norm(c)
        starts.append(c)
        
    # Symmetric starts: e.g., only m=0, only m=10, etc.
    # m=0
    c = np.zeros(dim); c[10] = 1.0; starts.append(c)
    # m=10, m=-10
    c = np.zeros(dim); c[20] = 1.0/np.sqrt(2); c[0] = 1.0/np.sqrt(2); starts.append(c)
    # m=9, m=-9
    c = np.zeros(dim); c[19] = 1.0/np.sqrt(2); c[1] = 1.0/np.sqrt(2); starts.append(c)
    # All ones
    c = np.ones(dim); c /= np.linalg.norm(c); starts.append(c)
    
    for i, c0 in enumerate(starts):
        if time.time() > deadline - 5:
            break
            
        try:
            res = minimize(objective, c0, method='SLSQP', jac=gradient, 
                           constraints=[constraint], options={'maxiter': 1000, 'ftol': 1e-12})
            
            if res.success:
                c_opt = res.x
                # Normalize to ensure constraint is met exactly (SLSQP might be slightly off)
                c_opt /= np.linalg.norm(c_opt)
                val = objective(c_opt)
                
                if val < best_val:
                    best_val = val
                    best_c = c_opt.copy()
        except Exception:
            continue
    
    # If SLSQP failed or didn't improve, try Nelder-Mead on the sphere parameterization?
    # Or just refine the best candidate with L-BFGS-B on the unconstrained problem 
    # using a penalty method if needed, but SLSQP should be good.
    
    # Additional refinement: Local search around the best candidate
    if best_c is not None and time.time() < deadline - 10:
        # Try to polish with high precision
        # SLSQP is already high precision.
        # Let's try a few more random restarts focused on the best region?
        # Or just accept the best found.
        
        # One more check: Ensure validity
        max_abs = np.max(np.abs(best_c))
        if max_abs < 1e-3:
            best_c *= (1e-3 / max_abs)
        elif max_abs > 1e5:
            best_c *= (1e5 / max_abs)
            
        if not np.all(np.isfinite(best_c)):
            best_c = np.ones(dim) / np.sqrt(dim)
            
    if best_c is None:
        best_c = np.ones(dim)
        best_c /= np.linalg.norm(best_c)
        
    return best_c.astype(np.float64)
