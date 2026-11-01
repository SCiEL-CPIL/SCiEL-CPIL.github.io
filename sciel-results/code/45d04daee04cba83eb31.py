import numpy as np
from scipy.special import hermite
from scipy.integrate import quad
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 180.0 # Safety margin for 210s limit
    
    # Basis indices
    K = [0, 4, 8, 12]
    n_basis = len(K)
    
    # Precompute Hermite polynomials
    # H_n(x)
    H = [hermite(k) for k in K]
    
    def phi(k_idx, xi):
        k = K[k_idx]
        # H_k(sqrt(pi) * xi)
        val = H[k_idx](np.sqrt(np.pi) * xi)
        return val * np.exp(-np.pi * xi**2)
    
    def dphi(k_idx, xi):
        k = K[k_idx]
        # Derivative of H_k(sqrt(pi) xi) e^{-pi xi^2}
        # d/dxi H_k(sqrt(pi) xi) = sqrt(pi) H'_k(sqrt(pi) xi) = sqrt(pi) 2k H_{k-1}(sqrt(pi) xi)
        # d/dxi e^{-pi xi^2} = -2 pi xi e^{-pi xi^2}
        if k == 0:
            return -2 * np.pi * xi * np.exp(-np.pi * xi**2)
        H_k_minus_1 = hermite(k-1)
        term1 = np.sqrt(np.pi) * 2 * k * H_k_minus_1(np.sqrt(np.pi) * xi) * np.exp(-np.pi * xi**2)
        term2 = -2 * np.pi * xi * H[k_idx](np.sqrt(np.pi) * xi) * np.exp(-np.pi * xi**2)
        return term1 + term2

    # Compute matrices D, Q, M
    # D_ii = int phi_i^2
    # Q_ij = int xi^2 phi_i phi_j
    # M_ij = int dphi_i dphi_j
    
    D = np.zeros((n_basis, n_basis))
    Q = np.zeros((n_basis, n_basis))
    M = np.zeros((n_basis, n_basis))
    
    # Integration limits
    L = 10.0
    
    for i in range(n_basis):
        if time.time() > deadline - 30: break
        for j in range(i, n_basis):
            # D
            f_D = lambda x: phi(i, x) * phi(j, x)
            val_D, _ = quad(f_D, -L, L)
            D[i, j] = val_D
            D[j, i] = val_D
            
            # Q
            f_Q = lambda x: x**2 * phi(i, x) * phi(j, x)
            val_Q, _ = quad(f_Q, -L, L)
            Q[i, j] = val_Q
            Q[j, i] = val_Q
            
            # M
            f_M = lambda x: dphi(i, x) * dphi(j, x)
            val_M, _ = quad(f_M, -L, L)
            M[i, j] = val_M
            M[j, i] = val_M

    # Constraint A c = 0
    # H_0(0)=1, H_4(0)=-12, H_8(0)=1680, H_12(0)=133056
    A = np.array([1.0, -12.0, 1680.0, 133056.0])
    
    # Optimization
    # Minimize (c^T M c) * (c^T Q c) subject to c^T D c = 1 and A c = 0
    
    # Initial guess
    c0 = np.array([1.0, 0.0, 0.0, 0.0])
    # Normalize c0
    norm0 = np.sqrt(c0 @ D @ c0)
    c0 = c0 / norm0
    
    def objective(c):
        val_M = c @ M @ c
        val_Q = c @ Q @ c
        return val_M * val_Q
        
    def constraint_norm(c):
        return c @ D @ c - 1.0
        
    def constraint_A(c):
        return A @ c
        
    constraints = [
        {'type': 'eq', 'fun': constraint_norm},
        {'type': 'eq', 'fun': constraint_A}
    ]
    
    # Use SLSQP
    try:
        res = minimize(objective, c0, method='SLSQP', constraints=constraints, 
                       options={'maxiter': 1000, 'ftol': 1e-12})
        
        if res.success:
            c_opt = res.x
            # Return first 3 coefficients
            return np.array([c_opt[0], c_opt[1], c_opt[2]])
        else:
            # Fallback to seed or best found
            pass
    except Exception as e:
        pass
        
    # Fallback to seed values if optimization fails
    return np.array([-113 / 100, 1 / 25, 1 / 3240])
