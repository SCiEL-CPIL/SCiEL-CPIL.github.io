import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 200.0  # Safety margin before 210s

def construct():
    """
    Finds coefficients c0, c4, c8 for the Hermite combination
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    such that g(0)=0 and the upper bound on A(g)*A(hat{g}) is minimized.
    Returns np.array([c0, c4, c8]).
    """
    
    # Indices of Hermite polynomials used
    K = [0, 4, 8, 12]
    n_vars = 3 # c0, c4, c8 are free, c12 is dependent
    
    # Pre-compute constants
    # H_n(0) values
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 945.0
    
    # Constraint: c0*1 + c4*3 + c8*105 + c12*945 = 0
    # c12 = -(c0 + 3*c4 + 105*c8) / 945
    # Jacobian of full coefficients c_full = [c0, c4, c8, c12] w.r.t u = [c0, c4, c8]
    J = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0/945.0, -3.0/945.0, -105.0/945.0]
    ])
    
    # Compute Norm Matrix C (diagonal)
    # C_ii = integral H_i^2 exp(-x^2) dx = sqrt(pi) * 2^i * i!
    C = np.zeros((4, 4))
    for idx, k in enumerate(K):
        # Use log to avoid overflow for large factorials, though 12! is small
        C[idx, idx] = np.sqrt(np.pi) * (2 ** k) * float(np.math.factorial(k))
        
    # Compute Position Second Moment Matrix A
    # A_ij = integral H_i(x) H_j(x) x^2 exp(-x^2) dx
    # Identity: x^2 H_j(x) = 0.25 H_{j+2}(x) + j H_j(x) + 0.5 j(j-1) H_{j-2}(x)
    A = np.zeros((4, 4))
    for i_idx, i in enumerate(K):
        for j_idx, j in enumerate(K):
            val = 0.0
            # Term 1: 0.25 * H_{j+2}. Non-zero if i == j+2
            if i == j + 2:
                # integral H_i H_{j+2} exp(-x^2) = sqrt(pi) 2^i i! (since i = j+2)
                # Wait, orthogonality: integral H_m H_n exp(-x^2) = sqrt(pi) 2^n n! delta_mn
                # Here m=i, n=j+2. So if i=j+2, integral = sqrt(pi) 2^i i!
                val += 0.25 * np.sqrt(np.pi) * (2 ** i) * float(np.math.factorial(i))
            
            # Term 2: j * H_j. Non-zero if i == j
            if i == j:
                val += j * np.sqrt(np.pi) * (2 ** i) * float(np.math.factorial(i))
            
            # Term 3: 0.5 j(j-1) * H_{j-2}. Non-zero if i == j-2
            if i == j - 2:
                # integral H_i H_{j-2} exp(-x^2). Here n = j-2 = i.
                val += 0.5 * j * (j - 1) * np.sqrt(np.pi) * (2 ** i) * float(np.math.factorial(i))
            
            A[i_idx, j_idx] = val
            
    # Since A and C are symmetric, and we established B=A for frequency variance
    # Objective is to minimize sigma_x^2 * sigma_xi^2 = (c^T A c / c^T C c)^2
    # Minimizing the square is equivalent to minimizing the base (since it's positive)
    # Let's minimize F(u) = (c^T A c) / (c^T C c)
    
    def objective_and_grad(u):
        # u = [c0, c4, c8]
        c_full = J @ u
        
        # Compute numerator N = c^T A c
        N = c_full @ (A @ c_full)
        
        # Compute denominator D = c^T C c
        D = c_full @ (C @ c_full)
        
        # Value
        val = N / D
        
        # Gradient w.r.t c_full
        # d/dc (N/D) = (D * dN/dc - N * dD/dc) / D^2
        # dN/dc = 2 A c
        # dD/dc = 2 C c
        grad_c = (D * (2 * A @ c_full) - N * (2 * C @ c_full)) / (D * D)
        
        # Chain rule: grad_u = J^T grad_c
        grad_u = J.T @ grad_c
        
        return val, grad_u

    # Initial guess: Seed values
    # Seed: [-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0]
    u0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Multi-start to ensure global minimum
    best_val = np.inf
    best_u = u0.copy()
    
    # Try seed
    val, grad = objective_and_grad(u0)
    if val < best_val:
        best_val = val
        best_u = u0.copy()
        
    # Generate random starts in a reasonable range
    # Coefficients might vary in magnitude.
    # c0 ~ O(1), c4 ~ O(1), c8 ~ O(0.1)
    np.random.seed(42)
    
    num_starts = 20
    
    for i in range(num_starts):
        if time.time() > TIME_LIMIT - 10:
            break
            
        # Log-uniform or uniform sampling around seed scale
        # Scale factors based on seed
        scale0 = 1.0
        scale4 = 0.04
        scale8 = 0.0003
        
        u_rand = np.array([
            np.random.uniform(-5*scale0, 5*scale0),
            np.random.uniform(-5*scale4, 5*scale4),
            np.random.uniform(-5*scale8, 5*scale8)
        ])
        
        try:
            res = minimize(objective_and_grad, u_rand, jac=True, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_u = res.x.copy()
        except Exception:
            pass
            
    # Final polish on the best found
    if time.time() < TIME_LIMIT - 5:
        try:
            res = minimize(objective_and_grad, best_u, jac=True, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_u = res.x.copy()
        except Exception:
            pass
            
    return best_u

