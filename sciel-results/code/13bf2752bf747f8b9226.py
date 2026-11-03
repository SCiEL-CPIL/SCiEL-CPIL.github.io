import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    # Coefficients for a12 in terms of a0, a4, a8
    # g(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # H0(0) = 1, H4(0) = 3, H8(0) = 105, H12(0) = 10395
    c0 = 1.0 / 10395.0
    c4 = 3.0 / 10395.0
    c8 = 105.0 / 10395.0
    
    # Factorials
    f0 = 1.0       
    f4 = 24.0      
    f8 = 40320.0   
    f12 = 479001600.0 
    
    # Spatial moment coefficients (2n+1)n!
    x2_0 = 1.0
    x2_4 = 216.0
    x2_8 = 685440.0
    x2_12 = 11975040000.0
    
    # Frequency moment coefficients n*n!
    k_0 = 0.0
    k_4 = 96.0
    k_8 = 322560.0
    k_12 = 5748019200.0

    # Build M matrix (4x3) mapping u=[a0, a4, a8] to a=[a0, a4, a8, a12]
    # a12 = -c0*a0 - c4*a4 - c8*a8
    M = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-c0, -c4, -c8]
    ])
    
    # Diagonal matrices for norms
    F = np.diag([f0, f4, f8, f12])
    FX = np.diag([x2_0, x2_4, x2_8, x2_12])
    FK = np.diag([k_0, k_4, k_8, k_12])
    
    # Compute 3x3 matrices
    N_mat = M.T @ F @ M
    X_mat = M.T @ FX @ M
    K_mat = M.T @ FK @ M

    def objective(u):
        N = u @ N_mat @ u
        if N < 1e-15:
            return 1e15
        X = u @ X_mat @ u
        K = u @ K_mat @ u
        return (X * K) / (N * N)

    def grad(u):
        N = u @ N_mat @ u
        if N < 1e-15:
            return np.zeros(3)
        X = u @ X_mat @ u
        K = u @ K_mat @ u
        J = (X * K) / (N * N)
        
        dN = 2 * N_mat @ u
        dX = 2 * X_mat @ u
        dK = 2 * K_mat @ u
        
        # dJ = (K dX + X dK - 2 J dN) / N^2
        g = (K * dX + X * dK - 2 * J * dN) / (N * N)
        return g

    # Global optimization using Differential Evolution
    # Bounds [-1, 1] are sufficient due to scale invariance
    bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    try:
        res_de = differential_evolution(objective, bounds, maxiter=200, popsize=15, seed=42, tol=1e-12, polish=False)
        best_x = res_de.x
        best_val = res_de.fun
    except Exception:
        best_x = np.array([1.0, -1.0, 0.1])
        best_val = objective(best_x)

    # Local optimization using L-BFGS-B
    try:
        res_lbfgs = minimize(objective, best_x, jac=grad, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
        if res_lbfgs.success and res_lbfgs.fun < best_val:
            best_x = res_lbfgs.x
            best_val = res_lbfgs.fun
    except Exception:
        pass

    return best_x

# Test
if __name__ == "__main__":
    coeffs = construct()
    print("Coefficients:", coeffs)
