"""Optimized program for the uncertainty inequality task using robust global and local optimization."""
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Return three coefficients (a0, a4, a8) for the Hermite combination.
    The degree-12 weight a12 is derived to impose g(0) = 0.
    We minimize the product of the spatial variance and the frequency variance.
    """
    # Constants for the constraint g(0) = 0
    # a0 + 3a4 + 105a8 + 10395a12 = 0
    c0 = 1.0 / 10395.0
    c4 = 3.0 / 10395.0
    c8 = 105.0 / 10395.0
    
    # Diagonal terms for Norm, X^2, and K (Frequency)
    # Norm: n!
    f0, f4, f8, f12 = 1.0, 24.0, 40320.0, 479001600.0
    # X^2: (2n+1)n!
    x2_0, x2_4, x2_8, x2_12 = 1.0, 216.0, 685440.0, 11975040000.0
    # K: n*n!
    k_0, k_4, k_8, k_12 = 0.0, 96.0, 322560.0, 5748019200.0

    # Construct the mapping from a = [a0, a4, a8] to v = [v0, v4, v8, v12]
    # v0 = a0
    # v4 = a4
    # v8 = a8
    # v12 = -c0*a0 - c4*a4 - c8*a8
    # This is a linear map v = M @ a
    M = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-c0, -c4, -c8]
    ])
    
    # Diagonal matrices for the full basis
    D_N = np.diag([f0, f4, f8, f12])
    D_X = np.diag([x2_0, x2_4, x2_8, x2_12])
    D_K = np.diag([k_0, k_4, k_8, k_12])
    
    # Projected matrices in the 3D space a
    # A = M^T D_N M
    # B = M^T D_X M
    # C = M^T D_K M
    A = M.T @ D_N @ M
    B = M.T @ D_X @ M
    C = M.T @ D_K @ M
    
    def objective(a):
        N = a @ A @ a
        if N < 1e-15:
            return 1e15
        X = a @ B @ a
        K = a @ C @ a
        return (X * K) / (N * N)
        
    def gradient(a):
        N = a @ A @ a
        X = a @ B @ a
        K = a @ C @ a
        val = (X * K) / (N * N)
        
        # d/da (X K / N^2)
        # = (K dX + X dK) / N^2 - 2 X K dN / N^3
        # dX = 2 B a, dK = 2 C a, dN = 2 A a
        dX = 2 * B @ a
        dK = 2 * C @ a
        dN = 2 * A @ a
        
        grad = (K * dX + X * dK) / (N * N) - 2 * val * dN / (N * N)
        return grad

    # Multi-start optimization with a wider variety of starts
    best_val = float('inf')
    best_x = None
    
    # Generate starts: axes, and random points on the sphere
    starts = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, 1.0]),
        np.array([1.0, -1.0, 1.0]),
        np.array([1.0, 1.0, -1.0]),
    ]
    
    # Add some random starts
    np.random.seed(42)
    for _ in range(20):
        v = np.random.randn(3)
        v /= np.linalg.norm(v)
        starts.append(v)
        
    for start in starts:
        try:
            res = minimize(objective, start, jac=gradient, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    if best_x is None:
        best_x = np.array([0.0, 0.0, 0.0])
        
    return best_x

# Test
if __name__ == "__main__":
    coeffs = construct()
    print("Coefficients:", coeffs)
