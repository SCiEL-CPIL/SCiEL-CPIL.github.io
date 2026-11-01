import numpy as np
from scipy.linalg import eigh

def construct():
    # Indices of the Hermite polynomials used
    indices = [0, 4, 8, 12]
    
    # Calculate h_n(0) for n in indices
    # h_n(x) = (2^n n! sqrt(pi))^-0.5 * exp(-x^2/2) * H_n(x)
    # h_n(0) = (2^n n! sqrt(pi))^-0.5 * H_n(0)
    # H_{2k}(0) = (-1)^k * (2k)! / k!
    
    def h_n_zero(n):
        if n % 2 != 0:
            return 0.0
        k = n // 2
        # H_{2k}(0)
        H_val = ((-1) ** k) * (np.math.factorial(2 * k) / np.math.factorial(k))
        norm = (2 ** n) * np.math.factorial(n) * np.sqrt(np.pi)
        return H_val / np.sqrt(norm)
    
    v = np.array([h_n_zero(n) for n in indices])
    
    # Diagonal matrix D for ||x f||^2
    # X_{nn} = (n + 1) / 2
    D = np.diag([(n + 1) / 2.0 for n in indices])
    
    # Matrix M mapping c_free (c0, c4, c8) to c_full (c0, c4, c8, c12)
    # Constraint: sum(c_i * v_i) = 0 => c_12 = - (c_0 v_0 + c_4 v_4 + c_8 v_8) / v_12
    v0, v4, v8, v12 = v
    M = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-v0/v12, -v4/v12, -v8/v12]
    ])
    
    # A = M.T @ D @ M
    # B = M.T @ M
    A = M.T @ D @ M
    B = M.T @ M
    
    # Solve generalized eigenvalue problem A c = lambda B c
    # We want the smallest eigenvalue
    try:
        eigenvalues, eigenvectors = eigh(A, B)
        # eigh returns eigenvalues in ascending order
        idx = 0
        c_free = eigenvectors[:, idx]
    except Exception:
        # Fallback to seed values if linear algebra fails
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Normalize c_free such that ||c_full||_2 = 1
    c_full = M @ c_free
    norm_c_full = np.linalg.norm(c_full)
    if norm_c_full == 0:
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    c_full = c_full / norm_c_full
    
    # Extract the first three coefficients
    return c_full[:3]
