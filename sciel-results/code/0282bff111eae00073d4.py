import numpy as np

def construct():
    # Hermite polynomial values at 0
    h = np.array([1.0, -3.0, 105.0, -10395.0])
    
    # Moments n + 1/2
    moments = np.array([0.5, 4.5, 8.5, 12.5])
    M = np.diag(moments)
    
    # Constraint matrix A such that A @ c = 0
    # g(0) = sum(c_i * h_i) = 0
    A = h.reshape(1, -1)
    
    # Find an orthonormal basis for the null space of A
    # Using SVD
    U, S, Vt = np.linalg.svd(A)
    # The last n-k rows of Vt form a basis for the null space
    # Here k=1, n=4, so the last 3 rows of Vt
    Q = Vt[1:].T
    
    # Project M onto the null space
    B = Q.T @ M @ Q
    
    # Find the smallest eigenvalue and eigenvector of B
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # The smallest eigenvalue corresponds to the minimum of the Rayleigh quotient
    idx_min = np.argmin(eigenvalues)
    v_min = eigenvectors[:, idx_min]
    
    # Reconstruct the optimal coefficient vector in the original space
    c_opt = Q @ v_min
    
    # The objective function is scale-invariant, so we can return c_opt directly
    # However, we might want to normalize it or just return the values.
    # The prompt asks for "three coefficients".
    # Returning the raw eigenvector components is fine.
    
    return np.array([c_opt[0], c_opt[1], c_opt[2]])
