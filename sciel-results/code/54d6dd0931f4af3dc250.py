import numpy as np

def construct():
    # Indices of the Hermite polynomials used
    indices = [0, 4, 8, 12]
    
    # Matrix D with diagonal elements 2n + 1
    D = np.diag([2 * n + 1 for n in indices])
    
    # Compute h_n = psi_n(0) for the given indices
    # psi_n(x) = (2^n n! sqrt(pi))^(-1/2) H_n(x) exp(-x^2/2)
    # psi_n(0) = (2^n n! sqrt(pi))^(-1/2) H_n(0)
    # H_n(0) for even n: (-1)^(n/2) * (2n)! / (2^(n/2) n!)
    # So h_n = (-1)^(n/2) * (2n)! / (2^(n/2) n! * (2^n n! sqrt(pi))^(1/2))
    # h_n = (-1)^(n/2) * (2n)! / (2^n n! sqrt(pi) * 2^(n/2) sqrt(n!)) ... wait
    # Let's just use scipy or manual calculation for precision
    
    def hermite_h(n, x):
        # Use scipy.special.hermite for stability? Or explicit formula.
        # For small n, explicit is fine.
        if n == 0: return 1
        if n == 1: return 2*x
        # Recurrence: H_{n+1} = 2x H_n - 2n H_{n-1}
        H_prev = 1
        H_curr = 2*x
        for k in range(1, n):
            H_next = 2*x*H_curr - 2*k*H_prev
            H_prev = H_curr
            H_curr = H_next
        return H_curr

    h = np.zeros(len(indices))
    for i, n in enumerate(indices):
        if n % 2 != 0:
            h[i] = 0
        else:
            # H_n(0) = (-1)^(n/2) * (2n)! / (2^(n/2) n!)
            from math import factorial
            H0 = ((-1)**(n//2)) * factorial(2*n) / (2**(n//2) * factorial(n))
            norm_factor = (2**n * factorial(n) * np.sqrt(np.pi))**0.5
            h[i] = H0 / norm_factor

    # We want to minimize a^T D a / a^T a subject to h^T a = 0
    # This is equivalent to finding the smallest eigenvalue of the projection of D onto the null space of h
    
    # Get an orthonormal basis for the null space of h^T
    # Use SVD of h (1x4)
    U, s, Vt = np.linalg.svd(h.reshape(1, -1))
    # The last 3 rows of Vt are the basis for the null space
    basis = Vt[1:, :].T  # 4x3 matrix
    
    # Project D onto this basis
    M = basis.T @ D @ basis  # 3x3 matrix
    
    # Find eigenvalues and eigenvectors of M
    eigvals, eigvecs = np.linalg.eigh(M)
    
    # The smallest eigenvalue corresponds to the minimum Rayleigh quotient
    # The optimal a in the original space is basis @ eigvec_min
    idx_min = np.argmin(eigvals)
    a_opt = basis @ eigvecs[:, idx_min]
    
    # Convert a_opt to c coefficients
    # a_n = c_n * (2^n n! sqrt(pi))^0.5
    # c_n = a_n / (2^n n! sqrt(pi))^0.5
    c_coeffs = np.zeros(len(indices))
    for i, n in enumerate(indices):
        norm_factor = (2**n * factorial(n) * np.sqrt(np.pi))**0.5
        c_coeffs[i] = a_opt[i] / norm_factor
        
    # The function returns c_0, c_4, c_8
    # The scale is arbitrary, but we can normalize e.g. c_0 to -1 or keep as is.
    # The seed had c_0 = -1.13. The relative ratios are what matters.
    # Let's return the first three coefficients.
    return c_coeffs[:3]
