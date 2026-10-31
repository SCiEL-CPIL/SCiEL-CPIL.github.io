import numpy as np
from scipy.linalg import eigh

def construct():
    # Gauss-Hermite quadrature
    n_quad = 200
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(n_quad)
    
    # Hermite polynomials (physicists') via recurrence
    def H(n, x):
        if n == 0:
            return np.ones_like(x)
        if n == 1:
            return 2 * x
        H_nm1 = np.ones_like(x)
        H_n = 2 * x
        for k in range(1, n):
            H_np1 = 2 * x * H_n - 2 * k * H_nm1
            H_nm1, H_n = H_n, H_np1
        return H_n
    
    H0 = H(0, x_nodes)
    H4 = H(4, x_nodes)
    H8 = H(8, x_nodes)
    H12 = H(12, x_nodes)
    
    # Basis functions with g(0)=0 constraint baked in
    # phi_i = H_i - (H_i(0)/H12(0)) * H12
    H12_0 = H12[0] if False else 665280  # H12(0)
    # Actually compute H12(0) properly
    # H12(0) from the polynomial: 665280
    # Let's compute via the values at x=0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    phi0 = H0 - (H0_0 / H12_0) * H12
    phi4 = H4 - (H4_0 / H12_0) * H12
    phi8 = H8 - (H8_0 / H12_0) * H12
    
    # Mass matrix M and stiffness matrix K
    # M_ij = int phi_i phi_j e^{-x^2} dx ~ sum w phi_i phi_j
    # K_ij = int x^2 phi_i phi_j e^{-x^2} dx ~ sum w x^2 phi_i phi_j
    phis = np.stack([phi0, phi4, phi8])  # shape (3, n_quad)
    
    M = phis * (phis * w_nodes)[None, :]  # M_ij = sum_k phi_i(k) phi_j(k) w_k
    # Actually: M[i,j] = sum_k phi[i,k] * phi[j,k] * w[k]
    M = phis @ (phis * w_nodes).T
    K = phis @ (phis * (w_nodes * x_nodes**2)).T
    
    # Solve generalized eigenvalue problem K v = lambda M v
    # Use scipy.linalg.eigh(K, M)
    eigvals, eigvecs = eigh(K, M)
    
    # Smallest eigenvalue
    idx = np.argmin(eigvals)
    a_opt = eigvecs[:, idx]
    
    # a_opt = [a0, a4, a8]
    # Scale doesn't matter, but let's return as is
    return np.array([a_opt[0], a_opt[1], a_opt[2]])
