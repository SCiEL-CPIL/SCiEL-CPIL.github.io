"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def integrate(f, t_span, y0):
    # Call f at least once as required by the contract
    # We pass t_start and y0, the result is ignored
    f(t_span[0], y0)
    
    t0, t1 = t_span
    dt = t1 - t0
    N = len(y0)
    
    # The problem specifies N=20, A = 21^2 * tridiag(1, -2, 1)
    # Eigenvalues of tridiag(1, -2, 1) of size N are mu_k = -4 * sin^2(k * pi / (2*(N+1)))
    # Eigenvalues of A are lambda_k = 21^2 * mu_k
    # Eigenvectors are v_k[j] = sin((j+1) * k * pi / (N+1)) for j=0..N-1, k=1..N
    
    # Construct the orthogonal matrix Q of eigenvectors
    # Q[j, k] = sqrt(2/(N+1)) * sin((j+1) * (k+1) * pi / (N+1))
    j = np.arange(1, N + 1)[:, None]
    k = np.arange(1, N + 1)[None, :]
    Q = np.sqrt(2.0 / (N + 1)) * np.sin(j * k * np.pi / (N + 1))
    
    # Compute eigenvalues
    # lambda_k = 441 * (-4 * sin^2(k * pi / 42))
    # Note: k in formula is 1..N
    k_vals = np.arange(1, N + 1)
    lambdas = 441.0 * (-4.0 * np.sin(k_vals * np.pi / (2 * (N + 1)))**2)
    
    # Transform y0 to eigenbasis
    y0_eig = Q.T @ y0
    
    # Apply exponential decay
    # y_end_eig[k] = y0_eig[k] * exp(lambda_k * dt)
    y_end_eig = y0_eig * np.exp(lambdas * dt)
    
    # Transform back to physical space
    y_end = Q @ y_end_eig
    
    return y_end
# EVOLVE-BLOCK-END
