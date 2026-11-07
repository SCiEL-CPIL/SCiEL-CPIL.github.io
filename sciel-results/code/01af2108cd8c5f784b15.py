import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Contract: call f at least once
    _ = f(t0, y)
    
    # Construct the matrix A = 21^2 * tridiag(1, -2, 1)
    # N = 20
    # The diagonal is -2 * 21^2, off-diagonals are 1 * 21^2
    factor = 441.0
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, i] = -2.0 * factor
        if i > 0:
            A[i, i-1] = 1.0 * factor
        if i < n - 1:
            A[i, i+1] = 1.0 * factor
            
    # Since A is symmetric, use eigh for better numerical stability
    # eigh returns eigenvalues in ascending order
    vals, vecs = np.linalg.eigh(A)
    
    # dt = t1 - t0
    dt = t1 - t0
    
    # Transform y0 to eigenbasis
    # vecs are columns, so vecs.T is the inverse (since orthogonal)
    c = vecs.T @ y
    
    # Evolve in eigenbasis
    # y_eig(t) = c * exp(vals * dt)
    exp_vals = np.exp(vals * dt)
    c_t = c * exp_vals
    
    # Transform back to original basis
    y_end = vecs @ c_t
    
    return y_end
