import numpy as np
import scipy.optimize

def integrate(f, t_span, y0):
    t0, t1 = t_span
    N = 40
    
    # Chebyshev-Lobatto points on [-1, 1]
    x = np.cos(np.pi * np.arange(N) / (N - 1))
    # Map to [t0, t1]
    t = 0.5 * (t1 - t0) * x + 0.5 * (t1 + t0)
    
    # Differentiation matrix
    c = np.ones(N)
    c[0] = 1.0
    c[-1] = 1.0
    c[1:-1] = 2.0
    
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (c[i] / c[j]) * (1.0 / (x[i] - x[j]))
            else:
                D[i, i] = -np.sum(D[i, :])
                
    # Scale for [t0, t1]
    D = D * (2.0 / (t1 - t0))
    
    def make_residual(y_flat):
        Y = np.zeros((N, 2))
        Y[0] = y0
        Y[1:] = y_flat.reshape((N-1, 2))
        
        dY = np.dot(D, Y)
        
        f_vals = np.zeros((N, 2))
        for i in range(N):
            f_vals[i] = f(t[i], Y[i])
            
        res = dY[1:] - f_vals[1:]
        return res.flatten()

    # Initial guess: constant y0
    y0_guess = np.tile(y0, (N-1, 1)).flatten()
    
    sol = scipy.optimize.root(make_residual, y0_guess, method='hybr')
    
    if not sol.success:
        sol = scipy.optimize.root(make_residual, y0_guess, method='lm')
        
    if not sol.success:
        return y0
        
    y_end = sol.y.reshape((N-1, 2))[-1]
    
    return y_end
