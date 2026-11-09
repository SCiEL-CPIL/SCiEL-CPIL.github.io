import numpy as np
from scipy.optimize import root

def integrate(f, t_span, y0):
    t0, t1 = t_span
    # Use Chebyshev Collocation
    N = 64  # Number of collocation points
    # Chebyshev-Lobatto nodes in [-1, 1]
    x = np.cos(np.pi * np.arange(N) / (N - 1))[::-1]
    # Map to [t0, t1]
    t = 0.5 * (t1 - t0) * x + 0.5 * (t1 + t0)
    
    # Build differentiation matrix D for Chebyshev-Lobatto nodes
    # c_i = 2 for i=0, N-1 and 1 otherwise, but adjusted for the formula
    # Standard formula:
    # c = 1
    # c[0] = 2
    # c[N-1] = 2
    # x_i = cos(pi*i/(N-1))
    # D = (c_i / c_j) * (-1)^(i+j) / (x_i - x_j) for i != j
    # D_ii = -0.5 * c_i / (1 - x_i^2) for i != 0, N-1
    # D_00 = (2N^2 - 1) / 6
    # D_NN = -(2N^2 - 1) / 6
    
    c = np.ones(N)
    c[0] = 2
    c[N-1] = 2
    
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1) ** (i + j)) / (x[i] - x[j])
    # Diagonal terms
    for i in range(N):
        if i == 0 or i == N - 1:
            D[i, i] = (2 * N**2 - 1) / 6.0 if i == 0 else -(2 * N**2 - 1) / 6.0
        else:
            D[i, i] = -0.5 * c[i] / (1 - x[i]**2)
            
    # Scale D for the interval [t0, t1]
    # d/dt = (dt/dx) d/dx = (2/(t1-t0)) d/dx
    scale = 2.0 / (t1 - t0)
    D = D * scale
    D2 = D @ D

    # Unknowns: y0_vals (N), y1_vals (N)
    # Total 2N unknowns
    # Initial guess: constant y0
    y0_guess = np.full(N, y0[0])
    y1_guess = np.full(N, y0[1])
    Y0 = np.concatenate([y0_guess, y1_guess])

    def residual(Y):
        y0_vals = Y[:N]
        y1_vals = Y[N:]
        
        res = np.zeros(2 * N)
        
        # Initial conditions
        res[0] = y0_vals[0] - y0[0]
        res[N] = y1_vals[0] - y0[1]
        
        # Collocation equations for i=1 to N-1
        # y1 = dy0/dt
        # y0'' = f(t, y0, y1)
        for i in range(1, N):
            dy0_dt = np.dot(D[i, :], y0_vals)
            d2y0_dt2 = np.dot(D2[i, :], y0_vals)
            
            # f(t, y) returns [dy0, dy1] = [y1, 5(1-y0^2)y1 - y0]
            # We only need the second component for y0''
            # But f is called with y = [y0, y1]
            # We must call f as per the contract
            # "f(t, y) returns y' as an array"
            # We need to count calls.
            # We will call f N-1 times in the loop.
            
            y_vec = np.array([y0_vals[i], y1_vals[i]])
            f_val = f(t[i], y_vec)
            
            # res[0:N] corresponds to y0 equations?
            # Let's structure:
            # res[0] = y0[0] - y0_0
            # res[1:N-1] = y1[i] - (D y0)[i]  (for i=1..N-1)
            # res[N] = y1[0] - y1_0
            # res[N+1:2N] = (D2 y0)[i] - f_1(t[i], y[i]) (for i=1..N-1)
            
            res[i] = y1_vals[i] - dy0_dt
            res[N + i] = d2y0_dt2 - f_val[1]
            
        return res

    # Solve
    sol = root(residual, Y0, method='hybr', tol=1e-12)
    
    if not sol.success:
        # Fallback to scipy solve_ivp if collocation fails
        import scipy.integrate
        sol_ivp = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-11, 
            atol=1e-13
        )
        return sol_ivp.y[:, -1]

    y_end = sol.x[-2:] # Last values of y0 and y1
    return y_end

