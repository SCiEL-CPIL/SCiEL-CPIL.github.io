import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator using Chebyshev Spectral Collocation.
    This method approximates the solution with a polynomial of degree N-1 on [t0, t1].
    It solves the resulting nonlinear algebraic system via Newton's method.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float).copy()
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0

    # Choose number of collocation points N
    # Van der Pol with mu=5 is smooth. N=30 to 50 should be sufficient for machine precision.
    # Let's start with N=40.
    N = 40
    
    # Chebyshev-Lobatto points on [t0, t1]
    # Standard construction: x_j = cos(j*pi/(N-1)) for j=0..N-1
    # x_0 = 1, x_{N-1} = -1
    # t_j = 0.5*(t1-t0)*x_j + 0.5*(t1+t0)
    # t_0 = t1, t_{N-1} = t0
    
    j = np.arange(N)
    x = np.cos(np.pi * j / (N - 1))
    t = 0.5 * (t1 - t0) * x + 0.5 * (t1 + t0)
    
    # Construct Chebyshev differentiation matrix D
    # D is N x N
    # Formula from Trefethen, "Spectral Methods in MATLAB"
    def chebdiff(N):
        x = np.cos(np.pi * np.arange(N) / (N - 1))
        c = np.r_[2, np.ones(N - 2), 2] * np.power(-1, np.arange(N))
        X = np.tile(x, (N, 1))
        dX = X - X.T
        D = (np.outer(c, 1.0 / c) / (dX + np.eye(N)))
        D = D - np.diag(D.sum(axis=1))
        return D, x

    D, x_cheb = chebdiff(N)
    # Note: chebdiff returns x in standard order (1 to -1).
    # t calculated above matches this order.
    # t[0] = t1, t[N-1] = t0.
    
    # We want y(t0) = y0. So y[N-1] = y0.
    # Unknowns: y[0], ..., y[N-2]. Let u be the vector of unknowns.
    # Total unknowns: 2*(N-1).
    # Let's flatten u.
    # u_flat[2*i] = y_i[0], u_flat[2*i+1] = y_i[1] for i=0..N-2.
    
    # Residual function R(u_flat) = 0
    # For i = 0..N-2:
    # (D @ y)[i] - f(t[i], y[i]) = 0
    # where y[i] = [u_flat[2*i], u_flat[2*i+1]] for i < N-1
    # and y[N-1] = y0.
    
    # Jacobian J is (2*(N-1)) x (2*(N-1))
    # J = dR/du_flat
    # R_i = D[i, 0:N-1] @ u_i_vector + D[i, N-1]*y0 - f(t_i, u_i_vector)
    # where u_i_vector is the 2-vector corresponding to u_flat[2*i:2*i+2].
    # Note: D acts on the full y vector.
    # y = [u_0, u_1, ..., u_{N-2}, y0] (stacked 2-vectors)
    # (D @ y)_k = sum_j D[k,j] y_j
    # So (D @ y)_k depends on all y_j.
    # Specifically, for the k-th component (k=0..N-2), the derivative part is:
    # sum_{j=0}^{N-2} D[k,j] y_j + D[k, N-1] y0
    # y_j for j<N-1 are unknowns. y0 is known.
    
    # Let's construct J block by block.
    # J is block matrix of size (N-1)x(N-1) with 2x2 blocks.
    # Block (k, m) for k, m in 0..N-2:
    # d R_k / d u_m
    # R_k = (D @ y)_k - f(t_k, y_k)
    # (D @ y)_k = sum_{j=0}^{N-1} D[k,j] y_j
    # = sum_{j=0}^{N-2} D[k,j] y_j + D[k, N-1] y0
    
    # d/d u_m (sum_{j=0}^{N-2} D[k,j] y_j)
    # If m == k: D[k, k] * I_2 (since y_k = u_k)
    # If m != k: D[k, m] * I_2 (since y_m = u_m)
    # So the derivative part of the block (k, m) is D[k, m] * I_2.
    
    # d/d u_m (- f(t_k, y_k))
    # If m == k: - df/dy (t_k, y_k)
    # If m != k: 0
    
    # So:
    # Block (k, m) = D[k, m] * I_2 - (df/dy if m==k else 0)
    
    # We can write J = kron(I_{N-1}, D[:, 0:N-1]) - diag(df/dy blocks)?
    # Wait, D[:, 0:N-1] is (N x (N-1)).
    # The derivative part for row k is D[k, 0:N-1] @ [u_0, ..., u_{N-2}]
    # This is a linear map from u_flat to R_k.
    # The matrix for the linear part is:
    # Let L be the matrix such that L @ u_flat = (D @ y)_k (for k<N-1, ignoring constant term).
    # L is 2*(N-1) x 2*(N-1).
    # Row 2*k and 2*k+1 of L correspond to component 0 and 1 of y'_k.
    # y'_k[0] = sum_j D[k,j] y_j[0]
    # y'_k[1] = sum_j D[k,j] y_j[1]
    # So L[2*k, 2*m] = D[k, m]
    #    L[2*k+1, 2*m+1] = D[k, m]
    #    L[2*k, 2*m+1] = 0
    #    L[2*k+1, 2*m] = 0
    # Basically L = kron(I_2, D[:, 0:N-1])? No.
    # kron(I_2, M) where M is (N-1)x(N-1) gives blocks M_ij * I_2.
    # Here we want M_km in position (2k, 2m) and (2k+1, 2m+1).
    # This is exactly kron(I_{N-1}, D[:, 0:N-1])? No.
    # Let's check dimensions.
    # D[:, 0:N-1] is N x (N-1). We only need rows 0..N-2.
    # D_sub = D[0:N-1, 0:N-1]
    # The linear part matrix L has:
    # L[2*k + a, 2*m + b] = D_sub[k, m] * delta_{a,b} for a,b in {0,1}.
    # This is block diagonal structure if we permute? No.
    # It's a block matrix where each block (k,m) is D_sub[k,m] * I_2.
    # So L = kron(D_sub, I_2).
    
    # The nonlinear part is - diag(df/dy blocks).
    # Let F be the block diagonal matrix with blocks -df/dy(t_k, y_k) for k=0..N-2.
    # J = L + F.
    
    # Initial guess for u
    # Use linear interpolation or constant.
    # Constant is simple.
    u_flat = np.tile(y0, (N - 1, 1)).flatten()
    
    max_iter = 50
    tol = 1e-12
    
    # Precompute constant term from y0
    # const_k = D[k, N-1] * y0
    # This is added to the linear part.
    # R_k = (L @ u_flat)_k + const_k - f(t_k, y_k)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Evaluate residual
        # Build y from u_flat
        y = np.empty((N, 2))
        y[0:N-1] = u_flat.reshape(N-1, 2)
        y[N-1] = y0
        
        # Compute f
        fy = np.empty((N, 2))
        for i in range(N):
            fy[i] = f(t[i], y[i])
            
        # Compute derivative part
        dy = D @ y
        
        # Residual for first N-1 points
        R_vec = (dy[0:N-1] - fy[0:N-1]).flatten()
        
        # Check convergence
        if np.linalg.norm(R_vec) < tol:
            break
            
        # Compute Jacobian
        # L = kron(D_sub, I_2)
        D_sub = D[0:N-1, 0:N-1]
        L = np.kron(D_sub, np.eye(2))
        
        # F = block diagonal of -df/dy
        F = np.zeros((2*(N-1), 2*(N-1)))
        for k in range(N-1):
            yk = y[k]
            # f(t, y) = [y1, 5*(1-y0^2)*y1 - y0]
            # df/dy = [[0, 1], [-10*y0*y1, 5*(1-y0^2)]]
            y0_val = yk[0]
            y1_val = yk[1]
            df_dy = np.array([
                [0.0, 1.0],
                [-10.0 * y0_val * y1_val, 5.0 * (1.0 - y0_val**2)]
            ])
            F[2*k:2*k+2, 2*k:2*k+2] = -df_dy
            
        J = L + F
        
        # Solve J @ delta_u = -R_vec
        try:
            delta_u = np.linalg.solve(J, -R_vec)
        except np.linalg.LinAlgError:
            # If singular, use least squares or break
            break
            
        # Update u
        u_flat = u_flat + delta_u
        
    # Extract final y
    y_end = y[N-1] # Wait, y[N-1] is y0?
    # No, y[N-1] is fixed to y0 in the residual setup.
    # The solution at t0 is y0.
    # The solution at t1 is y[0].
    # We want y(t1).
    # t[0] = t1.
    # So we want y[0].
    
    # Reconstruct y[0] from u_flat
    y_end = u_flat[0:2]
    
    return y_end
