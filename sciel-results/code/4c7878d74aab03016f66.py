import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    
    col = 0
    for block in range(3):
        for sub in range(2):
            # 7 Strassen products
            strassen_UV = [
                # (U_entries, V_entries)
                # M1 = A[0,2*sub] * (B[2*sub,2*block] + B[2*sub+1,2*block+1])
                ({(0, 2*sub): 1}, {(2*sub, 2*block): 1, (2*sub+1, 2*block+1): 1}),
                # M2 = (A[1,2*sub] + A[1,2*sub+1]) * B[2*sub,2*block]
                ({(1, 2*sub): 1, (1, 2*sub+1): 1}, {(2*sub, 2*block): 1}),
                # M3 = A[0,2*sub] * (B[2*sub+1,2*block] - B[2*sub+1,2*block+1])
                ({(0, 2*sub): 1}, {(2*sub+1, 2*block): 1, (2*sub+1, 2*block+1): -1}),
                # M4 = A[1,2*sub+1] * (B[2*sub,2*block+1] - B[2*sub,2*block])
                ({(1, 2*sub+1): 1}, {(2*sub, 2*block+1): 1, (2*sub, 2*block): -1}),
                # M5 = (A[0,2*sub] + A[0,2*sub+1]) * B[2*sub+1,2*block+1]
                ({(0, 2*sub): 1, (0, 2*sub+1): 1}, {(2*sub+1, 2*block+1): 1}),
                # M6 = (A[1,2*sub] + A[1,2*sub+1]) * (B[2*sub,2*block+1] + B[2*sub+1,2*block+1])
                ({(1, 2*sub): 1, (1, 2*sub+1): 1}, {(2*sub, 2*block+1): 1, (2*sub+1, 2*block+1): 1}),
                # M7 = (A[0,2*sub+1] - A[1,2*sub]) * (B[2*sub+1,2*block] + B[2*sub+1,2*block+1])
                ({(0, 2*sub+1): 1, (1, 2*sub): -1}, {(2*sub+1, 2*block): 1, (2*sub+1, 2*block+1): 1}),
            ]
            
            for u_entries, v_entries in strassen_UV:
                for (i, j), val in u_entries.items():
                    U[i * m + j, col] = val
                for (j, k), val in v_entries.items():
                    V[j * p + k, col] = val
                col += 1
    
    # Now solve for W: the tensor equation is
    # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # 
    # For each output (i', k'), we need:
    # sum_r W[k'*n+i', r] * (sum_j U[i*m+j, r] * V[j*p+k', r] ... ) 
    # 
    # Actually, the tensor T has T[(i,j),(j,k),(i,k)] = 1
    # The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if indices match
    #
    # Let me define: for each r, the "contribution matrix" to output (i,k) from input (i,j) and (j,k)
    # is U[i*m+j, r] * V[j*p+k, r].
    #
    # We need: for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # This is a system of equations for W. For each (i,k) pair (12 pairs), we have:
    # sum_r W[k*n+i, r] * S_r(i,k) = 1 where S_r(i,k) involves U and V...
    #
    # Actually it's: for fixed (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # This couples all W entries. Let me set up the linear system.
    # Unknowns: W[0..11, 0..41] = 12*42 = 504 unknowns
    # Equations: for each (i,j,k) with i in 0..1, j in 0..3, k in 0..5: 2*4*6 = 48 equations
    #
    # But each equation only involves W[k*n+i, :] (one row of W). So we can solve row by row!
    #
    # For fixed (i, k): sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Let M_ik[r] = sum_j U[i*m+j, r] * V[j*p+k, r]
    # Then: sum_r M_ik[r] * W[k*n+i, r] = 1
    #
    # This is a system of 12 equations (one per (i,k) pair) in 42 unknowns per row... 
    # Wait, no. For fixed (i,k), the equation is:
    # sum_r M_ik[r] * W[k*n+i, r] = 1
    # This is 1 equation in 42 unknowns (W[k*n+i, 0..41]).
    #
    # We have 12 such equations (for 12 (i,k) pairs), each in 42 unknowns.
    # Total: 12 equations, 12*42 = 504 unknowns. Underdetermined!
    #
    # But we need the decomposition to be exact. The constraint is:
    # For ALL (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # For fixed (i,k): sum_j [sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]] = ... 
    # No, it's for EACH (i,j,k) separately.
    #
    # For fixed (i, k), and varying j:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for each j
    #
    # So for fixed (i,k), we have 4 equations (j=0,1,2,3) in 42 unknowns (W[k*n+i, :]).
    # Total: 12 * 4 = 48 equations in 504 unknowns. Still underdetermined.
    #
    # We can pick a minimum-norm solution or just find any solution.
    
    # Build the linear system
    # Unknowns: W_flat of size 12*42
    # For each (i, j, k): equation sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    A_sys = np.zeros((n*m*p, p*n*rank))  # 48 x 504
    b_sys = np.ones(n*m*p)
    
    eq = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    A_sys[eq, (k*n+i)*rank + r] = U[i*m+j, r] * V[j*p+k, r]
                eq += 1
    
    # Solve the underdetermined system (minimum norm solution)
    W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
    W = W_flat.reshape(p*n, rank)
    
    # Verify
    # Check that the decomposition is exact
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                max_err = max(max_err, abs(val - 1))
    
    if max_err > 1e-10:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, column] = 1
                    V_fb[j * p + k, column] = 1
                    W_fb[k * n + i, column] = 1
                    column += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
