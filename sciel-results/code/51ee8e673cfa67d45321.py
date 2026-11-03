"""Find rank-minimal tensor decomposition for 2x4x5 matrix multiplication."""
import numpy as np
import time
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # The multiplication tensor: T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need to express T as sum of rank-1 tensors
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The tensor entry T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For the standard matrix multiplication tensor, T[i,j,k] = 1 for all valid (i,j,k)
    # This is because C[i,k] = sum_j A[i,j] * B[j,k], so each product A[i,j]*B[j,k] contributes
    # to C[i,k] with coefficient 1.
    
    # Let's verify: the tensor is T[i,j,k] = 1 for all i,j,k in range
    # We want to find rank r and factor matrices such that:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Strategy: Try ranks from theoretical minimum up to schoolbook
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # But for matrix multiplication, better bounds exist.
    
    # Let's try a direct approach: set up as a system of equations
    # For each (i,j,k), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is a system of n*m*p = 40 equations.
    # Unknowns: r * (n*m + m*p + p*n) = r * (8 + 20 + 10) = 38r
    
    # For r < 40, we have more unknowns than equations, so solutions may exist.
    # But we need the solution to be "nice" (integer or half-integer).
    
    # Let me try a different approach: use the fact that for 2xn matrix multiplication,
    # there are known decompositions. For 2x4x5, let me try rank 20, 21, ...
    
    # Actually, let me just try to find ANY decomposition with rank < 40 using optimization.
    
    # Approach: Fix V and W randomly, solve for U, check if it works.
    # Or better: use iterative approach.
    
    # Let me try: for a given rank r, generate random V and W, then for each (i,j,k),
    # we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is linear in U if V and W are fixed!
    
    # For fixed V and W, the equation for U[i*m+j, :] is:
    # sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
    # This gives p equations for (n*m) unknowns per (i,j) pair... wait no.
    
    # Let me re-index. For fixed (i,j), we have p equations (one for each k):
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1, for k=0,...,p-1
    # This is a system of p equations in r unknowns (the entries of U[i*m+j, :]).
    
    # For this to have a solution, we need r >= p = 5 (in general).
    
    # Let me try: for a given rank r, pick random V and W, then solve for U.
    # If the solution exists and is "nice", we have a decomposition.
    
    deadline = time.time() + 140  # Leave 10s margin
    
    # Try ranks from 20 to 39 (schoolbook is 40)
    for r in range(20, 40):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random V and W with small integer/half-integer entries
            # Use entries from {-1, 0, 1, 0.5, -0.5} or similar
            V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m*p, r))
            W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p*n, r))
            
            # For each (i, j), solve for U[i*m+j, :]
            U = np.zeros((n*m, r))
            feasible = True
            
            for i in range(n):
                for j in range(m):
                    # Build the p x r matrix M where M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    M = np.zeros((p, r))
                    for k in range(p):
                        for c in range(r):
                            M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    
                    # Solve M @ U[i*m+j, :] = ones(p)
                    # Use least squares
                    try:
                        sol, residuals, rank_s, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        # Check if it's an exact solution
                        if np.allclose(M @ sol, np.ones(p), atol=1e-10):
                            U[i*m+j, :] = sol
                        else:
                            feasible = False
                            break
                    except:
                        feasible = False
                        break
            
            if feasible:
                # Verify the full decomposition
                # Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                            if abs(val - 1.0) > 1e-10:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Convert to list format with integer/half-integer entries
                    U_list = [[float(U[i, c]) for c in range(r)] for i in range(n*m)]
                    V_list = [[float(V[i, c]) for c in range(r)] for i in range(m*p)]
                    W_list = [[float(W[i, c]) for c in range(r)] for i in range(p*n)]
                    
                    # Verify entries are integer or half-integer
                    def is_nice(x):
                        return abs(x * 2 - round(x * 2)) < 1e-10
                    
                    all_nice = True
                    for row in U_list:
                        for x in row:
                            if not is_nice(x):
                                all_nice = False
                                break
                        if not all_nice:
                            break
                    if all_nice:
                        for row in V_list:
                            for x in row:
                                if not is_nice(x):
                                    all_nice = False
                                    break
                            if not all_nice:
                                break
                    if all_nice:
                        for row in W_list:
                            for x in row:
                                if not is_nice(x):
                                    all_nice = False
                                    break
                            if not all_nice:
                                break
                    
                    if all_nice:
                        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
