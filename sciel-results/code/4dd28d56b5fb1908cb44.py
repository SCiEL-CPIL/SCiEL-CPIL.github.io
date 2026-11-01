"""Rewritten program: search for low-rank exact tensor decomposition of 2x3 * 3x10."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 2, 3, 10
    # Output is n x p = 2 x 10, indexed as W[k*n+i, r]
    # U: (n*m) x rank = 6 x r
    # V: (m*p) x rank = 30 x r
    # W: (p*n) x rank = 20 x r
    # Constraint: for all i in [0,n), k in [0,p):
    #   sum_r U[i*m + j, r] * V[j*p + k, r] = delta(i,k) for all j? 
    # No: the tensor is T[i,j,k] = delta_{i,k}? No.
    # Matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the identity in the "multiplication" sense)
    # Actually T[i,j,k] = 1 always. The decomposition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # Wait, that's not right either. Let me re-derive.
    #
    # The matrix multiplication tensor M has entries M[i,j,k] = 1 for all i,j,k
    # (because C[i,k] = sum_j A[i,j] B[j,k], so the coefficient of A[i,j]*B[j,k] in C[i,k] is 1).
    #
    # A rank-r decomposition means: M[i,j,k] = sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # for all i in [0,2), j in [0,3), k in [0,10).
    #
    # So we need: for all (i,j,k), sum_r U[i*3+j, r] * V[j*10+k, r] * W[k*2+i, r] = 1
    #
    # That's 60 equations. We want minimum R.
    
    best_result = None
    best_rank = n * m * p  # schoolbook = 60
    
    # Strategy: For a given rank R, we have 6*R + 30*R + 20*R = 56*R unknowns and 60 equations.
    # For R=12: 672 unknowns, 60 equations — very underdetermined.
    # We can fix some variables and solve.
    
    # Better approach: Fix W to have a specific structure, then solve for U and V.
    # Or: use the fact that for n=2, we can try to find R such that the system is solvable.
    
    # Let's try: for each rank R from 12 upward, attempt to find a solution.
    # Method: Randomly initialize one factor, then solve the resulting linear system.
    
    # The equations are: for each (i,j,k):
    #   sum_r U[i*3+j, r] * V[j*10+k, r] * W[k*2+i, r] = 1
    #
    # This is linear in U if V and W are fixed, and vice versa.
    # It's bilinear in (U,V) for fixed W.
    
    # Approach: Fix W to be a random sparse matrix, then the equations become linear in U*V (outer product structure).
    # Actually, for fixed W, the equation for (i,j,k) is:
    #   sum_r (U[i*3+j, r] * V[j*10+k, r]) * W[k*2+i, r] = 1
    # Let X[i*3+j, j*10+k] = U[i*3+j,:] dot V[j*10+k,:] (element-wise product sum)
    # Then sum_r X[i*3+j, j*10+k] * W[k*2+i, r] = 1
    # This is linear in X! But X must be a sum of R rank-1 matrices (outer products of U rows and V rows).
    
    # Alternative: For fixed W, the system is:
    #   For each (i,j,k): sum_r W[k*2+i, r] * (U[i*3+j, r] * V[j*10+k, r]) = 1
    # This is still bilinear.
    
    # Let's try a different approach: fix U to be sparse/structured, solve for V and W.
    
    # Actually, let's use a direct approach:
    # For a given R, set up the system. We have 56R unknowns, 60 equations.
    # For R >= ceil(60/56) = 2, it's underdetermined. But the nonlinearity makes it hard.
    
    # Better: Use the "one-factor-at-a-time" linearization.
    # Fix V and W randomly, then the equations are linear in U:
    #   For each (i,j,k): sum_r U[i*3+j, r] * (V[j*10+k, r] * W[k*2+i, r]) = 1
    #   => U[i*3+j, :] dot (V[j*10+k, :] * W[k*2+i, :]) = 1
    # For each (i,j), this gives p=10 equations for the 6 unknowns U[i*3+j, 0..R-1].
    # Wait, for fixed (i,j), varying k gives 10 equations in R unknowns.
    # We need R >= 10 for this to be solvable in general.
    
    # Similarly, fix U and W, solve for V: for each (j,k), varying i gives 2 equations in R unknowns.
    # Fix U and V, solve for W: for each (i,k), varying j gives 3 equations in R unknowns.
    
    # So: fix V and W, solve for U. Need R >= 10 (from 10 eqs per (i,j) block).
    # Let's try R=10,11,12,...
    
    # For fixed V, W: For each (i,j), we have a system:
    #   A_{(i,j)} * u_{(i,j)} = b_{(i,j)}
    # where A_{(i,j)}[k, r] = V[j*10+k, r] * W[k*2+i, r] for k=0..9, r=0..R-1
    # and b_{(i,j)}[k] = 1 for all k.
    # This is a 10 x R system. For R >= 10, we can solve (least squares or exact if consistent).
    
    # Let's try this approach with random V, W and check if the resulting U gives an exact decomposition.
    
    random.seed(42)
    
    for R in range(12, 25):  # try ranks 12 to 24
        if time.time() > deadline - 10:
            break
        for trial in range(200):
            if time.time() > deadline - 5:
                    break
            # Random V, W with entries from {-1, 0, 1} or small integers
            V = np.zeros((m * p, R), dtype=float)
            W = np.zeros((p * n, R), dtype=float)
            # Make them sparse-ish
            for idx in range(V.size):
                if random.random() < 0.5:
                    V.flat[idx] = random.choice([-1, 1])
            for idx in range(W.size):
                if random.random() < 0.5:
                    W.flat[idx] = random.choice([-1, 1])
            
            # Solve for U
            U = np.zeros((n * m, R), dtype=float)
            solvable = True
            for i in range(n):
                for j in range(m):
                    # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    A = np.zeros((p, R), dtype=float)
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b = np.ones(p, dtype=float)
                    # Solve A * u = b
                    if R >= p:
                        # Least squares
                        u, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        # Check if exact
                        if np.allclose(A @ u, b, atol=1e-9):
                            U[i * m + j, :] = u
                        else:
                            solvable = False
                            break
                    else:
                        solvable = False
                        break
                if not solvable:
                    break
            
            if not solvable:
                continue
            
            # Verify the full decomposition
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(R):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                # Round to nearest half-integer if close
                for mat in [U_list, V_list, W_list]:
                    for row in mat:
                        for idx, v in enumerate(row):
                            if abs(v - round(v * 2) / 2) < 1e-9:
                                row[idx] = round(v * 2) / 2
                            elif abs(v - round(v)) < 1e-9:
                                row[idx] = round(v)
                
                if best_result is None or R < best_rank:
                    best_rank = R
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                break  # found solution for this R, move to next R
        
        if best_result is not None and best_rank <= R:
            break
    
    if best_result is None:
        # Fall back to schoolbook
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
    
    return best_result
# EVOLVE-BLOCK-END
