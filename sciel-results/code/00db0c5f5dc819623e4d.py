"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys
import numpy as np
from fractions import Fraction
from itertools import combinations, product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 5 matrix.
    The output factor follows the official trace convention W[k*n+i,r].
    
    Strategy:
    1. Construct the schoolbook decomposition (Rank 60).
    2. Attempt to find a lower rank (59) decomposition using a randomized sparse search.
    3. If a valid lower rank decomposition with integer/half-integer entries is found, return it.
    4. Otherwise, return the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    
    # 1. Schoolbook Decomposition (Rank 60)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(dim_u)]
    V_sb = [[0] * rank_sb for _ in range(dim_v)]
    W_sb = [[0] * rank_sb for _ in range(dim_w)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_sb[u_idx][col] = 1
                V_sb[v_idx][col] = 1
                W_sb[w_idx][col] = 1
                col += 1
                
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    # 2. Search for Rank 59
    # We will try to find a decomposition of rank 59.
    # We use a randomized approach:
    # - Generate random sparse U and V with entries in {-1, 0, 1}.
    # - Solve for W using least squares.
    # - Check if W has entries close to integers or half-integers.
    # - Verify the decomposition exactly.
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def is_nice(w_vec, tol=1e-6):
        # Check if w_vec is close to integers or half-integers
        w2 = [2 * x for x in w_vec]
        for x in w2:
            if abs(x - round(x)) > tol:
                return False
        return True

    def solve_and_verify(U, V, r):
        # U: nm x r
        # V: mp x r
        # Returns W: pn x r or None if no nice solution found
        
        W = [[0] * r for _ in range(dim_w)]
        
        # The system is:
        # For each i in [0, n), k in [0, p):
        # sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # This is not a linear system for W directly because of the product U*V.
        # Wait, the tensor equation is sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
        # For fixed i, k, the equation is:
        # sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1 ? NO.
        # The sum is over c, but j is also summed in the matrix multiplication definition?
        # No, the tensor T[i,j,k] = 1.
        # The decomposition is sum_c U_i,c V_j,c W_k,c = T_{i,j,k}.
        # Here indices i, j, k refer to the flattened indices?
        # No, the tensor is on V_A (dim nm), V_B (dim mp), V_C (dim pn).
        # T[ u_idx, v_idx, w_idx ] = 1 if u_idx=(i,j), v_idx=(j,k), w_idx=(k,i).
        # So for a fixed triple (i,j,k), we have one equation:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
        
        # This is a system of 60 equations.
        # Unknowns are the entries of W (15 * r variables).
        # But the coefficients depend on U and V.
        # The equations are coupled?
        # Equation for (i,j,k): sum_c (U[i*m+j, c] V[j*p+k, c]) W[k*n+i, c] = 1.
        # Let A_{(i,j,k), c} = U[i*m+j, c] V[j*p+k, c].
        # Then sum_c A_{(i,j,k), c} W[k*n+i, c] = 1.
        # Note that W index is k*n+i.
        # So for a fixed (i,k), we have m equations (one for each j) that all involve the same W vector W[k*n+i, :].
        # Specifically, for fixed i, k:
        # For each j in 0..m-1:
        # sum_c (U[i*m+j, c] V[j*p+k, c]) W[k*n+i, c] = 1.
        # This is a linear system for W[k*n+i, :].
        # Matrix M_{i,k} has rows j=0..m-1 and columns c=0..r-1.
        # M_{i,k}[j, c] = U[i*m+j, c] * V[j*p+k, c].
        # We need M_{i,k} * W_vec = ones(m).
        
        for i in range(n):
            for k in range(p):
                w_idx = k * n + i
                # Build the m x r matrix
                A = np.zeros((m, r))
                for j in range(m):
                    for c in range(r):
                        A[j, c] = U[i*m+j][c] * V[j*p+k][c]
                
                # Solve A * w = 1
                try:
                    w_sol, residuals, rank, s = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    if is_nice(w_sol):
                        w_rounded = [round(x * 2) / 2.0 for x in w_sol]
                        W[w_idx] = w_rounded
                    else:
                        return None
                except Exception:
                    return None
        
        # Verify the decomposition
        # Check sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-6:
                        return None
        return W

    # Try to find rank 59
    target_rank = 59
    trials = 2000
    
    for _ in range(trials):
        if time.time() > deadline:
            break
            
        # Generate random sparse U and V
        # Density ~ 0.15 to ensure some connectivity but not too dense
        U = [[0] * target_rank for _ in range(dim_u)]
        V = [[0] * target_rank for _ in range(dim_v)]
        
        for i in range(dim_u):
            for c in range(target_rank):
                if random.random() < 0.15:
                    U[i][c] = random.choice([-1, 1])
                    
        for i in range(dim_v):
            for c in range(target_rank):
                if random.random() < 0.15:
                    V[i][c] = random.choice([-1, 1])
                    
        W = solve_and_verify(U, V, target_rank)
        
        if W is not None:
            best_result = {"U": U, "V": V, "W": W}
            best_rank = target_rank
            break # Found a solution, return immediately

    return best_result
# EVOLVE-BLOCK-END
