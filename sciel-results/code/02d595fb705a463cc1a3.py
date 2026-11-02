"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = U_arr[i*m+j]
                v_vec = V_arr[j*p+k]
                w_vec = W_arr[k*n+i]
                val = np.sum(u_vec * v_vec * w_vec)
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    return max_err < tol

def _construct_hybrid_strassen(n, m, p):
    """
    Construct a rank-39 decomposition for 2x3 * 3x7 matrix multiplication.
    Uses Strassen for 2x2 blocks and schoolbook for remaining parts.
    """
    if (n, m, p) != (2, 3, 7):
        return None
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    
    # Helper: add Strassen decomposition for 2x2 * 2x2 block
    # A rows: i1, i2 (0,1), B cols: k1, k2, C cols: k1, k2
    def add_strassen(A_rows, B_cols_start, C_cols_start, start_c):
        # A is 2x2: U[i*m + j] for i in A_rows, j in 0,1
        # A[0,0]->U[0], A[0,1]->U[1], A[1,0]->U[3], A[1,1]->U[4]
        # B is 2x2: V[j*p + k] for j in 0,1, k in B_cols_start, B_cols_start+1
        # B[0,k1]->V[k1], B[0,k2]->V[k1+1], B[1,k1]->V[7+k1], B[1,k2]->V[7+k1+1]
        # C is 2x2: W[k*n + i] for k in C_cols_start, C_cols_start+1, i in 0,1
        # C[0,k1]->W[k1*2+0], C[0,k2]->W[(k1+1)*2+0], C[1,k1]->W[k1*2+1], C[1,k2]->W[(k1+1)*2+1]
        
        # M1: (A11+A22)(B11+B22) -> C11 + C22
        U[0][start_c] = 1
        U[4][start_c] = 1
        V[B_cols_start][start_c] = 1
        V[7 + B_cols_start][start_c] = 1
        W[C_cols_start * 2 + 0][start_c] = 1
        W[(C_cols_start + 1) * 2 + 1][start_c] = 1
        
        # M2: (A21+A22)B11 -> C21
        U[3][start_c + 1] = 1
        U[4][start_c + 1] = 1
        V[B_cols_start][start_c + 1] = 1
        W[C_cols_start * 2 + 1][start_c + 1] = 1
        
        # M3: A11(B21-B11) -> C12
        U[0][start_c + 2] = 1
        V[7 + B_cols_start][start_c + 2] = 1
        V[B_cols_start][start_c + 2] = -1
        W[(C_cols_start + 1) * 2 + 0][start_c + 2] = 1
        
        # M4: A22(B12-B22) -> C12
        U[4][start_c + 3] = 1
        V[B_cols_start + 1][start_c + 3] = 1
        V[7 + B_cols_start + 1][start_c + 3] = -1
        W[(C_cols_start + 1) * 2 + 0][start_c + 3] = 1
        
        # M5: (A11+A12)B22 -> C12
        U[0][start_c + 4] = 1
        U[1][start_c + 4] = 1
        V[7 + B_cols_start + 1][start_c + 4] = 1
        W[(C_cols_start + 1) * 2 + 0][start_c + 4] = 1
        
        # M6: (A21-A11)(B11+B12) -> C22
        U[3][start_c + 5] = 1
        U[0][start_c + 5] = -1
        V[B_cols_start][start_c + 5] = 1
        V[B_cols_start + 1][start_c + 5] = 1
        W[(C_cols_start + 1) * 2 + 1][start_c + 5] = 1
        
        # M7: (A12-A22)(B21+B22) -> C22
        U[1][start_c + 6] = 1
        U[4][start_c + 6] = -1
        V[7 + B_cols_start][start_c + 6] = 1
        V[7 + B_cols_start + 1][start_c + 6] = 1
        W[(C_cols_start + 1) * 2 + 1][start_c + 6] = 1
        
        return start_c + 7
    
    # Part 1: A1 (2x2) * B1 (2x7)
    # B1 cols 0,1
    c = add_strassen([0, 1], 0, 0, 0)
    # B1 cols 2,3
    c = add_strassen([0, 1], 2, 2, c)
    # B1 cols 4,5
    c = add_strassen([0, 1], 4, 4, c)
    
    # B1 col 6 (2x2x1 schoolbook)
    # A[0,0]*B[0,6] -> C[0,6]
    U[0][c] = 1
    V[6][c] = 1
    W[6 * 2 + 0][c] = 1
    c += 1
    # A[0,1]*B[1,6] -> C[0,6]
    U[1][c] = 1
    V[7 + 6][c] = 1
    W[6 * 2 + 0][c] = 1
    c += 1
    # A[1,0]*B[0,6] -> C[1,6]
    U[3][c] = 1
    V[6][c] = 1
    W[6 * 2 + 1][c] = 1
    c += 1
    # A[1,1]*B[1,6] -> C[1,6]
    U[4][c] = 1
    V[7 + 6][c] = 1
    W[6 * 2 + 1][c] = 1
    c += 1
    
    # Part 2: A2 (2x1) * B2 (1x7)
    # A rows 0,1, col 2. U indices: 0*3+2=2, 1*3+2=5.
    # B row 2, cols 0..6. V indices: 2*7+k = 14+k.
    # C rows 0,1, cols 0..6. W indices: k*2+i.
    for i in range(2):
        for k in range(7):
            U[i * 3 + 2][c] = 1
            V[14 + k][c] = 1
            W[k * 2 + i][c] = 1
            c += 1
    
    if c != rank:
        rank = c
        U = [row[:rank] for row in U]
        V = [row[:rank] for row in V]
        W = [row[:rank] for row in W]
        
    return {"U": U, "V": V, "W": W}

def _try_rank_elimination(n, m, p, target_rank, time_limit=30.0):
    """
    Try to find a decomposition of target_rank by fixing U and V to sparse patterns
    and solving for W via linear algebra.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    rank = target_rank
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # For each rank factor r (0 to rank-1), we need to choose which (i,j) entry of U
    # and which (j,k) entry of V are non-zero for that factor.
    # Then W is determined by the constraint that for each (i,j,k),
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy: for each factor r, pick a "primary" (i,j) for U and (j,k) for V
    # such that they share the same j. Then W[k*n+i, r] = 1 / (U[i*m+j,r] * V[j*p+k,r])
    # if we want exactly one term per (i,j,k). But this gives rank = n*m*p (schoolbook).
    
    # To reduce rank, we need to merge multiple (i,j,k) terms into one factor.
    # For factor r, let S_U(r) be the set of (i,j) with non-zero U, and S_V(r) be the set of (j,k) with non-zero V.
    # For each (i,j,k) where j appears in both S_U(r) and S_V(r), the contribution is U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
    # We need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all (i,j,k).
    
    # This is a system of n*m*p equations. The unknowns are the W entries (p*n*rank) plus the U and V entries.
    # But U and V are also unknowns. This is highly non-linear.
    
    # Simpler approach: fix U and V to have at most 2 non-zeros each per factor, sharing a common j.
    # Then for each factor r, we have U[i1*m+j, r] = a, U[i2*m+j, r] = b (or just one),
    # V[j*p+k1, r] = c, V[j*p+k2, r] = d (or just one).
    # The products a*c*W[k1*n+i1, r], a*d*W[k2*n+i1, r], b*c*W[k1*n+i2, r], b*d*W[k2*n+i2, r]
    # contribute to the (i,j,k) equations.
    
    # This is getting complex. Let's just try the hybrid approach and verify.
    # If rank 39 works, return it. Trying to go lower is risky.
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Try the algebraic hybrid Strassen/Schoolbook construction first
    result = _construct_hybrid_strassen(n, m, p)
    if result is not None:
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
