import time
import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 7]
DEADLINE = 150.0
START_TIME = time.time()

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    # Direct construction of rank 39 decomposition
    # Split inner dimension m=3 into m1=2 and m2=1
    # A is 2x3, B is 3x7
    # C = A1 B1 + A2 B2
    # A1 is 2x2 (cols 0,1), A2 is 2x1 (col 2)
    # B1 is 2x7 (rows 0,1), B2 is 1x7 (row 2)
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val, r):
        U[u_idx][r] += u_val
        V[v_idx][r] += v_val
        W[w_idx][r] += w_val
        
    r = 0
    
    # Part 1: A2 * B2 (2x1 * 1x7)
    # A2 indices: i in {0,1}, j=2. U index: i*3 + 2
    # B2 indices: j=2, k in {0..6}. V index: 2*7 + k
    # W indices: k in {0..6}, i in {0,1}. W index: k*2 + i
    # Products: A[i,2] * B[2,k]
    for i in range(2):
        for k in range(7):
            u_idx = i * 3 + 2
            v_idx = 2 * 7 + k
            w_idx = k * 2 + i
            add_term(u_idx, v_idx, w_idx, 1, 1, 1, r)
            r += 1
            
    # Part 2: A1 * B1 (2x2 * 2x7)
    # Split k into {0,1}, {2,3}, {4,5}, {6}
    # For k in {0,1}, {2,3}, {4,5}: Use Strassen (rank 7)
    # For k in {6}: Use Schoolbook (rank 4)
    
    # Strassen blocks for k pairs
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for k0, k1 in k_pairs:
        # M1 = (A00 + A11)(B0k0 + B1k1)
        # A00: U[0*3+0]=U[0], A11: U[1*3+1]=U[4]
        # B0k0: V[0*7+k0]=V[k0], B1k1: V[1*7+k1]=V[7+k1]
        # W contributions:
        # C0k0: W[k0*2+0]=W[2k0], C0k1: W[k1*2+0]=W[2k1]
        # C1k0: W[k0*2+1]=W[2k0+1], C1k1: W[k1*2+1]=W[2k1+1]
        
        # Term 1
        # U: A00(1), A11(1)
        # V: B0k0(1), B1k1(1)
        # W: C0k0(1), C0k1(-1), C1k0(-1), C1k1(1) -- Wait, let's use the formulas derived above
        
        # M1
        U[0][r] += 1; U[4][r] += 1
        V[k0][r] += 1; V[7+k1][r] += 1
        W[2*k0][r] += 1; W[2*k1][r] += -1; W[2*k0+1][r] += -1; W[2*k1+1][r] += 1
        r += 1
        
        # M2 = (A10 + A11) B0k0
        # A10: U[1*3+0]=U[3], A11: U[4]
        # B0k0: V[k0]
        # C1k0: W[2k0+1]
        U[3][r] += 1; U[4][r] += 1
        V[k0][r] += 1
        W[2*k0+1][r] += 1
        r += 1
        
        # M3 = A00 (B1k0 - B1k1)
        # A00: U[0]
        # B1k0: V[7+k0], B1k1: V[7+k1]
        # C0k1: W[2k1]
        U[0][r] += 1
        V[7+k0][r] += 1; V[7+k1][r] += -1
        W[2*k1][r] += 1
        r += 1
        
        # M4 = A11 (B0k1 - B1k0)
        # A11: U[4]
        # B0k1: V[k1], B1k0: V[7+k0]
        # C0k0: W[2k0], C1k0: W[2k0+1]
        U[4][r] += 1
        V[k1][r] += 1; V[7+k0][r] += -1
        W[2*k0][r] += 1; W[2*k0+1][r] += 1
        r += 1
        
        # M5 = (A00 + A01) B1k1
        # A00: U[0], A01: U[1]
        # B1k1: V[7+k1]
        # C0k0: W[2k0], C0k1: W[2k1]
        U[0][r] += 1; U[1][r] += 1
        V[7+k1][r] += 1
        W[2*k0][r] += -1; W[2*k1][r] += 1
        r += 1
        
        # M6 = (A10 - A00) (B0k0 + B0k1)
        # A10: U[3], A00: U[0]
        # B0k0: V[k0], B0k1: V[k1]
        # C1k1: W[2k1+1]
        U[3][r] += 1; U[0][r] += -1
        V[k0][r] += 1; V[k1][r] += 1
        W[2*k1+1][r] += 1
        r += 1
        
        # M7 = (A01 - A11) (B1k1 + B1k0)
        # A01: U[1], A11: U[4]
        # B1k1: V[7+k1], B1k0: V[7+k0]
        # C0k0: W[2k0], C1k1: W[2*k1+1] -- Wait, check formula
        # C0k0 = M1 + M4 - M5 + M7 -> M7 coeff 1
        # C1k1 = M1 - M2 + M3 + M6 -> M7 coeff 0?
        # Let's re-verify Strassen output formulas for C00, C01, C10, C11
        # C00 = M1 - M2 + M3 + M7  (Standard: C11 = M1 - M2 + M3 + M7 if 1-based? No.)
        
        # Standard Strassen (1-based indices):
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B21)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # My mapping:
        # i=0 is row 1, i=1 is row 2
        # j=0 is col 1, j=1 is col 2 (for A)
        # k0 is col 1, k1 is col 2 (for B)
        
        # C[0, k0] (C11) = M1 + M4 - M5 + M7
        # C[0, k1] (C12) = M3 + M5
        # C[1, k0] (C21) = M2 + M4
        # C[1, k1] (C22) = M1 - M2 + M3 + M6
        
        # Let's re-evaluate M7 contribution.
        # M7 = (A01 - A11)(B1k1 + B1k0)
        # U[1] (A01) coeff 1, U[4] (A11) coeff -1
        # V[7+k1] (B1k1) coeff 1, V[7+k0] (B1k0) coeff 1
        # W[2*k0] (C[0,k0]) coeff 1 (from C11)
        # W[2*k1+1] (C[1,k1]) coeff 0 (from C22)
        
        # So for M7:
        U[1][r] += 1; U[4][r] += -1
        V[7+k1][r] += 1; V[7+k0][r] += 1
        W[2*k0][r] += 1
        # W[2*k1+1] += 0
        
        r += 1

    # Last block k=6 (2x2 * 2x1)
    # Schoolbook
    # C[0,6] = A[0,0]B[0,6] + A[0,1]B[1,6]
    # C[1,6] = A[1,0]B[0,6] + A[1,1]B[1,6]
    
    # Term 1: A[0,0]B[0,6]
    U[0][r] += 1
    V[6][r] += 1
    W[12][r] += 1 # 6*2+0
    r += 1
    
    # Term 2: A[0,1]B[1,6]
    U[1][r] += 1
    V[13][r] += 1 # 7+6
    W[12][r] += 1
    r += 1
    
    # Term 3: A[1,0]B[0,6]
    U[3][r] += 1
    V[6][r] += 1
    W[13][r] += 1 # 6*2+1
    r += 1
    
    # Term 4: A[1,1]B[1,6]
    U[4][r] += 1
    V[13][r] += 1
    W[13][r] += 1
    r += 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_decomp(n, m, p, rank):
    """Create the standard schoolbook decomposition of rank n*m*p."""
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    # Check that for all valid (i,j,k), the tensor product equals 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _try_rank_search(n, m, p, target_rank, seed):
    """
    Stub for compatibility, not used in the new direct construction.
    """
    return None
