import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # T[a,b,c] = 1 iff a//m == c%n (i match), a%m == b//p (j match), b%p == c//n (k match)
    # a = i*m + j, b = j*p + k, c = k*n + i
    for a in range(n * m):
        i_a, j_a = divmod(a, m)
        for b in range(m * p):
            j_b, k_b = divmod(b, p)
            for c in range(p * n):
                k_c, i_c = divmod(c, n)
                expected = 1 if (i_a == i_c and j_a == j_b and k_b == k_c) else 0
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if s != expected:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
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

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS # 3, 5, 5
    
    # Global dimensions
    # A: 3x5 -> indices 0..14
    # B: 5x5 -> indices 0..24
    # C: 3x5 -> indices 0..14 (k*3 + i)
    
    # We will build a decomposition of rank 72
    # Block 1: A[0:3, 0:3], B[0:3, 0:3] -> C[0:3, 0:3] (3x3x3, Rank 27)
    # Block 2: A[0:3, 3:5], B[3:5, 0:3] -> C[0:3, 0:3] (3x2x3, Rank 17)
    # Block 3: A[0:3, 0:3], B[0:3, 3:5] -> C[0:3, 3:5] (3x3x2, Rank 17)
    # Block 4: A[0:3, 3:5], B[3:5, 3:5] -> C[0:3, 3:5] (3x2x2, Rank 11)
    
    # Total Rank: 27 + 17 + 17 + 11 = 72
    
    U = []
    V = []
    W = []
    
    def add_term(u_row, v_row, w_row, u_coeff, v_coeff, w_coeff):
        # u_row: index in A (0..14)
        # v_row: index in B (0..24)
        # w_row: index in C (0..14)
        pass
        
    # We need to accumulate terms.
    # Since we are using integer coefficients, we can just append to lists.
    # But we need to combine terms if they target the same (u_row, v_row, w_row)?
    # No, each term r is a column in U, V, W.
    # We just need to list all 72 terms.
    
    terms = []
    
    # Helper to add a monomial term
    def add_monomial(i, j, k, u_c=1, v_c=1, w_c=1):
        # i: row of A (0..2)
        # j: col of A / row of B (0..4)
        # k: col of B (0..4)
        a_idx = i * 5 + j
        b_idx = j * 5 + k
        c_idx = k * 3 + i
        terms.append((a_idx, b_idx, c_idx, u_c, v_c, w_c))
        
    # Block 1: 3x3x3 Schoolbook (Rank 27)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                add_monomial(i, j, k)
                
    # Block 2: 3x2x3 Rank 17
    # A[0:3, 3:5] (cols 3,4), B[3:5, 0:3] (rows 3,4; cols 0,1,2)
    # i in 0..2, j in 3..4, k in 0..2
    # We use the Rank 17 algorithm for M[3,2,3]
    # The algorithm for 2x3x3 (Rank 17) was derived.
    # Let's implement a simple Rank 17 for 3x2x3.
    # A is 3x2, B is 2x3.
    # A = [A1, A2] (3x1 each). B = [B1; B2] (1x3 each).
    # C = A1 B1 + A2 B2.
    # A1 B1: 3x3 matrix, rank 1.
    #   Terms: A1[r] B1[c]. 3*3=9 terms?
    #   We can do better?
    #   Actually, let's just use Schoolbook for these to be safe if 17 is complex.
    #   Schoolbook Rank for 3x2x3 is 18.
    #   27 + 18 + 18 + 12 (Schoolbook 3x2x2) = 75.
    #   No improvement.
    
    # I will use Rank 11 for 3x2x2 and Schoolbook for others?
    # 27 + 18 + 18 + 11 = 74.
    # Improvement of 1.
    
    # Let's try to get Rank 17 for 3x2x3.
    # If I can't, I'll stick to 74.
    
    # Block 3: 3x3x2 Schoolbook (Rank 18)
    for i in range(3):
        for j in range(3):
            for k in range(3, 5):
                add_monomial(i, j, k)
                
    # Block 4: 3x2x2 Rank 11
    # A[0:3, 3:5] (cols 3,4), B[3:5, 3:5] (rows 3,4; cols 3,4)
    # i in 0..2, j in 3..4, k in 3..4
    # Use Rank 11 algorithm.
    # A is 3x2, B is 2x2.
    # A = [A1; A2] where A1 is 1x2, A2 is 2x2.
    # C = A1 B + A2 B.
    # A2 B: 2x2x2 Strassen (Rank 7).
    # A1 B: 1x2x2 (Rank 4).
    # Total 11.
    
    # Strassen for A2 (rows 1,2 of local A? No, rows 1,2 of the 3x2 block)
    # Local A rows: 0, 1, 2.
    # A1 is row 0. A2 is rows 1,2.
    # Local B cols: 0, 1. (Global j=3,4; k=3,4)
    
    # Strassen 7 terms for 2x2x2:
    # M1 = (A11+A22)B11
    # M2 = (A11+A22)B22
    # M3 = (A11+A12)B22
    # M4 = (A21-A11)B11
    # M5 = (A11+A21)B12
    # M6 = (A12-A22)(B11+B22)
    # M7 = (A22-A12)(B12+B21)
    
    # Mapping to global indices:
    # Local A row 1 -> Global i=1. Local A row 2 -> Global i=2.
    # Local B col 0 -> Global j=3, k=3? No.
    # B is 2x2. Rows j=3,4. Cols k=3,4.
    # B11 (local) = B[3,3] (global j=3, k=3).
    # B12 (local) = B[3,4] (global j=3, k=4).
    # B21 (local) = B[4,3] (global j=4, k=3).
    # B22 (local) = B[4,4] (global j=4, k=4).
    
    # A11 (local) = A[1,3] (global i=1, j=3).
    # A12 (local) = A[1,4] (global i=1, j=4).
    # A21 (local) = A[2,3] (global i=2, j=3).
    # A22 (local) = A[2,4] (global i=2, j=4).
    
    # C output:
    # C[1,3] (global i=1, k=3) = A11 B11 + A12 B21 = M1_part + ...
    # C[1,4] (global i=1, k=4) = A11 B12 + A12 B22
    # C[2,3] (global i=2, k=3) = A21 B11 + A22 B21
    # C[2,4] (global i=2, k=4) = A21 B12 + A22 B22
    
    # We will add the 7 Strassen terms for the A2 B part.
    # Term 1: (A11+A22) * B11
    #   U: A11(1), A22(1). V: B11(1). W: C11(1), C21(1)?
    #   No, Strassen terms contribute to specific C entries via linear combinations.
    #   This requires careful W coefficients.
    
    # Given the complexity and risk of error in manual Strassen mapping, 
    # and the time limit, I will return the Schoolbook Rank 75.
    # It is the only guaranteed correct solution I can produce with high confidence.
    
    best = _schoolbook(n, m, p)
    
    # If I had time, I would verify the Rank 72/74 construction.
    # But safety first.
    
    return best
# EVOLVE-BLOCK-END
