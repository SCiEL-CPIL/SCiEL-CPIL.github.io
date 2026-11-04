"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product
import scipy.optimize

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-22 exact tensor decomposition for 2x4x5 matrix multiplication.
    
    Strategy: 1+3 partition of the inner dimension.
    - A(2x4) = A1(2x1) + A2(2x3)
    - B(4x5) = B1(1x5) + B2(3x5)
    - C = A1*B1 + A2*B2
    
    Sub-decompositions:
    1. A1(2x1)*B1(1x5) -> 2x5: rank 2
    2. A2(2x3)*B2(3x5) -> 2x5: 
       - Further split inner 3 as 1+2:
         a. A21(2x1)*B21(1x5) -> 2x5: rank 2
         b. A22(2x2)*B22(2x5) -> 2x5: rank 18 (Strassen 7+7 + schoolbook 4)
       - Subtotal: 20
    Total rank: 2 + 20 = 22
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank = 22
    U = np.zeros((n*m, rank))  # 8 x 22
    V = np.zeros((m*p, rank))  # 20 x 22
    W = np.zeros((p*n, rank))  # 10 x 22
    
    def add_term(idx, u_coeffs, v_coeffs, w_coeffs):
        """
        Add a rank-1 term.
        u_coeffs: dict {(i, j): val} for A[i,j]
        v_coeffs: dict {(j, k): val} for B[j,k]
        w_coeffs: dict {(k, i): val} for C[k,i] (trace convention: W[k*n+i, r])
        """
        for (i, j), val in u_coeffs.items():
            U[i*m + j, idx] += val
        for (j, k), val in v_coeffs.items():
            V[j*p + k, idx] += val
        for (k, i), val in w_coeffs.items():
            W[k*n + i, idx] += val

    def strassen_2x2x2(a_col_off, b_row_off, c_col_off, idx_start):
        """Strassen decomposition for A[a_col_off:a_col_off+2, :] * B[:, c_col_off:c_col_off+2] -> C[:, c_col_off:c_col_off+2]
        where A is 2x2 block (rows 0,1; cols a_col_off, a_col_off+1)
        and B is 2x2 block (rows b_row_off, b_row_off+1; cols c_col_off, c_col_off+1)
        """
        # A block: rows {0,1}, cols {a_col_off, a_col_off+1}
        # B block: rows {b_row_off, b_row_off+1}, cols {c_col_off, c_col_off+1}
        # C block: rows {0,1}, cols {c_col_off, c_col_off+1}
        
        # S1 = (a+d)(e+h)
        add_term(idx_start,
                 {(0, a_col_off): 1, (1, a_col_off+1): 1},
                 {(b_row_off, c_col_off): 1, (b_row_off+1, c_col_off+1): 1},
                 {(c_col_off, 0): 1, (c_col_off+1, 1): 1})
        
        # S2 = (c+d)e
        add_term(idx_start+1,
                 {(1, a_col_off): 1, (1, a_col_off+1): 1},
                 {(b_row_off, c_col_off): 1},
                 {(c_col_off, 1): 1, (c_col_off+1, 1): -1})
        
        # S3 = a(f-h)
        add_term(idx_start+2,
                 {(0, a_col_off): 1},
                 {(b_row_off, c_col_off+1): 1, (b_row_off+1, c_col_off+1): -1},
                 {(c_col_off+1, 0): 1, (c_col_off+1, 1): 1})
        
        # S4 = d(g-e)
        add_term(idx_start+3,
                 {(1, a_col_off+1): 1},
                 {(b_row_off+1, c_col_off): 1, (b_row_off, c_col_off): -1},
                 {(c_col_off, 0): 1, (c_col_off, 1): 1})
        
        # S5 = (a+b)h
        add_term(idx_start+4,
                 {(0, a_col_off): 1, (0, a_col_off+1): 1},
                 {(b_row_off+1, c_col_off+1): 1},
                 {(c_col_off, 0): -1, (c_col_off+1, 0): 1})
        
        # S6 = (c-a)(e+f)
        add_term(idx_start+5,
                 {(1, a_col_off): 1, (0, a_col_off): -1},
                 {(b_row_off, c_col_off): 1, (b_row_off, c_col_off+1): 1},
                 {(c_col_off+1, 1): 1})
        
        # S7 = (b-d)(g+h)
        add_term(idx_start+6,
                 {(0, a_col_off+1): 1, (1, a_col_off+1): -1},
                 {(b_row_off+1, c_col_off): 1, (b_row_off+1, c_col_off+1): 1},
                 {(c_col_off, 0): 1})

    def schoolbook_2x2x1(a_col_off, b_row_off, c_col_off, idx_start):
        """Schoolbook for A 2x2 block * B 2x1 block -> C 2x1 block"""
        add_term(idx_start,
                 {(0, a_col_off): 1},
                 {(b_row_off, c_col_off): 1},
                 {(c_col_off, 0): 1})
        add_term(idx_start+1,
                 {(0, a_col_off+1): 1},
                 {(b_row_off+1, c_col_off): 1},
                 {(c_col_off, 0): 1})
        add_term(idx_start+2,
                 {(1, a_col_off): 1},
                 {(b_row_off, c_col_off): 1},
                 {(c_col_off, 1): 1})
        add_term(idx_start+3,
                 {(1, a_col_off+1): 1},
                 {(b_row_off+1, c_col_off): 1},
                 {(c_col_off, 1): 1})

    def rank2_2x1x5(a_col_off, b_row_off, idx_start):
        """Decompose A(2x1) * B(1x5) -> C(2x5) with rank 2.
        A has entries at (0, a_col_off) and (1, a_col_off).
        B has entry at (b_row_off, k) for k=0..4.
        C has entries at (k, 0) and (k, 1) for k=0..4.
        """
        # Term 1: a[0, a_col_off] * B row b_row_off -> C row 0
        v_coeffs_1 = {}
        w_coeffs_1 = {}
        for k in range(5):
            v_coeffs_1[(b_row_off, k)] = 1
            w_coeffs_1[(k, 0)] = 1
        add_term(idx_start, {(0, a_col_off): 1}, v_coeffs_1, w_coeffs_1)
        
        # Term 2: a[1, a_col_off] * B row b_row_off -> C row 1
        v_coeffs_2 = {}
        w_coeffs_2 = {}
        for k in range(5):
            v_coeffs_2[(b_row_off, k)] = 1
            w_coeffs_2[(k, 1)] = 1
        add_term(idx_start+1, {(1, a_col_off): 1}, v_coeffs_2, w_coeffs_2)

    idx = 0
    
    # Part 1: A1(2x1) * B1(1x5) where A1 = A[:, 0], B1 = B[0, :]
    # A1 has entries at (0,0) and (1,0). B1 has entries at (0, k) for k=0..4.
    rank2_2x1x5(a_col_off=0, b_row_off=0, idx_start=idx)
    idx += 2
    
    # Part 2: A2(2x3) * B2(3x5) where A2 = A[:, 1:4], B2 = B[1:4, :]
    # Split inner 3 as 1+2:
    #   A21(2x1) = A[:, 1], B21(1x5) = B[1, :]
    #   A22(2x2) = A[:, 2:4], B22(2x5) = B[2:4, :]
    
    # Part 2a: A21(2x1) * B21(1x5) -> rank 2
    rank2_2x1x5(a_col_off=1, b_row_off=1, idx_start=idx)
    idx += 2
    
    # Part 2b: A22(2x2) * B22(2x5) -> rank 18
    # A22: rows {0,1}, cols {2,3}
    # B22: rows {2,3}, cols {0,1,2,3,4}
    # Split B22 cols as 2+2+1:
    
    # B22 cols 0,1: 2x2x2 Strassen
    strassen_2x2x2(a_col_off=2, b_row_off=2, c_col_off=0, idx_start=idx)
    idx += 7
    
    # B22 cols 2,3: 2x2x2 Strassen
    strassen_2x2x2(a_col_off=2, b_row_off=2, c_col_off=2, idx_start=idx)
    idx += 7
    
    # B22 col 4: 2x2x1 schoolbook
    schoolbook_2x2x1(a_col_off=2, b_row_off=2, c_col_off=4, idx_start=idx)
    idx += 4
    
    if idx != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {idx}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
