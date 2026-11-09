import time
import random
import itertools
import numpy as np
from scipy.linalg import qr, solve

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    def make_strassen_2x2():
        """
        Return U, V, W for 2x2x2 Strassen with shapes (4,7), (4,7), (4,7).
        U indexed as U[i*2+j], V as V[j*2+k], W as W[k*2+i] for i,j,k in {0,1}.
        Must satisfy: for all (i,j,k) in {0,1}^3, sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r] = 1
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen 7 products with correct tensor decomposition
        # For each r, U[i,j,r]*V[j,k,r]*W[k,i,r] must sum to 1 over r for all (i,j,k)
        
        # M1 = (A00+A11)*(B00+B11): 
        # U picks A00,A11; V picks B00,B11; W distributes to C00,C11
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)*B00:
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A00*(B01-B11):
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11*(B10-B00):
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)*B11:
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)*(B00+B01):
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)*(B10+B11):
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def verify_decomposition(U, V, W, rank, n, m, p):
        """Check: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += W[k * n + i, r] * U[i * m + j, r] * V[j * p + k, r]
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True
    
    def verify_decomposition(U, V, W, rank, n, m, p):
        """Check: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += W[k * n + i, r] * U[i * m + j, r] * V[j * p + k, r]
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

    def make_blocked_strassen():
        """
        Build a blocked Strassen decomposition for 3x4x6.
        Split: A rows [0,1] and [2], A cols [0,1] and [2,3], B cols [0,1],[2,3],[4,5]
        6 blocks of type (2,2,2) using Strassen (rank 7 each)
        6 blocks of type (1,2,2) using schoolbook (rank 4 each)
        Total rank: 6*7 + 6*4 = 66
        """
        n, m, p = 3, 4, 6
        total_rank = 6 * 7 + 6 * 4  # 66
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        # Get Strassen 2x2x2 decomposition
        sU, sV, sW = make_strassen_2x2()
        
        col = 0
        # A row blocks: [0,1] (2 rows), [2] (1 row)
        # A col blocks (and B row blocks): [0,1] (2 cols/rows), [2,3] (2 cols/rows)
        # B col blocks: [0,1], [2,3], [4,5] (three 2-col blocks)
        
        # For each combination:
        # a_row_start, a_row_end: rows of A
        # a_col_start, a_col_end: cols of A = rows of B
        # b_col_start, b_col_end: cols of B = cols of C
        
        a_row_blocks = [(0, 2), (2, 3)]  # (start, end)
        a_col_blocks = [(0, 2), (2, 4)]  # (start, end)
        b_col_blocks = [(0, 2), (2, 4), (4, 6)]  # (start, end)
        
        for (ar_s, ar_e) in a_row_blocks:
            ar_size = ar_e - ar_s  # 2 or 1
            for (ac_s, ac_e) in a_col_blocks:
                ac_size = ac_e - ac_s  # 2
                b_row_s = ac_s  # B rows match A cols
                b_row_e = ac_e
                for (bc_s, bc_e) in b_col_blocks:
                    bc_size = bc_e - bc_s  # 2
                    
                    if ar_size == 2:
                        # Use Strassen for (2,2,2) block
                        for r in range(7):
                            # U: for i in [ar_s, ar_e), j in [ac_s, ac_e)
                            # U[i*m + j, col] = sU[i-ar_s]*2 + (j-ac_s) is wrong
                            # U index: (i-ar_s)*2 + (j-ac_s) in sU
                            for ii in range(2):
                                for jj in range(2):
                                    U[(ar_s + ii) * m + (ac_s + jj), col] = sU[ii * 2 + jj, r]
                            # V: for j in [b_row_s, b_row_e), k in [bc_s, bc_e)
                            # V[j*p + k, col] = sV[j-b_row_s]*2 + (k-bc_s)
                            for jj in range(2):
                                for kk in range(2):
                                    V[(b_row_s + jj) * p + (bc_s + kk), col] = sV[jj * 2 + kk, r]
                            # W: for k in [bc_s, bc_e), i in [ar_s, ar_e)
                            # W[k*n + i, col] = sW[k-bc_s]*2 + (i-ar_s)
                            for kk in range(2):
                                for ii in range(2):
                                    W[(bc_s + kk) * n + (ar_s + ii), col] = sW[kk * 2 + ii, r]
                            col += 1
                    else:
                        # ar_size == 1: use schoolbook for (1,2,2) block, rank 4
                        # i is fixed at ar_s (single row)
                        i0 = ar_s
                        for jj in range(2):
                            for kk in range(2):
                                # U[i0*m + (ac_s+jj), col] = 1
                                # V[(b_row_s+jj)*p + (bc_s+kk), col] = 1
                                # W[(bc_s+kk)*n + i0, col] = 1
                                U[i0 * m + (ac_s + jj), col] = 1.0
                                V[(b_row_s + jj) * p + (bc_s + kk), col] = 1.0
                                W[(bc_s + kk) * n + i0, col] = 1.0
                                col += 1
        
        assert col == total_rank, f"Column count {col} != expected {total_rank}"
        
        # Verify
        if not verify_decomposition(U, V, W, total_rank, n, m, p):
            return None
        
        return U, V, W, total_rank

    # Start with schoolbook
    best_U, best_V, best_W, best_rank = make_schoolbook()
    
    if time.time() - start_time > deadline - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try blocked Strassen
    result = make_blocked_strassen()
    if result is not None:
        bU, bV, bW, brank = result
        if brank < best_rank:
            best_U, best_V, best_W, best_rank = bU, bV, bW, brank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
