"""
Rank-78 decomposition for 3x4 * 4x7 matrix multiplication.

Strategy:
  Split the 3x4 matrix A into:
    A1 (2x4): rows 0,1
    A2 (1x4): row 2

  Split the 4x7 matrix B into:
    B1 (2x7): rows 0,1
    B2 (2x7): rows 2,3

  C = A * B, so:
    C1 (2x7) = A1 * B  = A1*B1 + A1*B2
    C2 (1x7) = A2 * B  = A2*B1 + A2*B2

  For C1 (2x4 * 4x7):
    Split B into B1 (2x7) and B2 (2x7).
    A1*B1 is 2x4 * 4x2 part? No.
    
    Actually, let's think differently. The known result from the cell findings:
    "Splitting n=3 into 2+1 allows using Strassen on three 2x2x2 blocks 
     and one 2x2x1 block for the 2x4x7 part, plus schoolbook for the 1x4x7 part"
    
    Wait, 2x4x7 means multiplying a 2x4 by a 4x7. The tensor for this has
    dimensions 2*4*7 = 56 entries, and schoolbook rank is 56.
    
    With Strassen-based decomposition for the 2x4x7 part:
    We can partition the 4 middle dimension into 2+2, and the 7 output into 2+2+2+1.
    
    A1 (2x4) * B (4x7):
      Let B = [B_top; B_bot] where B_top is 2x7, B_bot is 2x7.
      C1 = A1_top * B_top + A1_bot * B_bot  -- but A1 is 2x4, so A1_top is 2x2, A1_bot is 2x2.
      
      Actually A1 is 2x4: columns 0,1 (call it A1a) and columns 2,3 (call it A1b).
      B is 4x7: rows 0,1 (B_top) and rows 2,3 (B_bot).
      
      C1 = A1a * B_top + A1b * B_bot  (2x2 * 2x7 + 2x2 * 2x7)
      
      For each 2x2 * 2x7 block, we can use Strassen-like decomposition:
      Split the 7 output columns into 2+2+2+1.
      
      For 2x2 * 2x2: Strassen rank 7
      For 2x2 * 2x1: schoolbook rank 4
      
      So one 2x2 * 2x7 block has rank: 7+7+7+4 = 25
      Two such blocks: 50
      
      For C2 (1x4 * 4x7): schoolbook rank = 1*4*7 = 28
      
      Total: 50 + 28 = 78
"""

import numpy as np
import time

N, M, P = 3, 4, 7

def _strassen_2x2x2():
    """
    Return the 7 rank-1 terms for 2x2 * 2x2 matrix multiplication.
    Each term is (u, v, w) where u is in R^4 (A indices), v in R^4 (B indices), w in R^4 (C indices).
    
    Strassen:
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22)B11
      M3 = A11(B21-B22)
      M4 = A22(B12-B11)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
      
      C11 = M1 + M4 - M5 + M7
      C12 = M3 + M5
      C21 = M2 + M4
      C22 = M1 - M2 + M3 + M6
    """
    # Index conventions:
    # A is 2x2: A[0,0]=A11, A[0,1]=A12, A[1,0]=A21, A[1,1]=A22
    # B is 2x2: B[0,0]=B11, B[0,1]=B12, B[1,0]=B21, B[1,1]=B22
    # C is 2x2: C[0,0]=C11, C[0,1]=C12, C[1,0]=C21, C[1,1]=C22
    
    # For each multiplication, we define:
    # u: which A entries are involved (with coefficients)
    # v: which B entries are involved (with coefficients)
    # w: which C entries get this result (with coefficients)
    
    terms = []
    
    # M1 = (A11+A22)(B11+B22)
    # u = [1, 0, 0, 1] (A11 and A22)
    # v = [1, 0, 0, 1] (B11 and B22)
    # w: C11 += M1, C22 += M1 => [1, 0, 0, 1]
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    
    # M2 = (A21+A22)B11
    # u = [0, 0, 1, 1]
    # v = [1, 0, 0, 0]
    # w: C21 += M2, C22 -= M2 => [0, 0, 1, -1]
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    
    # M3 = A11(B21-B22)
    # u = [1, 0, 0, 0]
    # v = [0, 0, 1, -1]
    # w: C12 += M3, C22 += M3 => [0, 1, 0, 1]
    terms.append(([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]))
    
    # M4 = A22(B12-B11)
    # u = [0, 0, 0, 1]
    # v = [-1, 1, 0, 0]
    # w: C11 += M4, C21 += M4 => [1, 0, 1, 0]
    terms.append(([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 1, 0]))
    
    # M5 = (A11+A12)B22
    # u = [1, 1, 0, 0]
    # v = [0, 0, 0, 1]
    # w: C11 -= M5, C12 += M5 => [-1, 1, 0, 0]
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    
    # M6 = (A21-A11)(B11+B12)
    # u = [-1, 0, 1, 0]
    # v = [1, 1, 0, 0]
    # w: C22 += M6 => [0, 0, 0, 1]
    terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    
    # M7 = (A12-A22)(B21+B22)
    # u = [0, 1, 0, -1]
    # v = [0, 0, 1, 1]
    # w: C11 += M7 => [1, 0, 0, 0]
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    
    return terms

def _schoolbook_2x2x1():
    """
    Return the 4 rank-1 terms for 2x2 * 2x1 matrix multiplication.
    C[i] = sum_j A[i,j] * B[j]
    """
    terms = []
    for i in range(2):
        for j in range(2):
            u = [0]*4
            v = [0]*2  # B is 2x1, so 2 entries
            w = [0]*2  # C is 2x1, so 2 entries
            u[i*2 + j] = 1
            v[j] = 1
            w[i] = 1
            terms.append((u, v, w))
    return terms

def _build_rank_78():
    """
    Build the rank-78 decomposition.
    
    U shape: (n*m, rank) = (12, 78)
    V shape: (m*p, rank) = (28, 78)
    W shape: (p*n, rank) = (21, 78)
    
    Index conventions:
    U index: i*m + j where i in [0,3), j in [0,4)
    V index: j*p + k where j in [0,4), k in [0,7)
    W index: k*n + i where k in [0,7), i in [0,3)
    """
    rank = 78
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    
    col = 0
    
    # Part 1: C1 = A1 * B where A1 is rows 0,1 of A (2x4), B is 4x7
    # Split A1 into A1a (cols 0,1) and A1b (cols 2,3), each 2x2
    # Split B into B_top (rows 0,1) and B_bot (rows 2,3), each 2x7
    # C1 = A1a * B_top + A1b * B_bot
    
    # For each (A_block, B_block) pair, we do 2x2 * 2x7
    # Split the 7 columns of B_block into 2+2+2+1
    
    A_blocks = [
        (0, 0),  # A1a: rows 0-1, cols 0-1
        (0, 2),  # A1b: rows 0-1, cols 2-3
    ]
    B_blocks = [
        0,  # B_top: rows 0-1
        2,  # B_bot: rows 2-3
    ]
    
    for a_row, a_col in A_blocks:
        for b_row in B_blocks:
            # 2x2 * 2x7
            # Split 7 into 2+2+2+1
            col_splits = [(0, 2), (2, 4), (4, 6), (6, 7)]
            
            for (c_start, c_end) in col_splits:
                c_width = c_end - c_start
                
                if c_width == 2:
                    # Strassen 2x2 * 2x2: 7 terms
                    strassen_terms = _strassen_2x2x2()
                    for (u_local, v_local, w_local) in strassen_terms:
                        # u_local is 4 entries for A block (2x2)
                        # v_local is 4 entries for B block (2x2)
                        # w_local is 4 entries for C block (2x2)
                        
                        for idx in range(4):
                            if u_local[idx] != 0:
                                # A block row/col: local idx maps to (a_row + idx//2, a_col + idx%2)
                                ai = a_row + idx // 2
                                aj = a_col + idx % 2
                                U[ai * M + aj, col] += u_local[idx]
                            
                            if v_local[idx] != 0:
                                # B block row/col: local idx maps to (b_row + idx//2, c_start + idx%2)
                                bj = b_row + idx // 2
                                bk = c_start + idx % 2
                                V[bj * P + bk, col] += v_local[idx]
                            
                            if w_local[idx] != 0:
                                # C block row/col: local idx maps to (row, col) in C
                                ci = idx // 2  # C row (0 or 1, relative to A1)
                                ck = c_start + idx % 2  # C col
                                W[ck * N + ci, col] += w_local[idx]
                        
                        col += 1
                elif c_width == 1:
                    # Schoolbook 2x2 * 2x1: 4 terms
                    sb_terms = _schoolbook_2x2x1()
                    for (u_local, v_local, w_local) in sb_terms:
                        for idx in range(4):
                            if u_local[idx] != 0:
                                ai = a_row + idx // 2
                                aj = a_col + idx % 2
                                U[ai * M + aj, col] += u_local[idx]
                        
                        for idx in range(2):
                            if v_local[idx] != 0:
                                bj = b_row + idx
                                bk = c_start
                                V[bj * P + bk, col] += v_local[idx]
                        
                        for idx in range(2):
                            if w_local[idx] != 0:
                                ci = idx  # C row
                                ck = c_start
                                W[ck * N + ci, col] += w_local[idx]
                        
                        col += 1
    
    # Part 2: C2 = A2 * B where A2 is row 2 of A (1x4), B is 4x7
    # Schoolbook: 1*4*7 = 28 terms
    for j in range(4):
        for k in range(7):
            U[2 * M + j, col] = 1
            V[j * P + k, col] = 1
            W[k * N + 2, col] = 1
            col += 1
    
    # Verify col == rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify the decomposition is correct
    # T[i,j,k] = sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r]
    # This should be 1 if the (i,j,k) triple corresponds to a valid matrix mult term
    # i.e., for all i,j,k: sum_r U[i*M+j,r]*V[j*P+k,r]*W[k*N+i,r] = 1
    
    # Let's verify
    errors = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                total = 0.0
                for r in range(rank):
                    total += U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r]
                expected = 1.0
                if abs(total - expected) > 1e-9:
                    errors += 1
                    if errors <= 5:
                        print(f"Error at ({i},{j},{k}): got {total}, expected {expected}")
    
    if errors > 0:
        print(f"Verification failed with {errors} errors")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    return _build_rank_78()

_cached_result = None

def construct_final():
    global _cached_result
    if _cached_result is None:
        _cached_result = _build_rank_78()
    return _cached_result

def construct():
    global _cached_result
    if _cached_result is None:
        _cached_result = _build_rank_78()
    return _cached_result
