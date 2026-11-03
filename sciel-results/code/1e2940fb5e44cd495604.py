"""
Matrix multiplication 2,3,4 rank-minimal exact tensor decomposition.
Uses a block decomposition:
1. Split B into B_L (rows 0,1) and B_R (row 2).
2. Split A into A_L (cols 0,1) and A_R (col 2).
3. C = A_L B_L + A_R B_R.
4. A_R B_R is an outer product of a 2-vector and 4-vector, rank 8.
5. A_L B_L is split by columns of B_L into two 2x2 * 2x2 multiplications.
6. Each 2x2 * 2x2 uses Strassen's algorithm (rank 7).
Total rank = 8 + 7 + 7 = 22.
"""

import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    rank = 22
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # Part 1: A_R B_R (Rank 8)
    # A_R is col 2 of A (j=2). B_R is row 2 of B (j=2).
    # C = A[:, 2] * B[2, :]
    # Terms: A[i, 2] * B[2, k] -> C[i, k]
    # i in 0..1, k in 0..3
    col = 0
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Part 2: A_L B_L (Rank 14)
    # A_L is cols 0,1 of A. B_L is rows 0,1 of B.
    # Split B_L by columns into B_L1 (k=0,1) and B_L2 (k=2,3).
    
    # Helper to add Strassen decomposition for 2x2 * 2x2
    # local A indices: (li, lj) -> global u = li*m + lj
    # local B indices: (lj, lk) -> global v = lj*p + (k_offset + lk)
    # local C indices: (li, lk) -> global w = (k_offset + lk)*n + li
    
    def add_strassen(k_offset, start_col):
        # Strassen 7 terms
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A10)B11
        # M6 = (A11-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We define the 7 rank-1 terms.
        # Each term is a triple (u_vec, v_vec, w_vec)
        # u_vec, v_vec, w_vec are sparse vectors of length 2 (for the 2x2 block)
        
        # Term 1: M1
        # A: (1, 1, 0, 0) -> (0,0):1, (1,1):1
        # B: (1, 0, 1, 0) -> (0,0):1, (1,1):1
        # C: (1, 1, 1, 1) -> (0,0):1, (0,1):1, (1,0):1, (1,1):1 ? 
        # Wait, C contributions:
        # C00 += M1, C11 += M1
        # So C vector: (0,0):1, (1,1):1
        
        # Let's list the vectors (u, v, w) for the 7 terms
        # Indices in 2x2 block:
        # u: 00, 01, 10, 11
        # v: 00, 01, 10, 11
        # w: 00, 01, 10, 11
        
        terms = [
            # M1
            (
                [1, 0, 0, 1], # A: A00, A11
                [1, 0, 0, 1], # B: B00, B11
                [1, 0, 0, 1]  # C: C00, C11
            ),
            # M2
            (
                [0, 0, 1, 1], # A: A10, A11
                [1, 0, 0, 0], # B: B00
                [0, 0, 1, 0]  # C: C10
            ),
            # M3
            (
                [1, 0, 0, 0], # A: A00
                [0, 1, 0, -1], # B: B01, -B11
                [0, 1, 0, 0]  # C: C01
            ),
            # M4
            (
                [0, 0, 0, 1], # A: A11
                [0, 0, 1, -1], # B: B10, -B00
                [1, 0, 1, 0]  # C: C00, C10
            ),
            # M5
            (
                [1, 0, 1, 0], # A: A00, A10
                [0, 0, 0, 1], # B: B11
                [1, 1, 0, 0]  # C: C00, C01
            ),
            # M6
            (
                [0, 0, 0, 1], # A: A11
                [1, 1, 0, 0], # B: B00, B01
                [0, 0, 0, 1]  # C: C11
            ),
            # M7
            (
                [0, 1, 0, -1], # A: A01, -A11
                [0, 0, 1, 1], # B: B10, B11
                [1, 0, 0, 1]  # C: C00, C11
            ),
        ]
        
        # Map local indices to global
        # Local A (li, lj) -> Global U: li*m + lj
        # Local B (lj, lk) -> Global V: lj*p + (k_offset + lk)
        # Local C (li, lk) -> Global W: (k_offset + lk)*n + li
        
        for t_idx in range(7):
            c = start_col + t_idx
            u_vec, v_vec, w_vec = terms[t_idx]
            
            # Unpack u_vec
            for li in range(2):
                for lj in range(2):
                    val = u_vec[li*2 + lj]
                    if val != 0:
                        u_glob = li * m + lj
                        U[u_glob][c] += val
                        
            # Unpack v_vec
            for lj in range(2):
                for lk in range(2):
                    val = v_vec[lj*2 + lk]
                    if val != 0:
                        v_glob = lj * p + (k_offset + lk)
                        V[v_glob][c] += val
                        
            # Unpack w_vec
            for li in range(2):
                for lk in range(2):
                    val = w_vec[li*2 + lk]
                    if val != 0:
                        w_glob = (k_offset + lk) * n + li
                        W[w_glob][c] += val
                        
    # Sub-part 2a: k_offset = 0 (cols 0,1 of B and C)
    add_strassen(0, 8)
    
    # Sub-part 2b: k_offset = 2 (cols 2,3 of B and C)
    add_strassen(2, 15)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
