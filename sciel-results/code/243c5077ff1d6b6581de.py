"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # Define Strassen decomposition for 2x2x2
    # Strassen's 7 products:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We represent the 2x2x2 tensor as a sum of 7 rank-1 terms.
    # The indices for the 2x2x2 block are local (li, lj, lk) in {0,1}^3.
    # U[li*2+lj, r], V[lj*2+lk, r], W[lk*2+li, r]
    
    # Let's define the 7 terms as (U_vec, V_vec, W_vec)
    # U_vec is length 4 (indices 00, 01, 10, 11 corresponding to li*2+lj)
    # V_vec is length 4 (indices 00, 01, 10, 11 corresponding to lj*2+lk)
    # W_vec is length 4 (indices 00, 01, 10, 11 corresponding to lk*2+li)
    
    strassen_terms = []
    
    # M1: (A00 + A11) * (B00 + B11)
    # U: [1, 0, 0, 1] (A00, A01, A10, A11) -> A00+A11
    # V: [1, 0, 0, 1] (B00, B01, B10, B11) -> B00+B11
    # W: C00 gets +1, C11 gets +1.
    # W indices: 00 (C00), 01 (C01), 10 (C10), 11 (C11)
    # W: [1, 0, 0, 1]
    strassen_terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    
    # M2: (A10 + A11) * B00
    # U: [0, 0, 1, 1]
    # V: [1, 0, 0, 0]
    # W: C10 gets +1.
    # W: [0, 0, 1, 0]
    strassen_terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
    
    # M3: A00 * (B01 - B11)
    # U: [1, 0, 0, 0]
    # V: [0, 1, 0, -1]
    # W: C01 gets +1.
    # W: [0, 1, 0, 0]
    strassen_terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
    
    # M4: A11 * (B10 - B00)
    # U: [0, 0, 0, 1]
    # V: [0, 0, 1, -1]
    # W: C00 gets +1, C10 gets +1.
    # W: [1, 0, 1, 0]
    strassen_terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]))
    
    # M5: (A00 + A01) * B11
    # U: [1, 1, 0, 0]
    # V: [0, 0, 0, 1]
    # W: C00 gets -1, C01 gets +1.
    # W: [-1, 1, 0, 0]
    strassen_terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    
    # M6: (A10 - A00) * (B00 + B01)
    # U: [-1, 0, 1, 0]
    # V: [1, 1, 0, 0]
    # W: C11 gets +1.
    # W: [0, 0, 0, 1]
    strassen_terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    
    # M7: (A01 - A11) * (B10 + B11)
    # U: [0, 1, 0, -1]
    # V: [0, 0, 1, 1]
    # W: C00 gets +1, C11 gets +1.
    # W: [1, 0, 0, 1]
    strassen_terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]))
    
    # Define the 4 disjoint 2x2x2 blocks
    # (i_offset, j_offset, k_offset)
    blocks = [
        (0, 0, 0), # i in {0,1}, j in {0,1}, k in {0,1}
        (0, 2, 2), # i in {0,1}, j in {2,3}, k in {2,3}
        (1, 2, 0), # i in {1,2}, j in {2,3}, k in {0,1}
        (1, 0, 2)  # i in {1,2}, j in {0,1}, k in {2,3}
    ]
    
    # Mark which (i,j,k) triples are covered by Strassen blocks
    covered = set()
    for i_off, j_off, k_off in blocks:
        for li in range(2):
            for lj in range(2):
                for lk in range(2):
                    covered.add((i_off + li, j_off + lj, k_off + lk))
    
    # Calculate total rank
    rank = len(blocks) * 7 + (n * m * p - len(covered))
    
    # Initialize matrices
    U_rows = n * m  # 15
    V_rows = m * p  # 25
    W_rows = p * n  # 15
    
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    col = 0
    
    # Add Strassen blocks
    for i_off, j_off, k_off in blocks:
        for u_c, v_c, w_c in strassen_terms:
            # U: A[i_off+li, j_off+lj]
            for li in range(2):
                for lj in range(2):
                    U[(i_off + li) * m + (j_off + lj)][col] = u_c[li * 2 + lj]
            # V: B[j_off+lj, k_off+lk]
            for lj in range(2):
                for lk in range(2):
                    V[(j_off + lj) * p + (k_off + lk)][col] = v_c[lj * 2 + lk]
            # W: C[i_off+li, k_off+lk] -> W[(k_off+lk)*n + (i_off+li)]
            for lk in range(2):
                for li in range(2):
                    W[(k_off + lk) * n + (i_off + li)][col] = w_c[lk * 2 + li]
            col += 1
    
    # Add schoolbook for remaining (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
