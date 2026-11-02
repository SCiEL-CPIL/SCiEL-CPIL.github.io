"""Rewritten program for matrix multiplication 2x4x5 with rank 36."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 36
    rank = 36
    
    # Shapes
    # U: (n*m, rank) = (8, 36)
    # V: (m*p, rank) = (20, 36)
    # W: (p*n, rank) = (10, 36)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict { (i, j): coeff } for A
    # v_coeffs: dict { (j, k): coeff } for B
    # w_coeffs: dict { (i, k): coeff } for C
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for (i, j), val in u_coeffs.items():
            U[i * m + j][col] = val
        for (j, k), val in v_coeffs.items():
            V[j * p + k][col] = val
        for (i, k), val in w_coeffs.items():
            W[k * n + i][col] = val
        col += 1

    # Strassen helper for 2x2 * 2x2
    # A is 2x2, B is 2x2
    # Returns list of 7 terms (u, v, w)
    # u, v, w are dicts for the local 2x2 blocks
    # We need to map local indices to global indices
    
    def strassen_terms(offset_i, offset_j, offset_k):
        # offset_i: row offset for A (0)
        # offset_j: col offset for A (0 or 2)
        # offset_k: row offset for B (0 or 2) and col offset for B (0, 2, 4?)
        # Actually, Strassen is for 2x2 * 2x2.
        # A_local is 2x2, B_local is 2x2.
        # A indices: (offset_i + 0, offset_j + 0), (offset_i + 0, offset_j + 1), etc.
        # B indices: (offset_k + 0, offset_k + 0) ... wait, B is m x p.
        # B rows are offset_k + 0, offset_k + 1.
        # B cols are offset_k + 0, offset_k + 1? No, B cols are independent.
        # The "k" in B is the column of B.
        # In the block $A_L B_{T1}$:
        # A_L uses A cols 0,1.
        # B_T1 uses B rows 0,1 and B cols 0,1.
        # So B_local is 2x2.
        
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        u1 = {(offset_i+0, offset_j+0): 1, (offset_i+1, offset_j+1): 1}
        v1 = {(offset_k+0, offset_k+0): 1, (offset_k+1, offset_k+1): 1}
        w1 = {(offset_i+0, offset_k+0): 1, (offset_i+0, offset_k+1): 0, (offset_i+1, offset_k+0): 0, (offset_i+1, offset_k+1): 0}
        # Wait, w1 maps to C.
        # C00 gets M1. C01 gets nothing from M1?
        # Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So w for M1:
        # w[0,0] += 1
        # w[1,1] += 1
        
        # Let's define w dict for each M
        # M1: w(0,0)=1, w(1,1)=1
        # M2: w(1,0)=1, w(1,1)=-1
        # M3: w(0,1)=1, w(1,1)=1
        # M4: w(0,0)=1, w(1,0)=1
        # M5: w(0,0)=-1, w(0,1)=1
        # M6: w(1,1)=1
        # M7: w(0,0)=1
        
        # Let's re-verify Strassen output mapping
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A10)B11
        # M6 = (A01-A00)(B00+B10)
        # M7 = (A11-A01)(B01+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # u/v definitions:
        # M1: u: A00, A11. v: B00, B11.
        # M2: u: A10, A11. v: B00.
        # M3: u: A00. v: B10 (1), B11 (-1).
        # M4: u: A11. v: B01 (1), B00 (-1).
        # M5: u: A00, A10. v: B11.
        # M6: u: A01, A00(-1). v: B00, B10.
        # M7: u: A11, A01(-1). v: B01, B11.
        
        # Global indexing:
        # A[i, j]
        # B[j, k]  (Note: B row is j, col is k)
        # C[i, k]
        
        # For block A(offset_j) * B(offset_k):
        # A local (0,0) -> Global (offset_i, offset_j)
        # A local (1,1) -> Global (offset_i+1, offset_j+1)
        # B local (0,0) -> Global (offset_k, offset_k)
        # B local (1,1) -> Global (offset_k+1, offset_k+1)
        # C local (0,0) -> Global (offset_i, offset_k)
        # C local (1,1) -> Global (offset_i+1, offset_k+1)
        
        # Note: B row index is offset_k + local_row.
        # B col index is offset_k + local_col.
        
        def map_u(local):
            return {(offset_i + i, offset_j + j): v for (i,j), v in local.items()}
        def map_v(local):
            return {(offset_k + i, offset_k + j): v for (i,j), v in local.items()}
        def map_w(local):
            return {(offset_i + i, offset_k + j): v for (i,j), v in local.items()}

        # M1
        u = map_u({(0,0):1, (1,1):1})
        v = map_v({(0,0):1, (1,1):1})
        w = map_w({(0,0):1, (1,1):1})
        terms.append((u, v, w))
        
        # M2
        u = map_u({(1,0):1, (1,1):1})
        v = map_v({(0,0):1})
        w = map_w({(1,0):1, (1,1):-1})
        terms.append((u, v, w))
        
        # M3
        u = map_u({(0,0):1})
        v = map_v({(1,0):1, (1,1):-1})
        w = map_w({(0,1):1, (1,1):1})
        terms.append((u, v, w))
        
        # M4
        u = map_u({(1,1):1})
        v = map_v({(0,1):1, (0,0):-1})
        w = map_w({(0,0):1, (1,0):1})
        terms.append((u, v, w))
        
        # M5
        u = map_u({(0,0):1, (1,0):1})
        v = map_v({(1,1):1})
        w = map_w({(0,0):-1, (0,1):1})
        terms.append((u, v, w))
        
        # M6
        u = map_u({(0,1):1, (0,0):-1})
        v = map_v({(0,0):1, (1,0):1})
        w = map_w({(1,1):1})
        terms.append((u, v, w))
        
        # M7
        u = map_u({(1,1):1, (0,1):-1})
        v = map_v({(0,1):1, (1,1):1})
        w = map_w({(0,0):1})
        terms.append((u, v, w))
        
        return terms

    def standard_2x2x1_terms(offset_i, offset_j, offset_k):
        # A 2x2 * B 2x1
        # B col is fixed at offset_k
        # A local (0,0) -> Global (offset_i, offset_j)
        # A local (0,1) -> Global (offset_i, offset_j+1)
        # A local (1,0) -> Global (offset_i+1, offset_j)
        # A local (1,1) -> Global (offset_i+1, offset_j+1)
        # B local (0,0) -> Global (offset_k, offset_k)
        # B local (1,0) -> Global (offset_k+1, offset_k)
        # C local (0,0) -> Global (offset_i, offset_k)
        # C local (1,0) -> Global (offset_i+1, offset_k)
        
        terms = []
        # Term 1: A00 * B0
        u = {(offset_i+0, offset_j+0): 1}
        v = {(offset_k+0, offset_k): 1}
        w = {(offset_i+0, offset_k): 1}
        terms.append((u, v, w))
        
        # Term 2: A01 * B0
        u = {(offset_i+0, offset_j+1): 1}
        v = {(offset_k+0, offset_k): 1}
        w = {(offset_i+0, offset_k): 1}
        terms.append((u, v, w))
        
        # Term 3: A10 * B0
        u = {(offset_i+1, offset_j+0): 1}
        v = {(offset_k+0, offset_k): 1}
        w = {(offset_i+1, offset_k): 1}
        terms.append((u, v, w))
        
        # Term 4: A11 * B0
        u = {(offset_i+1, offset_j+1): 1}
        v = {(offset_k+0, offset_k): 1}
        w = {(offset_i+1, offset_k): 1}
        terms.append((u, v, w))
        
        # Term 5: A00 * B1
        u = {(offset_i+0, offset_j+0): 1}
        v = {(offset_k+1, offset_k): 1}
        w = {(offset_i+0, offset_k): 1}
        terms.append((u, v, w))
        
        # Term 6: A01 * B1
        u = {(offset_i+0, offset_j+1): 1}
        v = {(offset_k+1, offset_k): 1}
        w = {(offset_i+0, offset_k): 1}
        terms.append((u, v, w))
        
        # Term 7: A10 * B1
        u = {(offset_i+1, offset_j+0): 1}
        v = {(offset_k+1, offset_k): 1}
        w = {(offset_i+1, offset_k): 1}
        terms.append((u, v, w))
        
        # Term 8: A11 * B1
        u = {(offset_i+1, offset_j+1): 1}
        v = {(offset_k+1, offset_k): 1}
        w = {(offset_i+1, offset_k): 1}
        terms.append((u, v, w))
        
        return terms

    # 1. A_L (cols 0,1) * B_T1 (rows 0,1, cols 0,1)
    # offset_i=0, offset_j=0, offset_k=0
    for term in strassen_terms(0, 0, 0):
        add_term(*term)
        
    # 2. A_L (cols 0,1) * B_T2 (rows 0,1, cols 2,3)
    # offset_i=0, offset_j=0, offset_k=2
    for term in strassen_terms(0, 0, 2):
        add_term(*term)
        
    # 3. A_L (cols 0,1) * B_T3 (rows 0,1, col 4)
    # offset_i=0, offset_j=0, offset_k=4
    for term in standard_2x2x1_terms(0, 0, 4):
        add_term(*term)
        
    # 4. A_R (cols 2,3) * B_B1 (rows 2,3, cols 0,1)
    # offset_i=0, offset_j=2, offset_k=0
    for term in strassen_terms(0, 2, 0):
        add_term(*term)
        
    # 5. A_R (cols 2,3) * B_B2 (rows 2,3, cols 2,3)
    # offset_i=0, offset_j=2, offset_k=2
    for term in strassen_terms(0, 2, 2):
        add_term(*term)
        
    # 6. A_R (cols 2,3) * B_B3 (rows 2,3, col 4)
    # offset_i=0, offset_j=2, offset_k=4
    for term in standard_2x2x1_terms(0, 2, 4):
        add_term(*term)

    # Check if col == rank
    # It should be 7*4 + 4*2 = 28 + 8 = 36.
    
    return {"U": U, "V": V, "W": W}

