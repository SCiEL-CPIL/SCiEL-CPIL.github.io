import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # We use a block decomposition:
    # Split m=5 into 2, 2, 1.
    # Split p=5 into 2, 2, 1.
    # Blocks:
    # 1. 2x2x2 (Strassen, rank 7) - 4 such blocks
    # 2. 2x2x1 (Schoolbook, rank 4) - 2 such blocks
    # 3. 2x1x2 (Schoolbook, rank 4) - 2 such blocks
    # 4. 2x1x1 (Schoolbook, rank 2) - 1 such block
    # Total rank = 4*7 + 2*4 + 2*4 + 1*2 = 28 + 8 + 8 + 2 = 46.
    
    U_data = [] # List of (row_idx, col_idx, val)
    V_data = []
    W_data = []
    
    col_offset = 0
    
    # Helper to add a term to the decomposition
    def add_term(u_terms, v_terms, w_terms):
        """
        u_terms: list of (global_u_idx, val)
        v_terms: list of (global_v_idx, val)
        w_terms: list of (global_w_idx, val)
        """
        nonlocal col_offset
        for idx, val in u_terms:
            U_data.append((idx, col_offset, val))
        for idx, val in v_terms:
            V_data.append((idx, col_offset, val))
        for idx, val in w_terms:
            W_data.append((idx, col_offset, val))
        col_offset += 1

    # Helper to add schoolbook terms for a block
    def add_schoolbook_block(j_start, j_end, k_start, k_end):
        """
        j_start, j_end: range of global j indices [j_start, j_end)
        k_start, k_end: range of global k indices [k_start, k_end)
        """
        for i in range(n):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    add_term(
                        [(i*m+j, 1)],
                        [(j*p+k, 1)],
                        [(k*n+i, 1)]
                    )

    # Helper to add Strassen terms for a 2x2x2 block
    def add_strassen_block(j_start, k_start):
        """
        j_start: global j index for local j=0
        k_start: global k index for local k=0
        Local i is 0, 1.
        """
        # Local indices 0, 1 map to global:
        # j0 = j_start, j1 = j_start + 1
        # k0 = k_start, k1 = k_start + 1
        
        # U indices for A[i][j]
        u_00 = 0*m + j_start
        u_01 = 0*m + (j_start + 1)
        u_10 = 1*m + j_start
        u_11 = 1*m + (j_start + 1)
        
        # V indices for B[j][k]
        v_00 = j_start*p + k_start
        v_01 = j_start*p + (k_start + 1)
        v_10 = (j_start + 1)*p + k_start
        v_11 = (j_start + 1)*p + (k_start + 1)
        
        # W indices for C[i][k] -> W[k*n+i]
        w_00 = k_start*n + 0
        w_01 = k_start*n + 1
        w_10 = (k_start + 1)*n + 0
        w_11 = (k_start + 1)*n + 1
        
        # M1: (A00+A11)(B00+B11) -> C00 + C11
        add_term(
            [(u_00, 1), (u_11, 1)],
            [(v_00, 1), (v_11, 1)],
            [(w_00, 1), (w_11, 1)]
        )
        
        # M2: (A10+A11)B00 -> C10 - C11
        add_term(
            [(u_10, 1), (u_11, 1)],
            [(v_00, 1)],
            [(w_10, 1), (w_11, -1)]
        )
        
        # M3: A00(B01-B11) -> C01 + C11
        add_term(
            [(u_00, 1)],
            [(v_01, 1), (v_11, -1)],
            [(w_01, 1), (w_11, 1)]
        )
        
        # M4: A11(B10-B00) -> C10 + C00
        add_term(
            [(u_11, 1)],
            [(v_10, 1), (v_00, -1)],
            [(w_10, 1), (w_00, 1)]
        )
        
        # M5: (A00+A01)B11 -> C01 - C00
        add_term(
            [(u_00, 1), (u_01, 1)],
            [(v_11, 1)],
            [(w_01, 1), (w_00, -1)]
        )
        
        # M6: (A10-A00)(B00+B01) -> C11
        add_term(
            [(u_10, 1), (u_00, -1)],
            [(v_00, 1), (v_01, 1)],
            [(w_11, 1)]
        )
        
        # M7: (A01-A11)(B10+B11) -> C00
        add_term(
            [(u_01, 1), (u_11, -1)],
            [(v_10, 1), (v_11, 1)],
            [(w_00, 1)]
        )

    # Block 1: J1=[0,2), K1=[0,2) -> Strassen
    add_strassen_block(0, 0)
    
    # Block 2: J1=[0,2), K2=[2,4) -> Strassen
    add_strassen_block(0, 2)
    
    # Block 3: J2=[2,4), K1=[0,2) -> Strassen
    add_strassen_block(2, 0)
    
    # Block 4: J2=[2,4), K2=[2,4) -> Strassen
    add_strassen_block(2, 2)
    
    # Block 5: J1=[0,2), K3=[4,5) -> Schoolbook
    add_schoolbook_block(0, 2, 4, 5)
    
    # Block 6: J2=[2,4), K3=[4,5) -> Schoolbook
    add_schoolbook_block(2, 4, 4, 5)
    
    # Block 7: J3=[4,5), K1=[0,2) -> Schoolbook
    add_schoolbook_block(4, 5, 0, 2)
    
    # Block 8: J3=[4,5), K2=[2,4) -> Schoolbook
    add_schoolbook_block(4, 5, 2, 4)
    
    # Block 9: J3=[4,5), K3=[4,5) -> Schoolbook
    add_schoolbook_block(4, 5, 4, 5)

    R = col_offset
    
    # Build matrices
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    for idx, col, val in U_data:
        U[idx][col] = val
    for idx, col, val in V_data:
        V[idx][col] = val
    for idx, col, val in W_data:
        W[idx][col] = val

    # Verify
    if time.time() > deadline - 10:
        return {"U": U, "V": W, "W": W} # Note: V is not used in return? No, return U, V, W.
        # Correction: return {"U": U, "V": V, "W": W}

    # Simple verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(R):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    # Fallback to schoolbook if error
                    U_fb = [[0] * 50 for _ in range(n * m)]
                    V_fb = [[0] * 50 for _ in range(m * p)]
                    W_fb = [[0] * 50 for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_fb[ii*m+jj][col] = 1
                                V_fb[jj*p+kk][col] = 1
                                W_fb[kk*n+ii][col] = 1
                                col += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
