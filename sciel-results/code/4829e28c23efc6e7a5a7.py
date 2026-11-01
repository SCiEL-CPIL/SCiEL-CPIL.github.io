"""Matrix multiplication 4x4x4 tensor decomposition using recursive Strassen (rank 49)."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    
    Convention:
      C[i,j] = sum_k A[i,k] * B[k,j]
      U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] summed over r gives the tensor.
      Here n=m=p=4, so:
        U index: i*4 + k  (A row i, A col k)
        V index: k*4 + j  (B row k, B col j)
        W index: j*4 + i  (C col j, C row i)
    
    Uses a precomputed lookup table of the 49 non-zero tensor coordinates
    derived from the 7x7 Kronecker product structure of 2x2 Strassen.
    """
    t_start = time.time()
    deadline = 110.0  # safety margin before 150s kill
    
    n = m = p = 4
    rank = 49
    
    # Precompute the 49 terms as (U_entries, V_entries, W_entries)
    # Each entry is a list of (flat_index, coefficient)
    # U index: a_row * 4 + a_col
    # V index: b_row * 4 + b_col  
    # W index: c_col * 4 + c_row
    
    # Helper to compute flat indices
    def ui(r, c): return r * 4 + c
    def vi(r, c): return r * 4 + c
    def wi(r, c): return c * 4 + r  # W[j*n+i] = col*4+row
    
    # Outer Strassen: 7 terms, each with L and R block selections
    # L[t] = list of (block_row, block_col, coeff) for A
    # R[t] = list of (block_row, block_col, coeff) for B
    # C[t] = list of (block_row, block_col, coeff) for C contribution
    
    # Each block is 2x2, so block (br, bc) contains elements at
    # rows 2*br+0, 2*br+1 and cols 2*bc+0, 2*bc+1
    
    outer_L = [
        [(0,0,1), (1,1,1)],       # M1: A11+A22
        [(1,0,1), (1,1,1)],       # M2: A21+A22
        [(0,0,1)],                # M3: A11
        [(1,1,1)],                # M4: A22
        [(0,0,1), (0,1,1)],       # M5: A11+A12
        [(1,0,1), (0,0,-1)],      # M6: A21-A11
        [(0,1,1), (1,1,-1)],      # M7: A12-A22
    ]
    outer_R = [
        [(0,0,1), (1,1,1)],       # B11+B22
        [(0,0,1)],                # B11
        [(0,1,1), (1,1,-1)],      # B12-B22
        [(1,0,1), (0,0,-1)],      # B21-B11
        [(1,1,1)],                # B22
        [(0,0,1), (0,1,1)],       # B11+B12
        [(1,0,1), (1,1,1)],       # B21+B22
    ]
    outer_C = [
        [(0,0,1), (1,1,1)],       # C11+C22
        [(1,0,1), (1,1,-1)],      # C21-C22
        [(0,1,1), (1,1,1)],       # C12+C22
        [(0,0,1), (1,0,1)],       # C11+C21
        [(0,0,-1), (0,1,1)],      # -C11+C12
        [(1,1,1)],                # C22
        [(0,0,1)],                # C11
    ]
    
    # Inner Strassen: 7 terms for 2x2 multiplication
    # Each: (l_sel, r_sel, p_sel) where each is list of (row, col, coeff)
    inner_L = [
        [(0,0,1), (1,1,1)],       # s1: L00+L11
        [(1,0,1), (1,1,1)],       # s2: L10+L11
        [(0,0,1)],                # s3: L00
        [(1,1,1)],                # s4: L11
        [(0,0,1), (0,1,1)],       # s5: L00+L01
        [(1,0,1), (0,0,-1)],      # s6: L10-L00
        [(0,1,1), (1,1,-1)],      # s7: L01-L11
    ]
    inner_R = [
        [(0,0,1), (1,1,1)],       # R00+R11
        [(0,0,1)],                # R00
        [(0,1,1), (1,1,-1)],      # R01-R11
        [(1,0,1), (0,0,-1)],      # R10-R00
        [(1,1,1)],                # R11
        [(0,0,1), (0,1,1)],       # R00+R01
        [(1,0,1), (1,1,1)],       # R10+R11
    ]
    inner_P = [
        [(0,0,1), (1,1,1)],       # P00+P11
        [(1,0,1), (1,1,-1)],      # P10-P11
        [(0,1,1), (1,1,1)],       # P01+P11
        [(0,0,1), (1,0,1)],       # P00+P10
        [(0,0,-1), (0,1,1)],      # -P00+P01
        [(1,1,1)],                # P11
        [(0,0,1)],                # P00
    ]
    
    # Build the 49-term lookup table
    U_table = [None] * rank
    V_table = [None] * rank
    W_table = [None] * rank
    
    for t in range(7):
        for k in range(7):
            r = t * 7 + k
            
            # U entries: for each outer L block and inner L position
            u_entries = []
            for (obr, obc, oc) in outer_L[t]:
                for (ibr, ibc, ic) in inner_L[k]:
                    a_row = 2 * obr + ibr
                    a_col = 2 * obc + ibc
                    u_entries.append((ui(a_row, a_col), oc * ic))
            U_table[r] = u_entries
            
            # V entries: for each outer R block and inner R position
            v_entries = []
            for (obr, obc, oc) in outer_R[t]:
                for (ibr, ibc, ic) in inner_R[k]:
                    b_row = 2 * obr + ibr
                    b_col = 2 * obc + ibc
                    v_entries.append((vi(b_row, b_col), oc * ic))
            V_table[r] = v_entries
            
            # W entries: for each outer C block and inner P position
            w_entries = []
            for (obr, obc, oc) in outer_C[t]:
                for (ipr, ipc, ic) in inner_P[k]:
                    c_row = 2 * obr + ipr
                    c_col = 2 * obc + ipc
                    w_entries.append((wi(c_row, c_col), oc * ic))
            W_table[r] = w_entries
    
    # Build matrices from lookup table
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    for r in range(rank):
        if time.time() - t_start > deadline:
            break
        for (idx, coeff) in U_table[r]:
            U[idx, r] += coeff
        for (idx, coeff) in V_table[r]:
            V[idx, r] += coeff
        for (idx, coeff) in W_table[r]:
            W[idx, r] += coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
