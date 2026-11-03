import time
import sys

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 34 solution
    # 17 terms for k in {0,1}
    # 17 terms for k in {2,3}
    
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        U.append(u_idx)
        V.append(v_idx)
        W.append(w_idx)
        # Store coefficients? No, the interface expects matrices.
        # The task says "factor matrices of shapes ... with integer/half-integer entries".
        # So U, V, W are 2D lists.
        pass

    # Let's build the matrices directly.
    # Rank = 34
    rank = 34
    U_mat = [[0]*rank for _ in range(n*m)]
    V_mat = [[0]*rank for _ in range(m*p)]
    W_mat = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen_block(i_offset, j_offset, k_offset, col_start):
        # i_offset, j_offset, k_offset are 0
        # A indices: i_offset..i_offset+1, j_offset..j_offset+1
        # B indices: j_offset..j_offset+1, k_offset..k_offset+1
        # C indices: i_offset..i_offset+1, k_offset..k_offset+1
        
        # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        # We need to add 7 terms.
        
        # Term 1: (a+d)(e+h)
        # U: a, d
        # V: e, h
        # W: c00, c11 (with signs)
        
        # This is getting complicated to hardcode manually in the thought process.
        # I will generate the code to build it.
        
        pass

    # I will write the full code in the final block.
    return {"U": U_mat, "V": V_mat, "W": W_mat}
