import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Target Rank: 23
    # 2x2x7 (Rank 22) + 2x1x7 (Rank 1)
    
    rank = 23
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2x2(k_offset, p_offset):
        nonlocal col
        # k indices are local 0, 1. Global k = k_offset + local_k
        # p indices are local 0, 1. Global p = p_offset + local_p
        
        # M1
        # U: A00, A11 -> indices (0,0), (1,1) relative to k_offset
        # U_idx: (0*2+0) + k_offset*? No.
        # U is indexed by (i, k). i in 0..1, k in 0..1 (local).
        # Global U index: i * m + (k_offset + k_local)?
        # Wait, m is the total middle dimension (3).
        # In the 2x2x2 block, the middle dimension is 2.
        # But U is sized n*m = 2*3 = 6.
        # The indices for U correspond to the full A matrix.
        # So if we are using k_offset, the global k is k_offset + k_local.
        # U index = i * m + (k_offset + k_local).
        
        # Let's define a helper to get global indices
        def u_idx(i, k_loc):
            return i * m + (k_offset + k_loc)
            
        def v_idx(k_loc, p_loc):
            return (k_offset + k_loc) * p + (p_offset + p_loc)
            
        def w_idx(p_loc, i):
            return (p_offset + p_loc) * n + i

        # Strassen Terms for 2x2x2
        # M1: (A00+A11)(B00+B11) -> C00, C11
        U[u_idx(0,0)][col] += 1
        U[u_idx(1,1)][col] += 1
        V[v_idx(0,0)][col] += 1
        V[v_idx(1,1)][col] += 1
        W[w_idx(0,0)][col] += 1
        W[w_idx(1,1)][col] += 1
        col += 1
        
        # M2: (A10+A11)B00 -> C10, -C11
        U[u_idx(1,0)][col] += 1
        U[u_idx(1,1)][col] += 1
        V[v_idx(0,0)][col] += 1
        W[w_idx(0,1)][col] += 1
        W[w_idx(1,1)][col] += -1
        col += 1
        
        # M3: A00(B01-B11) -> C01, C11
        U[u_idx(0,0)][col] += 1
        V[v_idx(0,1)][col] += 1
        V[v_idx(1,1)][col] += -1
        W[w_idx(1,0)][col] += 1
        W[w_idx(1,1)][col] += 1
        col += 1
        
        # M4: A11(B10-B00) -> C00, C10
        U[u_idx(1,1)][col] += 1
        V[v_idx(1,0)][col] += 1
        V[v_idx(0,0)][col] += -1
        W[w_idx(0,0)][col] += 1
        W[w_idx(0,1)][col] += 1
        col += 1
        
        # M5: (A00+A01)B11 -> -C00, C01
        U[u_idx(0,0)][col] += 1
        U[u_idx(0,1)][col] += 1
        V[v_idx(1,1)][col] += 1
        W[w_idx(0,0)][col] += -1
        W[w_idx(1,0)][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11
        U[u_idx(1,0)][col] += 1
        U[u_idx(0,0)][col] += -1
        V[v_idx(0,0)][col] += 1
        V[v_idx(0,1)][col] += 1
        W[w_idx(1,1)][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00
        U[u_idx(0,1)][col] += 1
        U[u_idx(1,1)][col] += -1
        V[v_idx(1,0)][col] += 1
        V[v_idx(1,1)][col] += 1
        W[w_idx(0,0)][col] += 1
        col += 1

    def add_trivial_2x2x1(k_offset, p_offset):
        nonlocal col
        # Rank 1
        # U: All 1s for i in 0,1 and k in 0,1
        for i in range(2):
            for k_loc in range(2):
                U[i * m + (k_offset + k_loc)][col] += 1
        
        # V: All 1s for k in 0,1 and p_loc=0
        for k_loc in range(2):
            V[(k_offset + k_loc) * p + (p_offset + 0)][col] += 1
            
        # W: All 1s for p_loc=0 and i in 0,1
        for i in range(2):
            W[(p_offset + 0) * n + i][col] += 1
            
        col += 1

    def add_trivial_2x1x7(k_offset, p_offset):
        nonlocal col
        # Rank 1
        # n=2, m=1 (local), p=7 (local)
        # U: i in 0,1, k_loc=0
        for i in range(2):
            U[i * m + (k_offset + 0)][col] += 1
            
        # V: k_loc=0, p_loc in 0..6
        for p_loc in range(7):
            V[(k_offset + 0) * p + (p_offset + p_loc)][col] += 1
            
        # W: p_loc in 0..6, i in 0,1
        for p_loc in range(7):
            for i in range(2):
                W[(p_offset + p_loc) * n + i][col] += 1
                
        col += 1

    # Construct 2x2x7 part (k_offset=0)
    # p_offset 0, 2, 4 for Strassen
    add_strassen_2x2x2(0, 0)
    add_strassen_2x2x2(0, 2)
    add_strassen_2x2x2(0, 4)
    # p_offset 6 for Trivial
    add_trivial_2x2x1(0, 6)
    
    # Construct 2x1x7 part (k_offset=2)
    # p_offset 0
    add_trivial_2x1x7(2, 0)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
