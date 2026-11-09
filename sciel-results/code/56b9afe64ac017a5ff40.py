import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # We will build a rank 51 decomposition.
    # Structure: 3 blocks of 2x3x3.
    # Each 2x3x3 block has rank 17.
    
    rank_total = 51
    U = np.zeros((n*m, rank_total), dtype=int)
    V = np.zeros((m*p, rank_total), dtype=int)
    W = np.zeros((p*n, rank_total), dtype=int)
    
    # Helper to add a term
    # u_idx: index in U (0..n*m-1)
    # v_idx: index in V (0..m*p-1)
    # w_idx: index in W (0..p*n-1)
    # col: column index in the factor matrices
    
    def add_term(u_idx, v_idx, w_idx, col, val=1):
        U[u_idx, col] += val
        V[v_idx, col] += val
        W[w_idx, col] += val

    # Indices for 2x3x3 block (local i, j, k)
    # i in 0..1, j in 0..2, k in 0..2
    # U index: i*m + j = i*3 + j
    # V index: j*p_block + k = j*3 + k  (p_block=3)
    # W index: k*n + i = k*2 + i
    
    col = 0
    
    # Repeat for 3 blocks (k_global = 0..8)
    for block in range(3):
        # Offset for V and W indices
        # V global index: j*9 + (block*3 + k_local)
        # W global index: (block*3 + k_local)*2 + i
        
        # Part 1: 2x2x1 (A_L * B_L1) -> Rank 4
        # A_L is rows 0,1 of A (cols 0,1)
        # B_L1 is rows 0,1 of B (col 0)
        # C_L1 is rows 0,1 of C (col 0)
        # Terms:
        # u1 v1 -> w11
        # u1 v2 -> w12
        # u2 v1 -> w21
        # u2 v2 -> w22
        
        # Local indices for 2x2x1
        # i in 0..1, j in 0..1, k in 0..0 (k=0)
        
        # Term 1: i=0, j=0, k=0
        u_idx = 0*3 + 0
        v_idx = 0*9 + (block*3 + 0)
        w_idx = (block*3 + 0)*2 + 0
        add_term(u_idx, v_idx, w_idx, col); col += 1
        
        # Term 2: i=0, j=1, k=0
        u_idx = 0*3 + 1
        v_idx = 1*9 + (block*3 + 0)
        w_idx = (block*3 + 0)*2 + 0
        add_term(u_idx, v_idx, w_idx, col); col += 1
        
        # Term 3: i=1, j=0, k=0
        u_idx = 1*3 + 0
        v_idx = 0*9 + (block*3 + 0)
        w_idx = (block*3 + 0)*2 + 1
        add_term(u_idx, v_idx, w_idx, col); col += 1
        
        # Term 4: i=1, j=1, k=0
        u_idx = 1*3 + 1
        v_idx = 1*9 + (block*3 + 0)
        w_idx = (block*3 + 0)*2 + 1
        add_term(u_idx, v_idx, w_idx, col); col += 1

        # Part 2: 2x2x2 (A_L * B_L2) -> Rank 7 (Strassen)
        # A_L is 2x2. B_L2 is 2x2 (cols 1,2 of B).
        # C_L2 is 2x2 (cols 1,2 of C).
        
        # Strassen for 2x2x2:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 - M4 + M6 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5 + M6
        
        # We need to map these to U, V, W.
        # U is A_L. V is B_L2. W is C_L2.
        
        # Local indices for 2x2x2 part of the block
        # i in 0..1, j in 0..1 (local j for A_L, which is global j 0,1), k in 0..1 (local k for B_L2, which is global k 1,2)
        
        # M1: (A00+A11)(B00+B11)
        # U: A00 (0,0), A11 (1,1)
        # V: B00 (0,1), B11 (1,2) -> Note: B_L2 cols are 1,2. So B00 is (0,1), B11 is (1,2).
        # W: C00 (0,1), C11 (1,2)
        
        # Term M1
        # U: 0,0 and 1,1
        # V: 0,1 and 1,2
        # W: 0,1 and 1,2
        # This is a sum of 4 terms in the expansion, but Strassen combines them into 1 multiplication.
        # In the tensor decomposition, each "multiplication" M is one term in the rank sum.
        # So we add 1 term for M1.
        # U vector: e_00 + e_11
        # V vector: e_01 + e_12
        # W vector: e_01 + e_12
        
        # U indices: 0*3+0=0, 1*3+1=4
        # V indices: 0*9+(block*3+1), 1*9+(block*3+2)
        # W indices: (block*3+1)*2+0, (block*3+2)*2+1
        
        U[0, col] += 1
        U[4, col] += 1
        V[0*9 + block*3 + 1, col] += 1
        V[1*9 + block*3 + 2, col] += 1
        W[(block*3+1)*2 + 0, col] += 1
        W[(block*3+2)*2 + 1, col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        # U: 1,0 and 1,1 -> 3, 4
        # V: 0,1 -> 0*9+block*3+1
        # W: 1,0 -> (block*3+1)*2+1
        
        U[3, col] += 1
        U[4, col] += 1
        V[0*9 + block*3 + 1, col] += 1
        W[(block*3+1)*2 + 1, col] += 1
        col += 1
        
        # M3: A00(B01-B11)
        # U: 0,0 -> 0
        # V: 0,2 and 1,2 -> 0*9+block*3+2, 1*9+block*3+2 (with -1 for B11)
        # V: B01 is (0,2), B11 is (1,2).
        # W: 0,1 -> (block*3+1)*2+0
        
        U[0, col] += 1
        V[0*9 + block*3 + 2, col] += 1
        V[1*9 + block*3 + 2, col] -= 1
        W[(block*3+1)*2 + 0, col] += 1
        col += 1
        
        # M4: A11(B10-B00)
        # U: 1,1 -> 4
        # V: 1,1 and 0,1 -> 1*9+block*3+1, 0*9+block*3+1 (with -1 for B00)
        # W: 1,0 -> (block*3+1)*2+1
        
        U[4, col] += 1
        V[1*9 + block*3 + 1, col] += 1
        V[0*9 + block*3 + 1, col] -= 1
        W[(block*3+1)*2 + 1, col] += 1
        col += 1
        
        # M5: (A00+A01)B11
        # U: 0,0 and 0,1 -> 0, 1
        # V: 1,2 -> 1*9+block*3+2
        # W: 0,1 -> (block*3+1)*2+0
        
        U[0, col] += 1
        U[1, col] += 1
        V[1*9 + block*3 + 2, col] += 1
        W[(block*3+1)*2 + 0, col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        # U: 1,0 and 0,0 -> 3, 0 (with -1 for A00)
        # V: 0,1 and 0,2 -> 0*9+block*3+1, 0*9+block*3+2
        # W: 1,0 -> (block*3+1)*2+1
        
        U[3, col] += 1
        U[0, col] -= 1
        V[0*9 + block*3 + 1, col] += 1
        V[0*9 + block*3 + 2, col] += 1
        W[(block*3+1)*2 + 1, col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        # U: 0,1 and 1,1 -> 1, 4 (with -1 for A11)
        # V: 1,1 and 1,2 -> 1*9+block*3+1, 1*9+block*3+2
        # W: 1,0 -> (block*3+1)*2+1
        
        U[1, col] += 1
        U[4, col] -= 1
        V[1*9 + block*3 + 1, col] += 1
        V[1*9 + block*3 + 2, col] += 1
        W[(block*3+1)*2 + 1, col] += 1
        col += 1

        # Part 3: 2x1x3 (A_R * B_R) -> Rank 6
        # A_R is col 2 of A. B_R is row 2 of B.
        # C_R is col 2 of C.
        # A_R is 2x1. B_R is 1x3.
        # C_R is 2x3.
        # Terms:
        # u1 v1, u1 v2, u1 v3
        # u2 v1, u2 v2, u2 v3
        
        # U: A_R indices: 0*3+2=2, 1*3+2=5
        # V: B_R indices: 2*9+block*3+0, 2*9+block*3+1, 2*9+block*3+2
        # W: C_R indices: (block*3+0)*2+0, (block*3+1)*2+0, (block*3+2)*2+0 for row 0
        #                    (block*3+0)*2+1, (block*3+1)*2+1, (block*3+2)*2+1 for row 1
        
        # Term 1: u1 v1
        U[2, col] += 1
        V[2*9 + block*3 + 0, col] += 1
        W[(block*3+0)*2 + 0, col] += 1
        col += 1
        
        # Term 2: u1 v2
        U[2, col] += 1
        V[2*9 + block*3 + 1, col] += 1
        W[(block*3+1)*2 + 0, col] += 1
        col += 1
        
        # Term 3: u1 v3
        U[2, col] += 1
        V[2*9 + block*3 + 2, col] += 1
        W[(block*3+2)*2 + 0, col] += 1
        col += 1
        
        # Term 4: u2 v1
        U[5, col] += 1
        V[2*9 + block*3 + 0, col] += 1
        W[(block*3+0)*2 + 1, col] += 1
        col += 1
        
        # Term 5: u2 v2
        U[5, col] += 1
        V[2*9 + block*3 + 1, col] += 1
        W[(block*3+1)*2 + 1, col] += 1
        col += 1
        
        # Term 6: u2 v3
        U[5, col] += 1
        V[2*9 + block*3 + 2, col] += 1
        W[(block*3+2)*2 + 1, col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
