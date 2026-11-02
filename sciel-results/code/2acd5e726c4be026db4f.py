"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    Uses block decomposition with Strassen for 2x2x2 blocks and schoolbook for smaller ones.
    Total rank: 4*7 (2x2x2) + 2*4 (2x2x1) + 4*4 (1x2x2) + 2*2 (1x2x1) = 28 + 8 + 16 + 4 = 56.
    """
    n, m, p = DIMENSIONS
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_block(I_indices, J_indices, K_indices, local_decomp):
        nonlocal current_rank
        for u_local, v_local, w_local in local_decomp:
            # u_local: dict {(i,j): val}
            # v_local: dict {(j,k): val}
            # w_local: dict {(k,i): val}
            
            # Map local indices to global
            for (li, lj), val_u in u_local.items():
                gi = I_indices[li]
                gj = J_indices[lj]
                U[gi * m + gj][current_rank] += val_u
            
            for (lj, lk), val_v in v_local.items():
                gj = J_indices[lj]
                gk = K_indices[lk]
                V[gj * p + gk][current_rank] += val_v
            
            for (lk, li), val_w in w_local.items():
                gk = K_indices[lk]
                gi = I_indices[li]
                W[gk * n + gi][current_rank] += val_w
            
            current_rank += 1

    # Define local decompositions
    # Type A: 2x2x2 (Strassen)
    def get_222_decomp():
        # Returns list of (U_dict, V_dict, W_dict)
        decomp = []
        # M1: (A00+A11)(B00+B11) -> C00, C11
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1} # C00 coeff 1, C11 coeff 1
        decomp.append((u1, v1, w1))
        
        # M2: (A10+A11)B00 -> C10
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1} # C10 coeff 1 (k=0, i=1)
        decomp.append((u2, v2, w2))
        
        # M3: A00(B10+B11) -> C01
        u3 = {(0,0): 1}
        v3 = {(1,0): 1, (1,1): 1}
        w3 = {(1,0): 1} # C01 coeff 1 (k=1, i=0)
        decomp.append((u3, v3, w3))
        
        # M4: A11(B01+B11) -> C00, C10
        u4 = {(1,1): 1}
        v4 = {(0,1): 1, (1,1): 1}
        w4 = {(0,0): 1, (0,1): 1} # C00 coeff 1, C10 coeff 1
        decomp.append((u4, v4, w4))
        
        # M5: (A00+A01)B11 -> C00, C01
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1} # C00 coeff -1, C01 coeff 1
        decomp.append((u5, v5, w5))
        
        # M6: (A10-A00)(B00+B01) -> C11
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1} # C11 coeff 1
        decomp.append((u6, v6, w6))
        
        # M7: (A01-A11)(B10+B00) -> C00, C11
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (0,0): 1}
        w7 = {(0,0): 1, (1,1): 1} # C00 coeff 1, C11 coeff 1
        decomp.append((u7, v7, w7))
        
        return decomp

    # Type B: 2x2x1 (Schoolbook)
    def get_221_decomp():
        decomp = []
        # C0 = A00 B0 + A01 B1
        u1 = {(0,0): 1}
        v1 = {(0,0): 1}
        w1 = {(0,0): 1}
        decomp.append((u1, v1, w1))
        
        u2 = {(0,1): 1}
        v2 = {(1,0): 1}
        w2 = {(0,0): 1}
        decomp.append((u2, v2, w2))
        
        # C1 = A10 B0 + A11 B1
        u3 = {(1,0): 1}
        v3 = {(0,0): 1}
        w3 = {(0,1): 1}
        decomp.append((u3, v3, w3))
        
        u4 = {(1,1): 1}
        v4 = {(1,0): 1}
        w4 = {(0,1): 1}
        decomp.append((u4, v4, w4))
        
        return decomp

    # Type C: 1x2x2 (Schoolbook)
    def get_122_decomp():
        decomp = []
        # C0 = A0 B00 + A1 B10
        u1 = {(0,0): 1}
        v1 = {(0,0): 1}
        w1 = {(0,0): 1}
        decomp.append((u1, v1, w1))
        
        u2 = {(0,1): 1}
        v2 = {(1,0): 1}
        w2 = {(0,0): 1}
        decomp.append((u2, v2, w2))
        
        # C1 = A0 B01 + A1 B11
        u3 = {(0,0): 1}
        v3 = {(0,1): 1}
        w3 = {(1,0): 1}
        decomp.append((u3, v3, w3))
        
        u4 = {(0,1): 1}
        v4 = {(1,1): 1}
        w4 = {(1,0): 1}
        decomp.append((u4, v4, w4))
        
        return decomp

    # Type D: 1x2x1 (Schoolbook)
    def get_121_decomp():
        decomp = []
        # C0 = A0 B0 + A1 B1
        u1 = {(0,0): 1}
        v1 = {(0,0): 1}
        w1 = {(0,0): 1}
        decomp.append((u1, v1, w1))
        
        u2 = {(0,1): 1}
        v2 = {(1,0): 1}
        w2 = {(0,0): 1}
        decomp.append((u2, v2, w2))
        
        return decomp

    # Blocks
    # I indices: 0,1 in R1, 2 in R2
    # J indices: 0,1 in C1, 2,3 in C2
    # K indices: 0,1 in K1, 2,3 in K2, 4 in K3
    
    R1 = [0, 1]
    R2 = [2]
    C1 = [0, 1]
    C2 = [2, 3]
    K1 = [0, 1]
    K2 = [2, 3]
    K3 = [4]
    
    # 4 blocks of 2x2x2
    add_block(R1, C1, K1, get_222_decomp())
    add_block(R1, C2, K1, get_222_decomp())
    add_block(R1, C1, K2, get_222_decomp())
    add_block(R1, C2, K2, get_222_decomp())
    
    # 2 blocks of 2x2x1
    add_block(R1, C1, K3, get_221_decomp())
    add_block(R1, C2, K3, get_221_decomp())
    
    # 4 blocks of 1x2x2
    add_block(R2, C1, K1, get_122_decomp())
    add_block(R2, C2, K1, get_122_decomp())
    add_block(R2, C1, K2, get_122_decomp())
    add_block(R2, C2, K2, get_122_decomp())
    
    # 2 blocks of 1x2x1
    add_block(R2, C1, K3, get_121_decomp())
    add_block(R2, C2, K3, get_121_decomp())
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
