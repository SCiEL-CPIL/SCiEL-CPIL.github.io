"""Rewritten program: 98-rank 4x4x8 via recursive Strassen (rank 49 for 4x4x4)."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _get_strassen_2x2_terms():
    """
    Return the 7 terms of Strassen's 2x2x2 decomposition.
    Each term is a tuple (u, v, w) where u, v, w are lists of length 4.
    Indices for 2x2 matrix:
    0: (0,0), 1: (0,1)
    2: (1,0), 3: (1,1)
    """
    terms = []
    
    # M1 = (a00+a11)(b00+b11)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1]
    terms.append((u1, v1, w1))
    
    # M2 = (a10+a11)b00
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, -1]
    terms.append((u2, v2, w2))
    
    # M3 = a00(b01-b11)
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, 1]
    terms.append((u3, v3, w3))
    
    # M4 = a11(b10-b00)
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1]
    w4 = [1, 0, 1, 0]
    terms.append((u4, v4, w4))
    
    # M5 = (a00+a01)b11
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0]
    terms.append((u5, v5, w5))
    
    # M6 = (a10-a00)(b00+b01)
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1]
    terms.append((u6, v6, w6))
    
    # M7 = (a01-a11)(b10+b11)
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 0]
    terms.append((u7, v7, w7))
    
    return terms

def _build_49_rank_4x4x4():
    """
    Build rank 49 decomposition for 4x4x4 matrix multiplication.
    Uses recursive Strassen.
    Returns U (16x49), V (16x49), W (16x49).
    """
    U = np.zeros((16, 49), dtype=np.float64)
    V = np.zeros((16, 49), dtype=np.float64)
    W = np.zeros((16, 49), dtype=np.float64)
    
    terms_2x2 = _get_strassen_2x2_terms()
    
    # Strassen block coefficients for 4x4 from 2x2 blocks
    # Blocks are indexed 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # A indices in 4x4:
    # Block 0: 0,1,4,5
    # Block 1: 2,3,6,7
    # Block 2: 8,9,12,13
    # Block 3: 10,11,14,15
    
    # U_k = sum alpha_k[b] * A_block_b
    # V_k = sum beta_k[b] * B_block_b
    # C_block_i = sum gamma_k[i] * M_k
    
    # M1
    U_coeff = [[1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]]
    V_coeff = [[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
    # C = M1 - M4 + M6 for C00, M5+M7 for C01, M2+M4 for C10, M1-M3+M7 for C11
    W_coeff = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] # Identity for M1
    W_coeff[0] = [1, 0, 0, 0] # C00 gets M1
    W_coeff[1] = [0, 1, 0, 0] # C01 gets M1? No.
    
    # Let's do this more systematically.
    # For each k in 0..6 (M_k), we have contributions to C blocks.
    # gamma[k][c_block]
    gamma = np.zeros((7, 4))
    # C00 = M1 - M4 + M6
    gamma[0, 0] += 1
    gamma[3, 0] += -1
    gamma[5, 0] += 1
    # C01 = M5 + M7
    gamma[4, 1] += 1
    gamma[6, 1] += 1
    # C10 = M2 + M4
    gamma[1, 2] += 1
    gamma[3, 2] += 1
    # C11 = M1 - M3 + M7
    gamma[0, 3] += 1
    gamma[2, 3] += -1
    gamma[6, 3] += 1
    
    # U_coeff[k][b] is coefficient of A_block_b in U_k
    # V_coeff[k][b] is coefficient of B_block_b in V_k
    # From Strassen definitions:
    # M1: U=(1,0,0,1), V=(1,0,0,1)
    # M2: U=(0,0,1,1), V=(1,0,0,0)
    # M3: U=(1,0,0,0), V=(0,1,0,-1)
    # M4: U=(0,0,0,1), V=(0,0,1,-1)
    # M5: U=(1,1,0,0), V=(0,0,0,1)
    # M6: U=(-1,0,1,0), V=(1,1,0,0)
    # M7: U=(0,1,0,-1), V=(0,0,1,1)
    
    # Note: The U/V coefficients above are for the 2x2 block indices 0,1,2,3.
    # We need to map these to the 16-dimensional vectors.
    
    col = 0
    for k in range(7):
        u_k_blocks = U_coeff[k] # length 4
        v_k_blocks = V_coeff[k] # length 4
        w_k_blocks = gamma[k]   # length 4
        
        # For each of the 7 sub-terms in 2x2x2
        for t_idx in range(7):
            u_t, v_t, w_t = terms_2x2[t_idx]
            
            u_full = np.zeros(16)
            v_full = np.zeros(16)
            w_full = np.zeros(16)
            
            # Expand u_full
            for b in range(4):
                if u_k_blocks[b] != 0:
                    # Block b indices in 4x4
                    # b=0: 0,1,4,5
                    # b=1: 2,3,6,7
                    # b=2: 8,9,12,13
                    # b=3: 10,11,14,15
                    # Local index l in u_t (0..3) corresponds to:
                    # l=0 (0,0), l=1 (0,1), l=2 (1,0), l=3 (1,1)
                    # Global indices for block b:
                    # b=0: 0->0, 1->1, 2->4, 3->5
                    # b=1: 0->2, 1->3, 2->6, 3->7
                    # b=2: 0->8, 1->9, 2->12, 3->13
                    # b=3: 0->10, 1->11, 2->14, 3->15
                    # Pattern: 
                    # Row r = l // 2, Col c = l % 2
                    # Global Row = (b // 2) * 2 + r
                    # Global Col = (b % 2) * 2 + c
                    # Global Index = Global Row * 4 + Global Col
                    for l in range(4):
                        if u_t[l] != 0:
                            r = l // 2
                            c = l % 2
                            gr = (b // 2) * 2 + r
                            gc = (b % 2) * 2 + c
                            gidx = gr * 4 + gc
                            u_full[gidx] += u_k_blocks[b] * u_t[l]
                            
            # Expand v_full similarly
            for b in range(4):
                if v_k_blocks[b] != 0:
                    for l in range(4):
                        if v_t[l] != 0:
                            r = l // 2
                            c = l % 2
                            gr = (b // 2) * 2 + r
                            gc = (b % 2) * 2 + c
                            gidx = gr * 4 + gc
                            v_full[gidx] += v_k_blocks[b] * v_t[l]
                            
            # Expand w_full
            for b in range(4):
                if w_k_blocks[b] != 0:
                    for l in range(4):
                        if w_t[l] != 0:
                            r = l // 2
                            c = l % 2
                            gr = (b // 2) * 2 + r
                            gc = (b % 2) * 2 + c
                            gidx = gr * 4 + gc
                            w_full[gidx] += w_k_blocks[b] * w_t[l]
                            
            U[:, col] = u_full
            V[:, col] = v_full
            W[:, col] = w_full
            col += 1
            
    return U, V, W

def _verify(U, V, W, T, tol=1e-9):
    r = U.shape[1]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS  # 4, 4, 8

    # Build the full 4x4x8 multiplication tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Schoolbook decomposition (rank 128)
    rank_schoolbook = n * m * p
    U = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
    V = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
    W = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1

    best_U, best_V, best_W = U, V, W
    best_rank = rank_schoolbook

    # Try 49-rank 4x4x4 block, doubled to 98 for 4x4x8
    if time.time() < deadline - 5:
        U4, V4, W4 = _build_49_rank_4x4x4()
        rank4 = U4.shape[1]

        if rank4 <= 49:
            U_full = np.zeros((n*m, 2*rank4), dtype=np.float64)
            V_full = np.zeros((m*p, 2*rank4), dtype=np.float64)
            W_full = np.zeros((p*n, 2*rank4), dtype=np.float64)

            # Block 1: k=0..3 (Left half of B and C)
            U_full[:, 0:rank4] = U4
            # Map V4 (16x49) to V_full (32x49) for B_left
            for r in range(rank4):
                for local_idx in range(16):
                    val = V4[local_idx, r]
                    if val != 0:
                        # local_idx corresponds to j*4 + k'
                        j = local_idx // 4
                        k_prime = local_idx % 4
                        global_idx = j * 8 + k_prime
                        V_full[global_idx, r] += val

            # Map W4 (16x49) to W_full (32x49) for C_left
            for r in range(rank4):
                for local_idx in range(16):
                    val = W4[local_idx, r]
                    if val != 0:
                        # local_idx corresponds to k'*4 + i
                        k_prime = local_idx // 4
                        i = local_idx % 4
                        global_idx = k_prime * 4 + i
                        W_full[global_idx, r] += val

            # Block 2: k=4..7 (Right half of B and C)
            U_full[:, rank4:2*rank4] = U4
            # Map V4 to V_full for B_right
            for r in range(rank4):
                for local_idx in range(16):
                    val = V4[local_idx, r]
                    if val != 0:
                        j = local_idx // 4
                        k_prime = local_idx % 4
                        global_idx = j * 8 + (k_prime + 4)
                        V_full[global_idx, r + rank4] += val

            # Map W4 to W_full for C_right
            for r in range(rank4):
                for local_idx in range(16):
                    val = W4[local_idx, r]
                    if val != 0:
                        k_prime = local_idx // 4
                        i = local_idx % 4
                        global_idx = (k_prime + 4) * 4 + i
                        W_full[global_idx, r + rank4] += val

            if _verify(U_full, V_full, W_full, T):
                best_U, best_V, best_W = U_full, V_full, W_full
                best_rank = 2*rank4

    # Convert to half-integer lists
    def to_half_integer(mat):
        mat2 = mat * 2
        rounded = np.round(mat2)
        if np.allclose(mat2, rounded, atol=1e-6):
            return (rounded / 2).tolist()
        rounded_int = np.round(mat)
        if np.allclose(mat, rounded_int, atol=1e-6):
            return rounded_int.astype(int).tolist()
        return mat.tolist()

    U_out = to_half_integer(best_U)
    V_out = to_half_integer(best_V)
    W_out = to_half_integer(best_W)

    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END
