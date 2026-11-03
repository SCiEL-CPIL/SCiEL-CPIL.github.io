"""
Strategy: Use Kronecker product to correctly build the rank-49 Strassen decomposition for 4x4x4,
then extend to 4x4x8 by duplicating the structure for the two independent 4x4x4 blocks.
This avoids the manual index mapping errors in the seed program.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _get_strassen_2x2_basis():
    """
    Returns U, V, W matrices for 2x2x2 Strassen decomposition.
    Shapes: (4, 7)
    """
    U = np.array([
        [1, 0, 0, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=np.float64)
    
    V = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ], dtype=np.float64)
    
    W = np.array([
        [1, 0, 0, 0, -1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, -1, 0, 0, 1]
    ], dtype=np.float64)
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Builds the rank-49 decomposition for 4x4x4 matrix multiplication.
    Uses Kronecker products of the 2x2 Strassen basis.
    Returns U (16x49), V (16x49), W (16x49).
    """
    U2, V2, W2 = _get_strassen_2x2_basis()
    
    # The 4x4x4 Strassen decomposition is formed by taking all 7x7=49 combinations
    # of the 2x2 basis terms, but we must be careful with the indices.
    # Actually, the standard recursive Strassen for 4x4 uses 7 calls to 2x2x2.
    # This results in 7 * 7 = 49 terms.
    
    U4 = np.zeros((16, 49), dtype=np.float64)
    V4 = np.zeros((16, 49), dtype=np.float64)
    W4 = np.zeros((16, 49), dtype=np.float64)
    
    # Indices for 4x4 blocks:
    # Block 0: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # Block 1: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # Block 2: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # Block 3: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # We iterate over the 7 outer Strassen terms (k_outer) and 7 inner terms (k_inner)
    # Total 49 terms.
    
    # Coefficients for the outer 2x2 block decomposition (same as 2x2 Strassen)
    # U_outer[k_outer, block_idx]
    # V_outer[k_outer, block_idx]
    # W_outer[k_outer, block_idx]
    
    # Note: U2, V2, W2 from _get_strassen_2x2_basis are (4, 7).
    # U2[i, k] is the coefficient for A_i in term k.
    # We need to map these to the 4 blocks of the 4x4 matrix.
    
    # Let's define the mapping from 2x2 local indices (0,1,2,3) to 4x4 global indices.
    # Local 0 -> (0,0), Local 1 -> (0,1), Local 2 -> (1,0), Local 3 -> (1,1)
    
    # For each outer term k_out (0..6):
    #   U_k_out is a 2x2 matrix of coefficients for the 4 blocks.
    #   Actually, U2 is (4,7). U2[i, k] gives coeff for A_i in term k.
    #   The 4 indices i=0..3 correspond to the 4 blocks of the 4x4 matrix?
    #   No, in the recursive step, we treat the 4x4 as a 2x2 of 2x2 blocks.
    #   So the "i" in U2[i, k] refers to the block index (0,1,2,3).
    
    # So, for a given outer term k_out:
    #   The coefficient for Block b is U2[b, k_out].
    #   Similarly for V2[b, k_out] and W2[b, k_out].
    
    # Inside each block, we use the inner 2x2 Strassen decomposition.
    # The inner term k_in (0..6) has coefficients U2[i_local, k_in] for local indices.
    
    # Total term index = k_out * 7 + k_in.
    
    for k_out in range(7):
        for k_in in range(7):
            col = k_out * 7 + k_in
            
            # Build U vector for this term
            u_vec = np.zeros(16)
            v_vec = np.zeros(16)
            w_vec = np.zeros(16)
            
            # Iterate over the 4 blocks
            for b in range(4):
                coeff_u = U2[b, k_out]
                coeff_v = V2[b, k_out]
                coeff_w = W2[b, k_out]
                
                if coeff_u == 0 and coeff_v == 0 and coeff_w == 0:
                    continue
                
                # Map block b to global indices
                # b=0: (0,0) block -> rows 0,1 cols 0,1
                # b=1: (0,1) block -> rows 0,1 cols 2,3
                # b=2: (1,0) block -> rows 2,3 cols 0,1
                # b=3: (1,1) block -> rows 2,3 cols 2,3
                
                br = (b // 2) * 2
                bc = (b % 2) * 2
                
                # Iterate over local indices in the 2x2 block
                for l in range(4):
                    lr = l // 2
                    lc = l % 2
                    
                    gr = br + lr
                    gc = bc + lc
                    gidx = gr * 4 + gc
                    
                    u_coeff = U2[l, k_in]
                    v_coeff = V2[l, k_in]
                    w_coeff = W2[l, k_in]
                    
                    u_vec[gidx] += coeff_u * u_coeff
                    v_vec[gidx] += coeff_v * v_coeff
                    w_vec[gidx] += coeff_w * w_coeff
            
            U4[:, col] = u_vec
            V4[:, col] = v_vec
            W4[:, col] = w_vec
            
    return U4, V4, W4

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

    # Schoolbook decomposition (rank 128) as fallback
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

    # Try 98-rank 4x4x8 via two 49-rank 4x4x4 blocks
    if time.time() < deadline - 5:
        U4, V4, W4 = _build_4x4x4_strassen()
        rank4 = U4.shape[1]
        
        if rank4 == 49:
            U_full = np.zeros((n*m, 2*rank4), dtype=np.float64)
            V_full = np.zeros((m*p, 2*rank4), dtype=np.float64)
            W_full = np.zeros((p*n, 2*rank4), dtype=np.float64)

            # Block 1: k=0..3 (Left half of B and C)
            U_full[:, 0:rank4] = U4
            
            # Map V4 (16x49) to V_full (32x49) for B_left (j*4+k)
            # V4 indices are j*4+k' where k' in 0..3
            # V_full indices are j*8+k where k in 0..3
            for r in range(rank4):
                for local_idx in range(16):
                    val = V4[local_idx, r]
                    if val != 0:
                        j = local_idx // 4
                        k_prime = local_idx % 4
                        global_idx = j * 8 + k_prime
                        V_full[global_idx, r] = val

            # Map W4 (16x49) to W_full (32x49) for C_left (k'*4+i)
            # W4 indices are k'*4+i where k' in 0..3
            # W_full indices are k*4+i where k in 0..3
            for r in range(rank4):
                for local_idx in range(16):
                    val = W4[local_idx, r]
                    if val != 0:
                        k_prime = local_idx // 4
                        i = local_idx % 4
                        global_idx = k_prime * 4 + i
                        W_full[global_idx, r] = val

            # Block 2: k=4..7 (Right half of B and C)
            U_full[:, rank4:2*rank4] = U4
            
            # Map V4 to V_full for B_right (j*8 + (k'+4))
            for r in range(rank4):
                for local_idx in range(16):
                    val = V4[local_idx, r]
                    if val != 0:
                        j = local_idx // 4
                        k_prime = local_idx % 4
                        global_idx = j * 8 + (k_prime + 4)
                        V_full[global_idx, r + rank4] = val

            # Map W4 to W_full for C_right ((k'+4)*4 + i)
            for r in range(rank4):
                for local_idx in range(16):
                    val = W4[local_idx, r]
                    if val != 0:
                        k_prime = local_idx // 4
                        i = local_idx % 4
                        global_idx = (k_prime + 4) * 4 + i
                        W_full[global_idx, r + rank4] = val

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
