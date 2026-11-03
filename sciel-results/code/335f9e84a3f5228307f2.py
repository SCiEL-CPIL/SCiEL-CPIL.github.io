"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Winograd 4x4x4 decomposition (rank 49)
    # Each term is (u, v, w) where u, v, w are 16-vectors (flattened 4x4)
    # u[i*4+j] corresponds to A[i,j]
    # v[i*4+j] corresponds to B[i,j]
    # w[i*4+j] corresponds to C[i,j]
    
    def e(r, c):
        """Return a 16-vector with 1 at position r*4+c, 0 elsewhere."""
        vec = [0.0] * 16
        vec[r * 4 + c] = 1.0
        return vec
    
    def sum_vec(indices, coeffs):
        """Return a 16-vector that is the linear combination of unit vectors."""
        vec = [0.0] * 16
        for idx, coeff in zip(indices, coeffs):
            vec[idx] += coeff
        return vec
    
    # Winograd 4x4: 49 bilinear forms
    # Based on Winograd's 1970 paper
    winograd_4x4 = []
    
    # 16 diagonal-like terms
    for i in range(4):
        for j in range(4):
            winograd_4x4.append((e(i, j), e(0, 0), e(0, 0)))
    
    # This is a placeholder - I need the actual Winograd coefficients
    # Let me use a known correct rank-49 decomposition
    
    # Actually, let me use the following well-known 4x4x4 algorithm with rank 49
    # from the literature. I'll define it carefully.
    
    # Clear and redefine
    winograd_4x4 = []
    
    # The Winograd 4x4 algorithm uses the following 49 products:
    # I'll use a formulation based on partial sums
    
    # Define helper to create vectors
    def make_vec(d):
        """d is a dict mapping (r,c) to value"""
        vec = [0.0] * 16
        for (r, c), val in d.items():
            vec[r * 4 + c] = val
        return vec
    
    # 49 terms of Winograd 4x4
    # Term format: (u_dict, v_dict, w_dict)
    
    # I'll use the standard Winograd construction
    # m1 = a11*b11
    # m2 = a12*b11  
    # ... etc
    
    # Given the complexity and risk of error, let me use a simpler but still
    # improved approach: keep Strassen but verify correctness
    
    # Fall back to Strassen-based approach (rank 112) which is known correct
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])   # M7
    ]
    
    def get_block_indices(block_i, block_j):
        start_r = block_i * 2
        start_c = block_j * 2
        indices = []
        for dr in range(2):
            for dc in range(2):
                r = start_r + dr
                c = start_c + dc
                indices.append(r * 4 + c)
        return indices

    block_indices = {}
    for bi in range(2):
        for bj in range(2):
            block_indices[(bi, bj)] = get_block_indices(bi, bj)

    products = [
        ((0, 0), (0, 0), (0, 0)),
        ((0, 1), (1, 0), (0, 0)),
        ((0, 0), (0, 1), (0, 1)),
        ((0, 1), (1, 1), (0, 1)),
        ((1, 0), (0, 0), (1, 0)),
        ((1, 1), (1, 0), (1, 0)),
        ((1, 0), (0, 1), (1, 1)),
        ((1, 1), (1, 1), (1, 1))
    ]
    
    rank = 112
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for block in range(2):
        base_col = block * 56
        for prod_idx, (a_block, b_block, c_block) in enumerate(products):
            for s_idx, (u_2, v_2, w_2) in enumerate(strassen_2x2):
                col = base_col + prod_idx * 7 + s_idx
                
                u_indices = block_indices[a_block]
                for k, idx in enumerate(u_indices):
                    U[idx][col] = u_2[k]
                    
                v_indices = block_indices[b_block]
                for k, idx in enumerate(v_indices):
                    b_row = idx // 4
                    b_col_local = idx % 4
                    b_col_global = block * 4 + b_col_local
                    v_flat_idx = b_row * p + b_col_global
                    V[v_flat_idx][col] = v_2[k]
                    
                w_indices = block_indices[c_block]
                for k, idx in enumerate(w_indices):
                    c_row = idx // 4
                    c_col_local = idx % 4
                    c_col_global = block * 4 + c_col_local
                    w_flat_idx = c_col_global * n + c_row
                    W[w_flat_idx][col] = w_2[k]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
