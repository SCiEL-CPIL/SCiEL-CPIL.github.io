"""
Strategy: Build a rank-196 exact decomposition for 4x4x8 matrix multiplication using
tensor product (Kronecker) structure. Treat the 4x4x8 multiplication as:
- 4x4x4 part (along n, m, and first 4 of p) using Strassen rank-49 decomposition
- Plus 4x4x4 part (along n, m, and last 4 of p) using Strassen rank-49 decomposition
This gives rank 49+49 = 98.
Additionally, we can try to verify and potentially find lower rank via other methods.

Actually, let's be more careful. The multiplication is:
C[i,k] = sum_j A[i,j] B[j,k]  for i in 0..3, k in 0..7, j in 0..3

This is a single matrix multiplication of (4x4) * (4x8) = (4x8).
The tensor has indices (i, j, k) with T[i,j,k] = 1 at position (i*m+j, j*p+k, k*n+i)
where n=4, m=4, p=8.

We can decompose this by splitting k into two groups of 4:
- k=0..3: C[i,0..3] = A * B[:,0..3], which is a 4x4x4 multiplication
- k=4..7: C[i,4..7] = A * B[:,4..7], which is another 4x4x4 multiplication

Each 4x4x4 multiplication can be done with rank 49 (recursive Strassen).
Total rank = 49 + 49 = 98.

Let's build this properly.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _get_strassen_2x2():
    """
    Returns U, V, W for 2x2x2 Strassen.
    U[i,k] = coeff of A_i in term k
    V[j,k] = coeff of B_j in term k  
    W[i,k] = coeff of C_i in term k
    Indexing: i,j are in 0..3 (flattened 2x2), k is in 0..6
    A flattening: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    B flattening: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    C flattening: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
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
    Build rank-49 decomposition for 4x4x4 matrix multiplication.
    Uses recursive Strassen: treat 4x4 as 2x2 of 2x2 blocks.
    
    Returns U (16x49), V (16x49), W (16x49)
    where:
    - U index: i*4+j for A[i,j]
    - V index: j*4+k for B[j,k]  
    - W index: k*4+i for C[k,i]  (note: C[i,k] flattened as k*4+i per task convention)
    """
    U2, V2, W2 = _get_strassen_2x2()
    
    # We need to be very careful with index mappings.
    # For the outer 2x2 block structure:
    # A is divided into 4 blocks of 2x2:
    #   A00 = A[0:2,0:2], A01 = A[0:2,2:4]
    #   A10 = A[2:4,0:2], A11 = A[2:4,2:4]
    # B similarly:
    #   B00 = B[0:2,0:2], B01 = B[0:2,2:4]
    #   B10 = B[2:4,0:2], B11 = B[2:4,2:4]
    # C similarly:
    #   C00 = C[0:2,0:2], C01 = C[0:2,2:4]
    #   C10 = C[2:4,0:2], C11 = C[2:4,2:4]
    
    # But the task convention says W[k*n+i] = W[k*4+i] for C[i,k]
    # So C[i,k] is at index k*4+i in the W vector.
    # This means C[0,0]=0, C[0,1]=4, C[0,2]=8, C[0,3]=12
    # C[1,0]=1, C[1,1]=5, C[1,2]=9, C[1,3]=13
    # etc.
    
    # For the inner 2x2 blocks, we need to map local indices to global.
    # Let's define the mapping for each 2x2 block within the 4x4.
    
    # Block (br, bc) where br, bc in {0,1}:
    # Local indices (0,1,2,3) map to global:
    #   local 0 -> (br*2+0, bc*2+0) -> global idx = (br*2+0)*4 + (bc*2+0)
    #   local 1 -> (br*2+0, bc*2+1) -> global idx = (br*2+0)*4 + (bc*2+1)
    #   local 2 -> (br*2+1, bc*2+0) -> global idx = (br*2+1)*4 + (bc*2+0)
    #   local 3 -> (br*2+1, bc*2+1) -> global idx = (br*2+1)*4 + (bc*2+1)
    
    U4 = np.zeros((16, 49), dtype=np.float64)
    V4 = np.zeros((16, 49), dtype=np.float64)
    W4 = np.zeros((16, 49), dtype=np.float64)
    
    for k_out in range(7):
        for k_in in range(7):
            col = k_out * 7 + k_in
            
            u_vec = np.zeros(16)
            v_vec = np.zeros(16)
            w_vec = np.zeros(16)
            
            for b_out in range(4):
                # b_out: 0=A00, 1=A01, 2=A10, 3=A11
                br_out = b_out // 2
                bc_out = b_out % 2
                
                coeff_u = U2[b_out, k_out]
                coeff_v = V2[b_out, k_out]
                coeff_w = W2[b_out, k_out]
                
                if coeff_u == 0 and coeff_v == 0 and coeff_w == 0:
                    continue
                
                for b_in in range(4):
                    # b_in: local index within the 2x2 block
                    lr = b_in // 2
                    lc = b_in % 2
                    
                    gr = br_out * 2 + lr
                    gc = bc_out * 2 + lc
                    gidx = gr * 4 + gc
                    
                    u_coeff = U2[b_in, k_in]
                    v_coeff = V2[b_in, k_in]
                    w_coeff = W2[b_in, k_in]
                    
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
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 4, 4, 8

    # Build the full 4x4x8 multiplication tensor
    # T[a, b, c] where:
    # a = i*m + j (index into A, i=row, j=col)
    # b = j*p + k (index into B, j=row, k=col)
    # c = k*n + i (index into C, per task convention W[k*n+i])
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

            # Block 1: k=0..3 (Left half of B columns and C rows)
            U_full[:, 0:rank4] = U4
            
            # Map V4 (16x49) to V_full (32x49) for B_left
            # V4 indices are j*4+k' where j in 0..3, k' in 0..3
            # V_full indices are j*8+k where j in 0..3, k in 0..7
            # For k=0..3: V_full[j*8+k', r] = V4[j*4+k', r]
            for r in range(rank4):
                for j in range(4):
                    for k_prime in range(4):
                        local_idx = j * 4 + k_prime
                        global_idx = j * 8 + k_prime
                        V_full[global_idx, r] = V4[local_idx, r]

            # Map W4 (16x49) to W_full (32x49) for C_left
            # W4 indices are k'*4+i where k' in 0..3, i in 0..3
            # W_full indices are k*4+i where k in 0..7, i in 0..3
            # For k=0..3: W_full[k'*4+i, r] = W4[k'*4+i, r]
            for r in range(rank4):
                for k_prime in range(4):
                    for i in range(4):
                        local_idx = k_prime * 4 + i
                        global_idx = k_prime * 4 + i
                        W_full[global_idx, r] = W4[local_idx, r]

            # Block 2: k=4..7 (Right half of B columns and C rows)
            U_full[:, rank4:2*rank4] = U4
            
            # Map V4 to V_full for B_right
            # For k=4..7: V_full[j*8+(k'+4), r+rank4] = V4[j*4+k', r]
            for r in range(rank4):
                for j in range(4):
                    for k_prime in range(4):
                        local_idx = j * 4 + k_prime
                        global_idx = j * 8 + (k_prime + 4)
                        V_full[global_idx, r + rank4] = V4[local_idx, r]

            # Map W4 to W_full for C_right
            # For k=4..7: W_full[(k'+4)*4+i, r+rank4] = W4[k'*4+i, r]
            for r in range(rank4):
                for k_prime in range(4):
                    for i in range(4):
                        local_idx = k_prime * 4 + i
                        global_idx = (k_prime + 4) * 4 + i
                        W_full[global_idx, r + rank4] = W4[local_idx, r]

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
