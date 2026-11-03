"""Matrix multiplication 3x4x8 rank-minimal decomposition search."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify_safe(U, V, W, n=3, m=4, p=8, tol=1e-6):
    """
    Safely verify the decomposition using explicit loops to avoid einsum broadcasting issues.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Target: T[i,j,k] = 1 for all i,j,k.
    """
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    
    T_recon = np.zeros((n, m, p))
    for c in range(rank):
        # Extract vectors for this term
        u_c = U_arr[:, c]  # size n*m
        v_c = V_arr[:, c]  # size m*p
        w_c = W_arr[:, c]  # size p*n
        
        # Reshape to 2D
        u_2d = u_c.reshape(n, m)
        v_2d = v_c.reshape(m, p)
        w_2d = w_c.reshape(p, n)
        
        # Add outer product-like term
        # Term[i,j,k] = u_2d[i,j] * v_2d[j,k] * w_2d[k,i]
        # We can compute this by iterating or using broadcasting carefully.
        # Since n,m,p are small, a loop is fine and safe.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
    
    if not np.allclose(T_recon, 1.0, atol=tol):
        return False
    return True

def _construct_block_decomposition():
    """
    Construct a rank-88 decomposition using block splitting and Strassen's algorithm.
    Split A (3x4) into A1 (1x4) and A2 (2x4).
    Split B (4x8) into B1 (4x4) and B2 (4x4).
    
    C = A1 B1 + A1 B2 + A2 B1 + A2 B2
    
    We further split A1 into A11 (1x2) and A12 (1x2).
    We further split A2 into A21 (2x2) and A22 (2x2).
    We further split B1 into B11 (2x4) and B21 (2x4).
    We further split B2 into B12 (2x4) and B22 (2x4).
    
    Terms:
    1. A11 B11 (1x2x4) -> Rank 8
    2. A12 B21 (1x2x4) -> Rank 8
    3. A11 B12 (1x2x4) -> Rank 8
    4. A12 B22 (1x2x4) -> Rank 8
    5. A21 B11 (2x2x4) -> Rank 14 (Strassen)
    6. A22 B21 (2x2x4) -> Rank 14 (Strassen)
    7. A21 B12 (2x2x4) -> Rank 14 (Strassen)
    8. A22 B22 (2x2x4) -> Rank 14 (Strassen)
    
    Total Rank = 8*4 + 14*4 = 32 + 56 = 88.
    """
    n, m, p = 3, 4, 8
    
    # Global dimensions
    U = []
    V = []
    W = []
    
    # Helper to add a term from a sub-decomposition
    # sub_n, sub_m, sub_p: dimensions of the sub-problem
    # i_offset, j_offset, k_offset: offsets in global indices
    # u_sub, v_sub, w_sub: matrices for the sub-problem (shapes sub_n*sub_m, rank, etc.)
    def add_sub_term(i_off, j_off, k_off, sub_n, sub_m, sub_p, u_sub, v_sub, w_sub):
        rank_sub = u_sub.shape[1]
        for c in range(rank_sub):
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)
            
            # Fill U
            for i in range(sub_n):
                for j in range(sub_m):
                    global_i = i_off + i
                    global_j = j_off + j
                    u_col[global_i * m + global_j] = u_sub[i * sub_m + j, c]
            
            # Fill V
            for j in range(sub_m):
                for k in range(sub_p):
                    global_j = j_off + j
                    global_k = k_off + k
                    v_col[global_j * p + global_k] = v_sub[j * sub_p + k, c]
            
            # Fill W
            for k in range(sub_p):
                for i in range(sub_n):
                    global_k = k_off + k
                    global_i = i_off + i
                    w_col[global_k * n + global_i] = w_sub[k * sub_n + i, c]
            
            U.append(u_col)
            V.append(v_col)
            W.append(w_col)
    
    # --- Decomposition for 1x2x4 (Rank 8) ---
    # Schoolbook for 1x2x4 is sufficient and optimal for this small case?
    # Rank(1x2x4) = 8.
    def get_1x2x4():
        sn, sm, sp = 1, 2, 4
        u = np.zeros((sn * sm, 8))
        v = np.zeros((sm * sp, 8))
        w = np.zeros((sp * sn, 8))
        idx = 0
        for j in range(sm):
            for k in range(sp):
                u[j, idx] = 1.0
                v[j * sp + k, idx] = 1.0
                w[k * sn, idx] = 1.0
                idx += 1
        return u, v, w

    # --- Decomposition for 2x2x4 (Rank 14) ---
    # Use Strassen for 2x2x2 twice (for left and right 2 columns of B).
    def get_2x2x4():
        sn, sm, sp = 2, 2, 4
        # Strassen for 2x2x2
        # M1..M7
        # We need to map the 7 Strassen terms for the first 2 cols and 7 for the last 2 cols.
        
        # Strassen coefficients for 2x2x2
        # U indices (2x2), V indices (2x2), W indices (2x2)
        # W index is k*n+i. n=2.
        # k=0, i=0 -> 0. k=0, i=1 -> 1. k=1, i=0 -> 2. k=1, i=1 -> 3.
        
        strassen_u = [
            [1, 0, 1, 0, 1, 0, 0], # A11, A22, A11, A22, A11, A21, A12
            [0, 1, 1, 1, 0, 1, 1], # A21, A22, A21, A22, A12, A21, A22
            [0, 0, 0, 0, 0, 0, 0], # Wait, U is 2x2 = 4 elements.
        ]
        # Let's define Strassen properly.
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U vectors (A parts)
        # c=0: A11+A22 -> [1, 0, 0, 1] (Flattened A: A11, A12, A21, A22)
        # c=1: A21+A22 -> [0, 0, 1, 1]
        # c=2: A11 -> [1, 0, 0, 0]
        # c=3: A22 -> [0, 0, 0, 1]
        # c=4: A11+A12 -> [1, 1, 0, 0]
        # c=5: A21-A11 -> [-1, 0, 1, 0]
        # c=6: A12-A22 -> [0, 1, 0, -1]
        
        u_2x2x2 = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]).T # Shape (4, 7)
        
        # V vectors (B parts)
        # c=0: B11+B22 -> [1, 0, 0, 1] (Flattened B: B11, B12, B21, B22)
        # c=1: B11 -> [1, 0, 0, 0]
        # c=2: B12-B22 -> [0, 1, 0, -1]
        # c=3: B21-B11 -> [-1, 0, 1, 0]
        # c=4: B22 -> [0, 0, 0, 1]
        # c=5: B11+B12 -> [1, 1, 0, 0]
        # c=6: B21+B22 -> [0, 0, 1, 1]
        
        v_2x2x2 = np.array([
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]).T # Shape (4, 7)
        
        # W vectors (C parts)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # Flattened C: C11, C12, C21, C22
        # Indices in W: k*n+i. n=2.
        # C11 (i=0, k=0) -> W[0]
        # C12 (i=0, k=1) -> W[2]
        # C21 (i=1, k=0) -> W[1]
        # C22 (i=1, k=1) -> W[3]
        
        w_2x2x2 = np.zeros((4, 7))
        # M1
        w_2x2x2[0, 0] = 1 # C11
        w_2x2x2[3, 0] = 1 # C22
        # M2
        w_2x2x2[1, 1] = 1 # C21
        w_2x2x2[3, 1] = -1 # C22
        # M3
        w_2x2x2[2, 2] = 1 # C12
        w_2x2x2[3, 2] = 1 # C22
        # M4
        w_2x2x2[0, 3] = 1 # C11
        w_2x2x2[1, 3] = 1 # C21
        # M5
        w_2x2x2[0, 4] = -1 # C11
        w_2x2x2[2, 4] = 1 # C12
        # M6
        w_2x2x2[3, 5] = 1 # C22
        # M7
        w_2x2x2[0, 6] = 1 # C11
        
        # Now for 2x2x4, we use this for the first 2 columns of B (k=0,1) and last 2 columns (k=2,3).
        u_2x2x4 = np.zeros((4, 14))
        v_2x2x4 = np.zeros((8, 14))
        w_2x2x4 = np.zeros((8, 14)) # p=4, n=2 -> 4*2=8
        
        # First half (k=0,1)
        for c in range(7):
            u_2x2x4[:, c] = u_2x2x2[:, c]
            # V indices for k=0,1
            # v_2x2x2 has indices for B11, B12, B21, B22 which correspond to j=0,1 and k=0,1.
            # In 2x2x4, V is 2x4. Indices j*4+k.
            # B11 (j=0, k=0) -> V[0]
            # B12 (j=0, k=1) -> V[1]
            # B21 (j=1, k=0) -> V[4]
            # B22 (j=1, k=1) -> V[5]
            v_2x2x4[0, c] = v_2x2x2[0, c]
            v_2x2x4[1, c] = v_2x2x2[1, c]
            v_2x2x4[4, c] = v_2x2x2[2, c]
            v_2x2x4[5, c] = v_2x2x2[3, c]
            
            # W indices for k=0,1
            # w_2x2x2 has indices for C11, C12, C21, C22.
            # In 2x2x4, W is 4x2. Indices k*2+i.
            # C11 (i=0, k=0) -> W[0]
            # C12 (i=0, k=1) -> W[2]
            # C21 (i=1, k=0) -> W[1]
            # C22 (i=1, k=1) -> W[3]
            w_2x2x4[0, c] = w_2x2x2[0, c]
            w_2x2x4[2, c] = w_2x2x2[1, c] # Wait, w_2x2x2[1] is C12? No.
            # My w_2x2x2 mapping:
            # w_2x2x2[0] corresponds to C11 coeff
            # w_2x2x2[1] corresponds to C12 coeff?
            # Let's re-verify w_2x2x2 construction.
            # I put C11 at index 0, C12 at 2, C21 at 1, C22 at 3.
            # So w_2x2x2[0] is for C11.
            # w_2x2x2[1] is for C21? No, I wrote:
            # w_2x2x2[1, 1] = 1 # C21
            # w_2x2x2[2, 2] = 1 # C12
            # w_2x2x2[3, 0] = 1 # C22
            # So w_2x2x2[0] -> C11
            # w_2x2x2[1] -> C21
            # w_2x2x2[2] -> C12
            # w_2x2x2[3] -> C22
            
            # In 2x2x4, W indices k*2+i.
            # C11 (i=0, k=0) -> W[0]
            # C21 (i=1, k=0) -> W[1]
            # C12 (i=0, k=1) -> W[2]
            # C22 (i=1, k=1) -> W[3]
            
            w_2x2x4[0, c] = w_2x2x2[0, c] # C11
            w_2x2x4[1, c] = w_2x2x2[1, c] # C21
            w_2x2x4[2, c] = w_2x2x2[2, c] # C12
            w_2x2x4[3, c] = w_2x2x2[3, c] # C22
            
        # Second half (k=2,3)
        for c in range(7):
            u_2x2x4[:, c+7] = u_2x2x2[:, c]
            # V indices for k=2,3
            # B11 (j=0, k=2) -> V[2]
            # B12 (j=0, k=3) -> V[3]
            # B21 (j=1, k=2) -> V[6]
            # B22 (j=1, k=3) -> V[7]
            v_2x2x4[2, c+7] = v_2x2x2[0, c]
            v_2x2x4[3, c+7] = v_2x2x2[1, c]
            v_2x2x4[6, c+7] = v_2x2x2[2, c]
            v_2x2x4[7, c+7] = v_2x2x2[3, c]
            
            # W indices for k=2,3
            # C11 (i=0, k=2) -> W[4]
            # C21 (i=1, k=2) -> W[5]
            # C12 (i=0, k=3) -> W[6]
            # C22 (i=1, k=3) -> W[7]
            w_2x2x4[4, c+7] = w_2x2x2[0, c] # C11
            w_2x2x4[5, c+7] = w_2x2x2[1, c] # C21
            w_2x2x4[6, c+7] = w_2x2x2[2, c] # C12
            w_2x2x4[7, c+7] = w_2x2x2[3, c] # C22
            
        return u_2x2x4, v_2x2x4, w_2x2x4

    # Get sub-decompositions
    u_124, v_124, w_124 = get_1x2x4()
    u_224, v_224, w_224 = get_2x2x4()
    
    # Term 1: A11 B11
    # A11 is row 0, cols 0-1 of A. (i=0, j=0,1)
    # B11 is rows 0-1, cols 0-3 of B. (j=0,1, k=0..3)
    add_sub_term(0, 0, 0, 1, 2, 4, u_124, v_124, w_124)
    
    # Term 2: A12 B21
    # A12 is row 0, cols 2-3 of A. (i=0, j=2,3)
    # B21 is rows 2-3, cols 0-3 of B. (j=2,3, k=0..3)
    add_sub_term(0, 2, 0, 1, 2, 4, u_124, v_124, w_124)
    
    # Term 3: A11 B12
    # A11 is row 0, cols 0-1 of A.
    # B12 is rows 0-1, cols 4-7 of B. (j=0,1, k=4..7)
    add_sub_term(0, 0, 4, 1, 2, 4, u_124, v_124, w_124)
    
    # Term 4: A12 B22
    # A12 is row 0, cols 2-3 of A.
    # B22 is rows 2-3, cols 4-7 of B. (j=2,3, k=4..7)
    add_sub_term(0, 2, 4, 1, 2, 4, u_124, v_124, w_124)
    
    # Term 5: A21 B11
    # A21 is rows 1-2, cols 0-1 of A. (i=1,2, j=0,1)
    # B11 is rows 0-1, cols 0-3 of B. (j=0,1, k=0..3)
    add_sub_term(1, 0, 0, 2, 2, 4, u_224, v_224, w_224)
    
    # Term 6: A22 B21
    # A22 is rows 1-2, cols 2-3 of A. (i=1,2, j=2,3)
    # B21 is rows 2-3, cols 0-3 of B. (j=2,3, k=0..3)
    add_sub_term(1, 2, 0, 2, 2, 4, u_224, v_224, w_224)
    
    # Term 7: A21 B12
    # A21 is rows 1-2, cols 0-1 of A.
    # B12 is rows 0-1, cols 4-7 of B. (j=0,1, k=4..7)
    add_sub_term(1, 0, 4, 2, 2, 4, u_224, v_224, w_224)
    
    # Term 8: A22 B22
    # A22 is rows 1-2, cols 2-3 of A.
    # B22 is rows 2-3, cols 4-7 of B. (j=2,3, k=4..7)
    add_sub_term(1, 2, 4, 2, 2, 4, u_224, v_224, w_224)
    
    # Convert lists to 2D arrays
    U_arr = np.array(U).T # Transpose to (n*m, rank)
    V_arr = np.array(V).T
    W_arr = np.array(W).T
    
    return U_arr, V_arr, W_arr

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    best_result = None
    best_rank = float('inf')
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try block decomposition
    if time.time() < deadline:
        U_block, V_block, W_block = _construct_block_decomposition()
        if _verify_safe(U_block.tolist(), V_block.tolist(), W_block.tolist()):
            rank_block = U_block.shape[1]
            if rank_block < best_rank:
                best_result = {"U": U_block.tolist(), "V": V_block.tolist(), "W": W_block.tolist()}
                best_rank = rank_block
    
    return best_result
# EVOLVE-BLOCK-END
