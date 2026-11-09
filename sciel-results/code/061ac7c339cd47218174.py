import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    dim_u = n * m
    dim_v = m * p
    dim_w = p * n

    # Build the target tensor T
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
                    col += 1
        return U, V, W, rank

    def check_exact(U, V, W):
        """
        Check if U, V, W (assumed to be close to half-integers) form an exact decomposition.
        Returns (U_r, V_r, W_r) if valid, else None.
        """
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        
        # Reconstruct
        try:
            T_rec = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
            if np.allclose(T_rec, T, atol=1e-9):
                return U_r, V_r, W_r
        except Exception:
            pass
        return None

    def strassen_2x2x2():
        """
        Returns U, V, W for 2x2 * 2x2 multiplication using Strassen's rank-7 algorithm.
        Dimensions: n=2, m=2, p=2.
        U shape: (4, 7)
        V shape: (4, 7)
        W shape: (4, 7)
        """
        # Strassen's 7 multiplications
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Indices:
        # A: [[0,0],[0,1],[1,0],[1,1]] -> u_idx = i*2+j
        # B: [[0,0],[0,1],[1,0],[1,1]] -> v_idx = j*2+k
        # C: [[0,0],[0,1],[1,0],[1,1]] -> w_idx = k*2+i
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[3, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1
        
        return U, V, W

    def standard_2x1x2():
        """
        Returns U, V, W for 2x1 * 1x2 multiplication (rank 2).
        Dimensions: n=2, m=1, p=2.
        U shape: (2, 2)
        V shape: (2, 2)
        W shape: (4, 2)
        """
        # A is 2x1, B is 1x2. C is 2x2.
        # C[i,k] = A[i,0] * B[0,k]
        # 4 terms, rank 4? No, rank is m=1? No, rank is n*p = 4?
        # Wait, tensor is 2x1x2. Rank is n*m*p = 4.
        # Can we do better?
        # T[i, j, k] is non-zero only if j=0.
        # T[i, 0, k] = 1.
        # This is a sum of 4 rank-1 tensors?
        # Actually, for 2x1 * 1x2, the multiplication is just outer product of A and B.
        # Rank is 4.
        
        U = np.zeros((2, 4))
        V = np.zeros((2, 4))
        W = np.zeros((4, 4))
        
        for i in range(2):
            for k in range(2):
                col = i * 2 + k
                u_idx = i * 1 + 0
                v_idx = 0 * 2 + k
                w_idx = k * 2 + i
                
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                
        return U, V, W

    def construct_block_decomposition():
        """
        Construct a decomposition using block structure.
        Split A (2x3) and B (3x10) into blocks.
        
        Strategy:
        Split columns of B into 5 blocks of width 2.
        For each block b (0..4):
           B_block is 3x2.
           Split B_block into top 2 rows (2x2) and bottom 1 row (1x2).
           
           For the 2x2 part (rows 0,1 of B):
              A_part is 2x2 (rows 0,1 of A, cols 0,1? No, A is 2x3).
              Wait, A is 2x3. B is 3x10.
              We need to multiply A * B.
              
              Let's split B into 5 blocks of 3x2.
              Let B_b be the b-th block (cols 2b, 2b+1).
              
              We can split the multiplication A * B_b into:
              (A_top * B_b_top) + (A_bottom * B_b_bottom)
              
              A_top: rows 0,1 of A? No, A is 2x3.
              Let's split the middle dimension m=3.
              
              Option 1: Split m into 2+1.
              A = [A1 (2x2), A2 (2x1)]
              B = [B1 (2x10), B2 (1x10)]  -- wait, B is 3x10.
              B = [B1 (2x10); B2 (1x10)]
              
              C = A1 * B1 + A2 * B2
              
              For each column block of width 2 in B1 and B2:
              B1_block is 2x2. B2_block is 1x2.
              
              A1 is 2x2. A2 is 2x1.
              
              Term 1: A1 * B1_block (2x2 * 2x2 -> 2x2). Rank 7 (Strassen).
              Term 2: A2 * B2_block (2x1 * 1x2 -> 2x2). Rank 4 (Schoolbook).
              
              Total rank per column block = 7 + 4 = 11.
              Number of column blocks = 5.
              Total rank = 5 * 11 = 55.
              
              This matches the "inspiration" finding of rank 55.
        """
        # Initialize
        rank_total = 55
        U = np.zeros((dim_u, rank_total))
        V = np.zeros((dim_v, rank_total))
        W = np.zeros((dim_w, rank_total))
        
        col_offset = 0
        
        # Precompute Strassen and Standard blocks
        U_s, V_s, W_s = strassen_2x2x2()
        U_std, V_std, W_std = standard_2x1x2()
        
        for b in range(5):
            # Block b corresponds to columns 2b, 2b+1 of B
            # And rows of C are 0,1. Cols of C are 2b, 2b+1.
            
            # --- Part 1: A1 (2x2) * B1_block (2x2) ---
            # A1 uses rows 0,1 of A, cols 0,1 of A.
            # B1_block uses rows 0,1 of B, cols 2b, 2b+1 of B.
            
            # Map indices
            # A1[i, j] corresponds to A[i, j] where i in {0,1}, j in {0,1}.
            # u_idx for A1[i,j] is i*3 + j.
            # V1_block[j, k] corresponds to B[j, 2b+k] where j in {0,1}, k in {0,1}.
            # v_idx for B1_block[j,k] is j*10 + (2b+k).
            # W1[i, k] corresponds to C[i, 2b+k].
            # w_idx for C[i, 2b+k] is (2b+k)*2 + i.
            
            for r in range(7):
                # U
                for i in range(2):
                    for j in range(2):
                        if U_s[i*2+j, r] != 0:
                            u_idx = i * 3 + j
                            U[u_idx, col_offset + r] += U_s[i*2+j, r]
                
                # V
                for j in range(2):
                    for k in range(2):
                        if V_s[j*2+k, r] != 0:
                            v_idx = j * 10 + (2*b + k)
                            V[v_idx, col_offset + r] += V_s[j*2+k, r]
                
                # W
                for i in range(2):
                    for k in range(2):
                        if W_s[i*2+k, r] != 0:
                            w_idx = (2*b + k) * 2 + i
                            W[w_idx, col_offset + r] += W_s[i*2+k, r]
            
            col_offset += 7
            
            # --- Part 2: A2 (2x1) * B2_block (1x2) ---
            # A2 uses rows 0,1 of A, col 2 of A.
            # B2_block uses row 2 of B, cols 2b, 2b+1 of B.
            
            # A2[i, 0] corresponds to A[i, 2].
            # u_idx for A2[i,0] is i*3 + 2.
            # B2_block[0, k] corresponds to B[2, 2b+k].
            # v_idx for B2_block[0,k] is 2*10 + (2b+k) = 20 + 2b + k.
            # W2[i, k] corresponds to C[i, 2b+k].
            # w_idx for C[i, 2b+k] is (2*b + k) * 2 + i.
            
            for r in range(4):
                # U
                for i in range(2):
                    if U_std[i, r] != 0:
                        u_idx = i * 3 + 2
                        U[u_idx, col_offset + r] += U_std[i, r]
                
                # V
                for k in range(2):
                    if V_std[k, r] != 0:
                        v_idx = 2 * 10 + (2*b + k)
                        V[v_idx, col_offset + r] += V_std[k, r]
                
                # W
                for i in range(2):
                    for k in range(2):
                        if W_std[i*2+k, r] != 0:
                            w_idx = (2*b + k) * 2 + i
                            W[w_idx, col_offset + r] += W_std[i*2+k, r]
            
            col_offset += 4
        
        return U, V, W, rank_total

    # 1. Try the structured block decomposition (Rank 55)
    try:
        U_block, V_block, W_block, rank_block = construct_block_decomposition()
        sol = check_exact(U_block, V_block, W_block)
        if sol is not None:
            U, V, W = sol
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass

    # 2. Try ALS optimization to find a lower rank (e.g., 54, 53...)
    # Since we have a valid rank 55, we can try to optimize for rank 54.
    # However, ALS is stochastic and might not find it.
    # Given the time limit, and that 55 is a solid improvement over 60,
    # we might just return 55 if ALS fails quickly.
    
    # Let's try a few quick ALS runs for rank 54 to see if we can beat 55.
    if time.time() < deadline - 10:
        for rank in [54, 53]:
            if time.time() > deadline:
                break
            found = False
            for restart in range(5):
                if time.time() > deadline:
                    break
                # Random init
                U = np.random.randn(dim_u, rank)
                V = np.random.randn(dim_v, rank)
                W = np.random.randn(dim_w, rank)
                
                # Simple ALS loop
                for it in range(50):
                    if time.time() > deadline:
                        break
                    try:
                        # Solve U
                        T_u = T.reshape(dim_u, dim_v * dim_w)
                        A_u = np.einsum('vr,wr->vwr', V, W).reshape(dim_v * dim_w, rank)
                        U_new = np.linalg.lstsq(A_u, T_u, rcond=None)[0].T
                        
                        # Solve V
                        T_v = T.reshape(dim_u * dim_w, dim_v).T
                        A_v = np.einsum('ur,wr->uwr', U_new, W).reshape(dim_u * dim_w, rank)
                        V_new = np.linalg.lstsq(A_v, T_v, rcond=None)[0].T
                        
                        # Solve W
                        T_w = T.reshape(dim_u * dim_v, dim_w).T
                        A_w = np.einsum('ur,vr->uvr', U_new, V_new).reshape(dim_u * dim_v, rank)
                        W_new = np.linalg.lstsq(A_w, T_w, rcond=None)[0].T
                        
                        U, V, W = U_new, V_new, W_new
                        
                        # Check
                        T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
                        err = np.linalg.norm(T - T_rec)
                        if err < 1e-9:
                            sol = check_exact(U, V, W)
                            if sol is not None:
                                U, V, W = sol
                                found = True
                                break
                    except:
                        break
                if found:
                    break
            if found:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 3. Fallback to the block decomposition (Rank 55) if ALS failed or wasn't run
    try:
        U_block, V_block, W_block, rank_block = construct_block_decomposition()
        sol = check_exact(U_block, V_block, W_block)
        if sol is not None:
            U, V, W = sol
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # 4. Fallback to Schoolbook (Rank 60)
    U, V, W, rank = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
