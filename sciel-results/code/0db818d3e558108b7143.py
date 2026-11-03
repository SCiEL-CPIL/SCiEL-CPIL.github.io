import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Strategy 1: Strassen tiling (rank 88)
    # Strategy 2: Tensor product approach (target rank ~44)
    
    best_result = None
    best_rank = float('inf')
    
    # --- Strategy 2: Tensor product M_{3,4,4} ⊗ M_{1,1,2} ---
    # For 3x4x4: split j into pairs, use Strassen for rows 0,1 and schoolbook for row 2
    # Then extend k dimension by factor 2
    
    if time.time() < deadline - 30:
        # Build 3x4x4 core decomposition
        # j pairs: (0,1) and (2,3)
        # For each j-pair: Strassen for i={0,1} (rank 7), schoolbook for i={2} (rank 2 per 2 k's, but we have 2 k's in the 4x4, so rank 2)
        # Wait, for 3x4x4: i in {0,1,2}, j in {0,1,2,3}, k in {0,1,2,3}
        # Split j into (0,1) and (2,3), k into (0,1) and (2,3)
        # For i={0,1}: 2x2x2 Strassen blocks: 2 j-pairs × 2 k-pairs × 7 = 28
        # For i={2}: 1x2x2 schoolbook: 2 j-pairs × 2 k-pairs × 4 = 16
        # Total 3x4x4 rank = 44
        
        # Then tensor with 1x1x2 (rank 2): total rank = 44 * 2 = 88? No...
        # Actually the tensor product structure: M_{3,4,8}[i,j,k] where k=2l+k'
        # This isn't a pure tensor product because the middle dimension is shared.
        
        # Let me try a direct approach: for each (i,j) pair, the output is sum over k
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # Let me try: split k into two halves (0-3, 4-7)
        # For each half, we have a 3x4x4 problem
        # But we can share the A factors!
        
        # Actually, let's try: C[i, 2l+k'] = sum_j A[i,j] * B[j, 2l+k']
        # = sum_j (A[i,j]) * B[j, 2l+k']
        
        # If we define for each l: B_l[j, k'] = B[j, 2l+k']
        # Then C[i, 2l+k'] = sum_j A[i,j] * B_l[j, k']
        # This is 4 independent 3x4x2 problems, each of rank 3*4*2 = 24 schoolbook, or lower with Strassen.
        
        # For 3x4x2: i in {0,1,2}, j in {0,1,2,3}, k in {0,1}
        # Split j into pairs: (0,1) and (2,3)
        # For i={0,1}: 2x2x2 Strassen: 2 j-pairs × 7 = 14
        # For i={2}: 1x2x2 schoolbook: 2 j-pairs × 4 = 8
        # Total per 3x4x2: 22
        # 4 such blocks: 4 × 22 = 88
        
        # Same as before. Let me try a different split.
        
        # What if we use Winograd-style for the 3x4x4 core?
        # Actually, let me just optimize the rank-88 solution with better column removal.
        
        pass
    
    # --- Strategy 1: Efficient Strassen construction (rank 88) ---
    strassen_decomp = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    rank = 88
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for j1 in range(2):
        for l in range(4):
            for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                for li, coeff in u_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_i = li_i
                    global_j = 2 * j1 + li_j
                    U[global_i * m + global_j, col] += coeff
                for li, coeff in v_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_j = 2 * j1 + li_i
                    global_k = 2 * l + li_j
                    V[global_j * p + global_k, col] += coeff
                for li, coeff in w_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_i = li_i
                    global_k = 2 * l + li_j
                    W[global_k * n + global_i, col] += coeff
                col += 1
        
        for k in range(8):
            for j2 in range(2):
                global_i = 2
                global_j = 2 * j1 + j2
                U[global_i * m + global_j, col] = 1
                V[global_j * p + k, col] = 1
                W[k * n + global_i, col] = 1
                col += 1
    
    # Verify
    def verify(U_arr, V_arr, W_arr):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback: schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=np.float64)
        V_fb = np.zeros((m * p, rank_fb), dtype=np.float64)
        W_fb = np.zeros((p * n, rank_fb), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, idx] = 1
                    V_fb[j * p + k, idx] = 1
                    W_fb[k * n + i, idx] = 1
                    idx += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # --- Efficient column removal using linear algebra ---
    # A column r is redundant if the vector (U[:,r], V[:,r], W[:,r]) can be expressed
    # as a linear combination of other columns in a way that preserves the tensor product.
    # This is hard to check directly. Instead, use a simpler heuristic:
    # Check if column r is a linear combination of others in each matrix simultaneously.
    
    if time.time() < deadline - 20:
        current_rank = rank
        U_curr = U[:, :current_rank].copy()
        V_curr = V[:, :current_rank].copy()
        W_curr = W[:, :current_rank].copy()
        
        # Try removing columns from the end (schoolbook columns first, as they're more likely redundant)
        # Actually, let's try all columns but use a faster verification
        removed_any = True
        while removed_any and time.time() < deadline - 20:
            removed_any = False
            current_rank = U_curr.shape[1]
            for r in range(current_rank - 1, -1, -1):
                if time.time() > deadline - 20:
                    break
                
                # Try removing column r
                U_test = np.delete(U_curr, r, axis=1)
                V_test = np.delete(V_curr, r, axis=1)
                W_test = np.delete(W_curr, r, axis=1)
                
                # Fast verification using matrix operations
                # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
                # We can compute this as: for each c, the contribution is U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                # Sum over c. We can vectorize:
                
                verified = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U_test[i * m + j] * V_test[j * p + k], W_test[k * n + i])
                            if abs(s - 1.0) > 1e-9:
                                verified = False
                                break
                        if not verified:
                            break
                    if not verified:
                        break
                
                if verified:
                    U_curr = U_test
                    V_curr = V_test
                    W_curr = W_test
                    removed_any = True
                    break
        
        if U_curr.shape[1] < rank:
            U, V, W = U_curr, V_curr, W_curr
    
    # Convert to list with half-integer rounding
    def to_list(mat):
        mat_rounded = np.round(mat * 2) / 2
        if np.allclose(mat, mat_rounded, atol=1e-9):
            return mat_rounded.tolist()
        else:
            return mat.tolist()
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
