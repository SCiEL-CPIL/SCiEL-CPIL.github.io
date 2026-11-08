import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 6 matrix.
    Uses a search for a low-rank decomposition of the 2x4x2 sub-problem and replicates it.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # The problem decomposes into 3 independent 2x4x2 multiplications for the 3 column blocks of C.
    # C[:, 0:2] = A @ B[:, 0:2]
    # C[:, 2:4] = A @ B[:, 2:4]
    # C[:, 4:6] = A @ B[:, 4:6]
    
    # We search for a rank-r decomposition for 2x4x2.
    # If we find one, total rank is 3*r.
    
    def find_2x4x2_decomp(target_rank, max_iters=2000, tol=1e-6):
        # Initialize with random or Strassen-like structure
        # U: 8 x r, V: 8 x r, W: 4 x r
        # A indices: i*m + j (i=0,1; j=0..3) -> 0..7
        # B indices: j*p + k (j=0..3; k=0,1) -> 0..7
        # C indices: k*n + i (k=0,1; i=0,1) -> 0..3
        
        # Use a heuristic initialization based on splitting into two 2x2x2 blocks
        # A = [A0 A1], B = [B0; B1]
        # C = A0 B0 + A1 B1
        # This can be done with rank 14 (two Strassen).
        # We try to compress.
        
        np.random.seed(42)
        U = np.random.randn(8, target_rank) * 0.1
        V = np.random.randn(8, target_rank) * 0.1
        W = np.random.randn(4, target_rank) * 0.1
        
        # Objective: || sum_c U_c V_c W_c^T - T ||^2
        # T is the tensor of matrix multiplication.
        # T[i,j,k,l] = 1 if j == k, else 0. (A_i,j B_j,k = C_i,k)
        # In our indices:
        # A_idx = i*4 + j
        # B_idx = j*2 + k
        # C_idx = k*2 + i
        
        # We want sum_c U[A_idx, c] V[B_idx, c] W[C_idx, c] = delta(j,k)
        
        # Gradient descent on U, V, W
        # f = sum_{i,j,k} (sum_c U[i*4+j, c] V[j*2+k, c] W[k*2+i, c] - delta(j,k))^2
        
        best_loss = float('inf')
        best_params = (U.copy(), V.copy(), W.copy())
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Compute current tensor
            C_calc = np.zeros((2, 4, 2)) # i, j, k
            for i in range(2):
                for j in range(4):
                    for k in range(2):
                        a_idx = i * 4 + j
                        b_idx = j * 2 + k
                        c_idx = k * 2 + i
                        C_calc[i, j, k] = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :])
                        
            # Target tensor
            C_target = np.zeros((2, 4, 2))
            for i in range(2):
                for j in range(4):
                    for k in range(2):
                        if j == k:
                            C_target[i, j, k] = 1.0
                            
            loss = np.sum((C_calc - C_target)**2)
            
            if loss < best_loss:
                best_loss = loss
                best_params = (U.copy(), V.copy(), W.copy())
                
            if loss < tol:
                break
                
            # Gradients
            # dL/dU[a,c] = 2 * sum_{i,j,k: a=i*4+j} (C_calc[i,j,k] - C_target[i,j,k]) * V[b,c] * W[c_idx,c]
            # This is expensive to compute explicitly in Python loops.
            # Use vectorized operations.
            
            # Reshape U, V, W to (8, r), (8, r), (4, r)
            # C_calc is (2, 4, 2).
            # We can compute gradients using einsum or matmul tricks.
            
            # Let's try a simple step size for now, or use a library if available? No, standard numpy.
            # We will use a simple gradient descent with small steps.
            
            eps = 1e-3
            U_new = U.copy()
            V_new = V.copy()
            W_new = W.copy()
            
            # Compute gradients numerically? Too slow.
            # Analytical gradients:
            # G_U[a, c] = 2 * sum_{i,j,k: a=i*4+j} Err[i,j,k] * V[j*2+k, c] * W[k*2+i, c]
            # G_V[b, c] = 2 * sum_{i,j,k: b=j*2+k} Err[i,j,k] * U[i*4+j, c] * W[k*2+i, c]
            # G_W[c_idx, c] = 2 * sum_{i,j,k: c_idx=k*2+i} Err[i,j,k] * U[i*4+j, c] * V[j*2+k, c]
            
            Err = C_calc - C_target
            
            G_U = np.zeros_like(U)
            G_V = np.zeros_like(V)
            G_W = np.zeros_like(W)
            
            for i in range(2):
                for j in range(4):
                    for k in range(2):
                        e = Err[i, j, k]
                        if abs(e) < 1e-10: continue
                        a_idx = i * 4 + j
                        b_idx = j * 2 + k
                        c_idx = k * 2 + i
                        G_U[a_idx, :] += 2 * e * V[b_idx, :] * W[c_idx, :]
                        G_V[b_idx, :] += 2 * e * U[a_idx, :] * W[c_idx, :]
                        G_W[c_idx, :] += 2 * e * U[a_idx, :] * V[b_idx, :]
                        
            # Update
            U -= 0.01 * G_U
            V -= 0.01 * G_V
            W -= 0.01 * G_W
            
        # Check best params
        U_b, V_b, W_b = best_params
        C_check = np.zeros((2, 4, 2))
        for i in range(2):
            for j in range(4):
                for k in range(2):
                    a_idx = i * 4 + j
                    b_idx = j * 2 + k
                    c_idx = k * 2 + i
                    C_check[i, j, k] = np.sum(U_b[a_idx, :] * V_b[b_idx, :] * W_b[c_idx, :])
                    
        C_target = np.zeros((2, 4, 2))
        for i in range(2):
            for j in range(4):
                for k in range(2):
                    if j == k:
                        C_target[i, j, k] = 1.0
        final_loss = np.sum((C_check - C_target)**2)
        
        if final_loss < 1e-4:
            # Round to nearest integer/half-integer
            U_int = np.round(U_b * 2) / 2
            V_int = np.round(V_b * 2) / 2
            W_int = np.round(W_b * 2) / 2
            
            # Verify rounded
            C_round = np.zeros((2, 4, 2))
            for i in range(2):
                for j in range(4):
                    for k in range(2):
                        a_idx = i * 4 + j
                        b_idx = j * 2 + k
                        c_idx = k * 2 + i
                        C_round[i, j, k] = np.sum(U_int[a_idx, :] * V_int[b_idx, :] * W_int[c_idx, :])
                        
            if np.sum((C_round - C_target)**2) < 1e-6:
                return U_int, V_int, W_int
        return None

    best_rank = None
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 7 to 12
    for r in range(7, 13):
        if time.time() > deadline - 20:
            break
        res = find_2x4x2_decomp(r, max_iters=500)
        if res is not None:
            best_rank = r
            best_U, best_V, best_W = res
            break # Found a valid one, stop searching higher ranks
            
    if best_rank:
        total_rank = 3 * best_rank
        U_full = [[0] * total_rank for _ in range(8)]
        V_full = [[0] * total_rank for _ in range(24)]
        W_full = [[0] * total_rank for _ in range(12)]
        
        for block in range(3):
            offset = block * best_rank
            for i in range(8):
                for c in range(best_rank):
                    U_full[i][offset + c] = best_U[i][c]
                    
            for j in range(4):
                for k in range(2):
                    b_idx = j * 2 + k
                    for c in range(best_rank):
                        V_full[b_idx * 3 + block][offset + c] = best_V[b_idx][c] # Wait, B index mapping
                        # B is 4x6. Index j*6+k.
                        # In sub-problem, B is 4x2. Index j*2+k_sub.
                        # Global B index: j*6 + block*2 + k_sub.
                        
            # Correct V mapping
            V_full = [[0] * total_rank for _ in range(24)]
            for j in range(4):
                for k_sub in range(2):
                    b_local = j * 2 + k_sub
                    b_global = j * 6 + block * 2 + k_sub
                    for c in range(best_rank):
                        V_full[b_global][offset + c] = best_V[b_local][c]
                        
            for k_sub in range(2):
                for i in range(2):
                    c_local = k_sub * 2 + i
                    c_global = (block * 2 + k_sub) * 2 + i
                    for c in range(best_rank):
                        W_full[c_global][offset + c] = best_W[c_local][c]
                        
        # Final validation
        valid = True
        for i in range(2):
            for j in range(4):
                for k in range(6):
                    s = 0.0
                    a_idx = i * 4 + j
                    b_idx = j * 6 + k
                    c_idx = k * 2 + i
                    for c in range(total_rank):
                        s += U_full[a_idx][c] * V_full[b_idx][c] * W_full[c_idx][c]
                    if abs(s - (1.0 if j == k // 2 and j == k % 2 else 0.0)) > 1e-6: 
                         # Check condition: C[i,k] = sum_j A[i,j]B[j,k]
                         # Term is 1 if j==k? No.
                         # The tensor T[i,j,k] is 1 if j==k.
                         # Here k is global col. j is global row of B.
                         # So yes, if j==k, term is 1.
                         if abs(s - (1.0 if j == k else 0.0)) > 1e-6:
                             valid = False
                             break
                if not valid: break
            if not valid: break
        if valid:
            return {'U': U_full, 'V': V_full, 'W': W_full}
            
    # Fallback to current rank 42 solution
    rank = 42
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(24)]
    W = [[0] * rank for _ in range(12)]
    
    def get_a_idx(group, local_i, local_j):
        global_col = local_j + 2 * group
        global_row = local_i
        return global_row * 4 + global_col

    def get_b_idx(group, local_i, local_j, block_col):
        global_row = local_i + 2 * group
        global_col = local_j + 2 * block_col
        return global_row * 6 + global_col

    def get_c_idx(local_i, local_j, block_col):
        global_col = local_j + 2 * block_col
        global_row = local_i
        return global_col * 2 + global_row

    col = 0
    for group in [0, 1]:
        for block_col in [0, 1, 2]:
            U[get_a_idx(group, 0, 0)][col] += 1
            U[get_a_idx(group, 1, 1)][col] += 1
            V[get_b_idx(group, 0, 0, block_col)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] += 1
            W[get_c_idx(0, 0, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] += 1
            col += 1
            
            U[get_a_idx(group, 1, 0)][col] += 1
            U[get_a_idx(group, 1, 1)][col] += 1
            V[get_b_idx(group, 0, 0, block_col)][col] += 1
            W[get_c_idx(1, 0, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] -= 1
            col += 1
            
            U[get_a_idx(group, 0, 0)][col] += 1
            V[get_b_idx(group, 0, 1, block_col)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] -= 1
            W[get_c_idx(0, 1, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] += 1
            col += 1
            
            U[get_a_idx(group, 1, 1)][col] += 1
            V[get_b_idx(group, 1, 0, block_col)][col] += 1
            V[get_b_idx(group, 0, 0, block_col)][col] -= 1
            W[get_c_idx(0, 0, block_col)][col] += 1
            W[get_c_idx(1, 0, block_col)][col] += 1
            col += 1
            
            U[get_a_idx(group, 0, 0)][col] += 1
            U[get_a_idx(group, 0, 1)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] += 1
            W[get_c_idx(0, 0, block_col)][col] -= 1
            W[get_c_idx(0, 1, block_col)][col] += 1
            col += 1
            
            U[get_a_idx(group, 1, 0)][col] += 1
            U[get_a_idx(group, 0, 0)][col] -= 1
            V[get_b_idx(group, 0, 0, block_col)][col] += 1
            V[get_b_idx(group, 0, 1, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] += 1
            col += 1
            
            U[get_a_idx(group, 0, 1)][col] += 1
            U[get_a_idx(group, 1, 1)][col] -= 1
            V[get_b_idx(group, 1, 0, block_col)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] += 1
            W[get_c_idx(0, 0, block_col)][col] += 1
            col += 1
             
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {'U': U, 'V': V, 'W': W}
