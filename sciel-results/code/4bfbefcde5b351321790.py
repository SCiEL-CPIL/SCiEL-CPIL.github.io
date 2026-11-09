import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    def find_2x4x2_decomp(target_rank, max_iters=5000, tol=1e-10, n_restarts=5):
        # Precompute the nonzero entries of the tensor
        # T[a,b,c] = 1 if a=i*4+j, b=j*2+k, c=k*2+i
        entries = []
        for i in range(2):
            for j in range(4):
                for k in range(2):
                    a_idx = i * 4 + j
                    b_idx = j * 2 + k
                    c_idx = k * 2 + i
                    entries.append((a_idx, b_idx, c_idx))
        
        entries = np.array(entries)  # shape (16, 3)
        A_idx = entries[:, 0]
        B_idx = entries[:, 1]
        C_idx = entries[:, 2]
        
        best_loss = float('inf')
        best_params = None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 20:
                break
                
            np.random.seed(42 + restart)
            U = np.random.randn(8, target_rank) * 0.5
            V = np.random.randn(8, target_rank) * 0.5
            W = np.random.randn(4, target_rank) * 0.5
            
            for it in range(max_iters):
                if time.time() > deadline - 20:
                    break
                    
                # Compute current tensor values at nonzero entries
                # C_calc[t] = sum_r U[A_idx[t], r] * V[B_idx[t], r] * W[C_idx[t], r]
                C_calc = np.einsum('ar,br,cr->t', U[A_idx], V[B_idx], W[C_idx])
                
                # Target is all ones
                Err = C_calc - 1.0
                loss = np.sum(Err**2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_params = (U.copy(), V.copy(), W.copy())
                    
                if loss < tol:
                    break
                
                # Compute gradients
                # dL/dU[a,r] = 2 * sum_{t: A_idx[t]==a} Err[t] * V[B_idx[t],r] * W[C_idx[t],r]
                # Vectorized:
                G_U = np.zeros_like(U)
                G_V = np.zeros_like(V)
                G_W = np.zeros_like(W)
                
                # For each term t, add Err[t] * V[B_idx[t]] * W[C_idx[t]] to G_U[A_idx[t]]
                # G_U[A_idx[t], :] += Err[t] * V[B_idx[t], :] * W[C_idx[t], :]
                
                # Vectorized using advanced indexing
                # This is tricky. Let's do it in a loop over the 16 entries
                for t in range(16):
                    e = Err[t]
                    if abs(e) < 1e-12:
                        continue
                    a = A_idx[t]
                    b = B_idx[t]
                    c = C_idx[t]
                    G_U[a, :] += 2 * e * V[b, :] * W[c, :]
                    G_V[b, :] += 2 * e * U[a, :] * W[c, :]
                    G_W[c, :] += 2 * e * U[a, :] * V[b, :]
                
                # Adaptive learning rate
                lr = 0.01
                U -= lr * G_U
                V -= lr * G_V
                W -= lr * G_W
        
        if best_params is not None:
            U_b, V_b, W_b = best_params
            # Verify
            C_calc = np.einsum('ar,br,cr->t', U_b[A_idx], V_b[B_idx], W_b[C_idx])
            final_loss = np.sum((C_calc - 1.0)**2)
            
            if final_loss < 1e-6:
                # Round to nearest integer/half-integer
                U_int = np.round(U_b * 2) / 2
                V_int = np.round(V_b * 2) / 2
                W_int = np.round(W_b * 2) / 2
                
                C_round = np.einsum('ar,br,cr->t', U_int[A_idx], V_int[B_idx], W_int[C_idx])
                if np.sum((C_round - 1.0)**2) < 1e-6:
                    return U_int, V_int, W_int
        return None
    
    best_rank = None
    best_U = None
    best_V = None
    best_W = None
    
    for r in range(7, 13):
        if time.time() > deadline - 30:
            break
        res = find_2x4x2_decomp(r, max_iters=3000, n_restarts=3)
        if res is not None:
            best_rank = r
            best_U, best_V, best_W = res
            break
    
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
                    expected = 1.0 if j == k else 0.0
                    if abs(s - expected) > 1e-6:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
        if valid:
            return {'U': U_full, 'V': V_full, 'W': W_full}
    
    # Fallback to rank 42
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
            
    return {'U': U, 'V': U, 'W': W}
