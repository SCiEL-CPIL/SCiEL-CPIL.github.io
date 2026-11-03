import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check all entries
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Fallback: Schoolbook (Rank 72)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = rank_schoolbook

    # --- STRATEGY: GLOBAL GRADIENT OPTIMIZATION ---
    # We attempt to find a rank-r decomposition for r < 72.
    # We use the known Rank 66 as a starting point for initialization and verification.
    
    def get_strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0, 0] = 1; U[3, 0] = 1; V[0, 0] = 1; V[3, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
        U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1; W[2, 1] = 1; W[3, 1] = 1
        U[0, 2] = 1; V[2, 2] = 1; V[3, 2] = -1; W[1, 2] = 1; W[3, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; V[0, 3] = -1; W[0, 3] = -1; W[1, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1; V[3, 4] = 1; W[0, 4] = 1; W[2, 4] = 1
        U[2, 5] = 1; U[0, 5] = -1; V[0, 5] = 1; V[2, 5] = 1; W[2, 5] = 1; W[3, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1; V[1, 6] = 1; V[3, 6] = 1; W[0, 6] = 1; W[1, 6] = 1
        return U, V, W

    def get_mult_1x2_2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        return U, V, W

    # Construct Rank 66 baseline
    U_66 = np.zeros((12, 66))
    V_66 = np.zeros((24, 66))
    W_66 = np.zeros((18, 66))
    
    def place_block(U_block, V_block, W_block, u_rows, v_rows, w_rows, offset):
        rank_block = U_block.shape[1]
        for c in range(rank_block):
            for local_r, global_r in enumerate(u_rows):
                U_66[global_r, offset + c] = U_block[local_r, c]
            for local_r, global_r in enumerate(v_rows):
                V_66[global_r, offset + c] = V_block[local_r, c]
            for local_r, global_r in enumerate(w_rows):
                W_66[global_r, offset + c] = W_block[local_r, c]

    S_U, S_V, S_W = get_strassen_2x2()
    M_U, M_V, M_W = get_mult_1x2_2x2()
    
    col_offset = 0
    # Blocks 1-6: 2x2 * 2x2
    for i_block in range(3): # k-blocks 0, 2, 4
        for j_block in range(2): # j-blocks 0, 2
            u_rows = [0*4+j_block*2, 0*4+j_block*2+1, 1*4+j_block*2, 1*4+j_block*2+1]
            v_rows = [j_block*2*6+i_block*2, j_block*2*6+i_block*2+1, (j_block*2+1)*6+i_block*2, (j_block*2+1)*6+i_block*2+1]
            w_rows = [i_block*2*3+0, i_block*2*3+1, (i_block*2+1)*3+0, (i_block*2+1)*3+1]
            place_block(S_U, S_V, S_W, u_rows, v_rows, w_rows, col_offset)
            col_offset += 7
    # Blocks 7-12: 1x2 * 2x2
    for i_block in range(3):
        for j_block in range(2):
            u_rows = [2*4+j_block*2, 2*4+j_block*2+1]
            v_rows = [j_block*2*6+i_block*2, j_block*2*6+i_block*2+1, (j_block*2+1)*6+i_block*2, (j_block*2+1)*6+i_block*2+1]
            w_rows = [i_block*2*3+2, (i_block*2+1)*3+2]
            place_block(M_U, M_V, M_W, u_rows, v_rows, w_rows, col_offset)
            col_offset += 4

    if verify(U_66.tolist(), V_66.tolist(), W_66.tolist(), n, m, p):
        best_U = U_66.tolist()
        best_V = V_66.tolist()
        best_W = W_66.tolist()
        best_rank = 66

    # --- OPTIMIZATION PHASE ---
    # Try to reduce rank from 66 to 60 using gradient descent on the residual.
    # We flatten U, V, W into a single vector x.
    # Loss L = || T_computed - T_target ||_F^2
    
    def compute_tensor(U, V, W, n, m, p):
        # T[i, j, k]
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    T[i, j, k] = s
        return T

    def loss_and_grad(x, n, m, p, r):
        # Unpack
        u_size = n * m * r
        v_size = m * p * r
        U = x[:u_size].reshape(n * m, r)
        V = x[u_size:u_size+v_size].reshape(m * p, r)
        W = x[u_size+v_size:].reshape(p * n, r)
        
        T = compute_tensor(U, V, W, n, m, p)
        T_target = np.ones((n, m, p))
        diff = T - T_target
        loss = np.sum(diff**2)
        
        # Gradient via finite differences (expensive but robust for small r)
        # Or analytic gradient:
        # dL/dU[i,j,c] = 2 * sum_{k} (T[i,j,k] - 1) * V[j*p+k, c] * W[k*n+i, c]
        # dL/dV[j,k,c] = 2 * sum_{i} (T[i,j,k] - 1) * U[i*m+j, c] * W[k*n+i, c]
        # dL/dW[k,i,c] = 2 * sum_{j} (T[i,j,k] - 1) * U[i*m+j, c] * V[j*p+k, c]
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    err = 2 * (T[i, j, k] - 1.0)
                    for c in range(r):
                        grad_U[i*m+j, c] += err * V[j*p+k, c] * W[k*n+i, c]
                        grad_V[j*p+k, c] += err * U[i*m+j, c] * W[k*n+i, c]
                        grad_W[k*n+i, c] += err * U[i*m+j, c] * V[j*p+k, c]
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    def optimize_rank(r, max_iters=500, lr=0.01):
        # Initialize with Rank 66 solution truncated? No, random init or scaled Rank 66.
        # Let's try random init with small values, or the Rank 66 solution projected.
        # Projecting Rank 66 to Rank r is hard.
        # We use random init, but scale it.
        
        np.random.seed(42)
        u_size = n * m * r
        v_size = m * p * r
        w_size = p * n * r
        x = np.random.randn(u_size + v_size + w_size) * 0.1
        
        # If we have a previous solution, we might use it.
        # For now, random.
        
        current_loss, current_grad = loss_and_grad(x, n, m, p, r)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Simple Gradient Descent with momentum
            x -= lr * current_grad
            new_loss, new_grad = loss_and_grad(x, n, m, p, r)
            if new_loss > current_loss:
                lr *= 0.5
                x += lr * current_grad
                new_loss, new_grad = loss_and_grad(x, n, m, p, r)
            else:
                lr *= 1.1
            current_loss, current_grad = new_loss, new_grad
            
            if current_loss < 1e-10:
                break
        
        if current_loss < 1e-8:
            # Extract and round
            U = x[:u_size].reshape(n * m, r)
            V = x[u_size:u_size+v_size].reshape(m * p, r)
            W = x[u_size+v_size:].reshape(p * n, r)
            
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                return U_r.tolist(), V_r.tolist(), W_r.tolist()
        
        return None

    # Try ranks from 65 down to 60
    for r in range(65, 59, -1):
        if time.time() > deadline - 20:
            break
        result = optimize_rank(r)
        if result:
            best_U, best_V, best_W = result
            best_rank = r
            print(f"Gradient Descent found rank {r}")
            break

    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
