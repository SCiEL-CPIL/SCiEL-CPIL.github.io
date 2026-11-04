import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    
    n, m, p = 3, 3, 8
    
    # Helper to check time
    def time_left():
        return deadline - time.time()
        
    # Function to verify a decomposition
    def verify_decomp(U, V, W, n, m, p):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        R = U.shape[1]
        if R == 0:
            return False
        # We want to check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i, j, k
        # Vectorized check:
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_reshaped = U.reshape(n, m, R)
        V_reshaped = V.reshape(m, p, R)
        W_reshaped = W.reshape(p, n, R)
        
        # Compute the tensor T_est[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # We can do this by iterating over r, but R is small. 
        # Or use einsum: 'ijr,jkr,ki r -> ijk'
        # Note: W is indexed by (k, i) in the formula W[k*n+i, r]. 
        # In W_reshaped (p, n, R), the index is [k, i, r].
        # So we want einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        try:
            T_est = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        except Exception:
            return False
            
        # The target tensor T_target[i, j, k] is 1 for all valid i, j, k
        # Note: In matrix multiplication C = A @ B, C[i, k] = sum_j A[i, j] B[j, k].
        # The tensor entries correspond to the coefficient of A[i, j] B[j, k] in C[i, k].
        # This coefficient is 1 for all i, j, k.
        return np.allclose(T_est, 1.0, atol=1e-9)

    def search_decomp_fast(n, m, p, target_rank, time_limit):
        """
        Searches for a rank target_rank decomposition for M_{n,m,p} using vectorized SGD.
        """
        if time_left() < time_limit:
            return None
            
        np.random.seed(42)
        
        # Pre-compute indices for vectorization
        # U indices: (i, j) -> i*m + j
        # V indices: (j, k) -> j*p + k
        # W indices: (k, i) -> k*n + i
        
        # We will use a simple full-batch gradient descent with adaptive learning rate.
        # Since n, m, p are small (3,3,3), the number of constraints is 27.
        # Rank is ~23.
        
        best_loss = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # Try a few random initializations
        for attempt in range(10):
            if time_left() < time_limit:
                break
                
            # Initialize with small integers/half-integers
            # Using a range of [-1, 1] often works better for convergence
            U = np.random.uniform(-1, 1, (n*m, target_rank))
            V = np.random.uniform(-1, 1, (m*p, target_rank))
            W = np.random.uniform(-1, 1, (p*n, target_rank))
            
            # Reshape for easy einsum
            U_r = U.reshape(n, m, target_rank)
            V_r = V.reshape(m, p, target_rank)
            W_r = W.reshape(p, n, target_rank)
            
            lr = 0.05
            momentum = 0.9
            vel_U = np.zeros_like(U_r)
            vel_V = np.zeros_like(V_r)
            vel_W = np.zeros_like(W_r)
            
            for step in range(2000):
                if time_left() < time_limit:
                    break
                
                # Compute current tensor estimate
                # T_est[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                T_est = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                
                # Error
                error = 1.0 - T_est
                loss = np.sum(error**2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
                    # If loss is very small, try to round
                    if loss < 1e-6:
                        # Round to half-integers
                        U_round = np.round(U * 2) / 2
                        V_round = np.round(V * 2) / 2
                        W_round = np.round(W * 2) / 2
                        if verify_decomp(U_round, V_round, W_round, n, m, p):
                            return U_round, V_round, W_round
                        # Also try rounding to integers if half-integers fail?
                        # The problem allows integer/half-integer.
                
                if loss < 1e-10:
                    break
                
                # Compute gradients
                # dL/dU[i,j,r] = -2 * sum_k error[i,j,k] * V[j,k,r] * W[k,i,r]
                # dL/dV[j,k,r] = -2 * sum_i error[i,j,k] * U[i,j,r] * W[k,i,r]
                # dL/dW[k,i,r] = -2 * sum_j error[i,j,k] * U[i,j,r] * V[j,k,r]
                
                # Vectorized gradients using einsum
                # grad_U[i,j,r] = -2 * sum_k error[i,j,k] V[j,k,r] W[k,i,r]
                # We can compute this by summing over k
                # Let's do it step by step to be safe with indices
                # grad_U: (n, m, R)
                # Term: error[i,j,k] * V[j,k,r] * W[k,i,r]
                # Sum over k
                
                # Efficient way:
                # 1. Compute E_V_W[k, j, i, r]? No.
                # Let's use einsum for gradients directly.
                
                # grad_U[i, j, r] = -2 * einsum('ijk,jkr,kir->ijr', error, V_r, W_r)
                grad_U = -2.0 * np.einsum('ijk,jkr,kir->ijr', error, V_r, W_r)
                # grad_V[j, k, r] = -2 * einsum('ijk,ijr,kir->jkr', error, U_r, W_r)
                grad_V = -2.0 * np.einsum('ijk,ijr,kir->jkr', error, U_r, W_r)
                # grad_W[k, i, r] = -2 * einsum('ijk,ijr,jkr->kir', error, U_r, V_r)
                grad_W = -2.0 * np.einsum('ijk,ijr,jkr->kir', error, U_r, V_r)
                
                # Momentum update
                vel_U = momentum * vel_U - lr * grad_U
                vel_V = momentum * vel_V - lr * grad_V
                vel_W = momentum * vel_W - lr * grad_W
                
                U_r += vel_U
                V_r += vel_V
                W_r += vel_W
                
                # Optional: Decay learning rate
                if step % 100 == 0 and step > 0:
                    lr *= 0.95

            # After loop, check best
            if best_U is not None and best_loss < 1e-3:
                U_round = np.round(best_U * 2) / 2
                V_round = np.round(best_V * 2) / 2
                W_round = np.round(best_W * 2) / 2
                if verify_decomp(U_round, V_round, W_round, n, m, p):
                    return U_round, V_round, W_round
                # Try integer rounding as well
                U_int = np.round(best_U)
                V_int = np.round(best_V)
                W_int = np.round(best_W)
                if verify_decomp(U_int, V_int, W_round, n, m, p): # Typo in logic, fix below
                    pass
                if verify_decomp(U_int, V_int, W_int, n, m, p):
                    return U_int, V_int, W_int

        return None

    # Search for 3x3x3 decomposition (Target Rank 23)
    U_3, V_3, W_3 = None, None, None
    rank_3 = 27
    # Try rank 23 first
    if time_left() > 60:
        res = search_decomp_fast(3, 3, 3, 23, 50)
        if res is not None:
            U_3, V_3, W_3 = res
            rank_3 = 23
        else:
            # Fallback to schoolbook for 3x3x3
            rank_3 = 27
            U_3 = np.zeros((9, rank_3))
            V_3 = np.zeros((9, rank_3))
            W_3 = np.zeros((9, rank_3))
            col = 0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        U_3[i*3+j, col] = 1
                        V_3[j*3+k, col] = 1
                        W_3[k*3+i, col] = 1
                        col += 1
    else:
        # Not enough time, use schoolbook
        rank_3 = 27
        U_3 = np.zeros((9, rank_3))
        V_3 = np.zeros((9, rank_3))
        W_3 = np.zeros((9, rank_3))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_3[i*3+j, col] = 1
                    V_3[j*3+k, col] = 1
                    W_3[k*3+i, col] = 1
                    col += 1

    # Search for 3x3x2 decomposition (Target Rank 18, which is schoolbook)
    # We assume rank 18 is minimal for 3x3x2.
    U_2, V_2, W_2 = None, None, None
    rank_2 = 18
    # Just use schoolbook for 3x3x2 as it's likely minimal or close enough
    U_2 = np.zeros((9, rank_2))
    V_2 = np.zeros((6, rank_2))
    W_2 = np.zeros((6, rank_2))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U_2[i*3+j, col] = 1
                V_2[j*2+k, col] = 1
                W_2[k*3+i, col] = 1
                col += 1

    # Combine decompositions
    # 3x3x8 = 3x3x3 (cols 0-2) + 3x3x3 (cols 3-5) + 3x3x2 (cols 6-7)
    
    total_rank = rank_3 + rank_3 + rank_2
    
    U_final = np.zeros((9, total_rank))
    V_final = np.zeros((24, total_rank))
    W_final = np.zeros((24, total_rank))
    
    # Block 1: k=0,1,2
    U_final[:, 0:rank_3] = U_3
    for j in range(3):
        for k in range(3):
            V_final[j*8+k, 0:rank_3] = V_3[j*3+k]
    for k in range(3):
        for i in range(3):
            W_final[k*3+i, 0:rank_3] = W_3[k*3+i]
            
    # Block 2: k=3,4,5
    U_final[:, rank_3:2*rank_3] = U_3
    for j in range(3):
        for k in range(3):
            V_final[j*8+(k+3), rank_3:2*rank_3] = V_3[j*3+k]
    for k in range(3):
        for i in range(3):
            W_final[(k+3)*3+i, rank_3:2*rank_3] = W_3[k*3+i]
            
    # Block 3: k=6,7
    U_final[:, 2*rank_3:2*rank_3+rank_2] = U_2
    for j in range(3):
        for k in range(2):
            V_final[j*8+(k+6), 2*rank_3:2*rank_3+rank_2] = V_2[j*2+k]
    for k in range(2):
        for i in range(3):
            W_final[(k+6)*3+i, 2*rank_3:2*rank_3+rank_2] = W_2[k*3+i]
            
    # Verify final result
    if not verify_decomp(U_final, V_final, W_final, 3, 3, 8):
        # Fallback to pure schoolbook
        rank = 72
        U_final = np.zeros((9, rank))
        V_final = np.zeros((24, rank))
        W_final = np.zeros((24, rank))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(8):
                    U_final[i*3+j, col] = 1
                    V_final[j*8+k, col] = 1
                    W_final[k*3+i, col] = 1
                    col += 1
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
