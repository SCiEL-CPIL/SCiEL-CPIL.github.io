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
        # Check all i, j, k
        # T[i, j, k] = 1
        # Sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can vectorize this
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_reshaped = U.reshape(n, m, R)
        V_reshaped = V.reshape(m, p, R)
        W_reshaped = W.reshape(p, n, R)
        
        # Compute contraction
        # Result shape (n, p)
        # C[i, k] = sum_j sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        # Note: W index is (k, i) in the tensor definition?
        # Task says: W[k*n+i, r]. So W is indexed by (k, i).
        # My W_reshaped is (p, n, R) so W_reshaped[k, i, r] is correct.
        
        # Perform multiplication
        # Temp[j, k, r] = sum_i? No.
        # Let's do it step by step.
        # Step 1: X[i, j, k, r] = U[i, j, r] * V[j, k, r]
        # This is large.
        # Better:
        # A_r = U[:, r] -> (n*m, 1)
        # B_r = V[:, r] -> (m*p, 1)
        # C_r = W[:, r] -> (p*n, 1)
        
        # We want to check if sum_r outer(U_r, V_r, W_r) == T
        # T is 1 at (i, j, k) for all valid i,j,k.
        
        # Efficient check:
        # Compute M = sum_r U_r * V_r * W_r
        # U_r is vector of length n*m.
        # V_r is vector of length m*p.
        # W_r is vector of length p*n.
        
        # Let's just loop over r and accumulate
        T_acc = np.zeros((n, m, p))
        for r in range(R):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            # The tensor term is u[i,j] * v[j,k] * w[k,i]
            # We need to check if the sum equals 1 for all i,j,k
            # Actually, the decomposition defines the bilinear form.
            # The tensor T_{ijk} is 1.
            # So we check if sum_r u[i,j] v[j,k] w[k,i] == 1 for all i,j,k.
            
            # We can accumulate into a 3D array
            # term[i, j, k] = u[i, j] * v[j, k] * w[k, i]
            # This is an outer product of u, v, w with specific index mapping.
            # u: (n, m)
            # v: (m, p)
            # w: (p, n)
            
            # term[i, j, k] = u[i, j] * v[j, k] * w[k, i]
            # We can compute this as:
            # temp[j, k] = sum_i? No.
            # Let's just use einsum if possible, or loops.
            # n, m, p are small (3, 3, 2 or 3, 3, 3).
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_acc[i, j, k] += u[i, j] * v[j, k] * w[k, i]
                        
        # Check if T_acc is all 1s
        return np.allclose(T_acc, 1.0, atol=1e-9)

    def search_decomp(n, m, p, target_rank, time_limit):
        """
        Searches for a rank target_rank decomposition for M_{n,m,p}.
        Returns (U, V, W) if found, else None.
        """
        if time_left() < time_limit:
            return None
            
        np.random.seed(42)
        best_loss = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # Initialize with small integers
        # Try a few random initializations
        for attempt in range(5):
            if time_left() < time_limit:
                break
                
            U = np.random.randint(-2, 3, (n*m, target_rank)) * 0.5
            V = np.random.randint(-2, 3, (m*p, target_rank)) * 0.5
            W = np.random.randint(-2, 3, (p*n, target_rank)) * 0.5
            
            # Gradient Descent
            lr = 0.01
            for step in range(500):
                if time_left() < time_limit:
                    return best_U, best_V, best_W if best_loss < 1e-6 else None
                
                # Compute gradient
                # Loss = sum_{i,j,k} (T_{ijk} - sum_r U_{ij,r} V_{jk,r} W_{ki,r})^2
                # T_{ijk} = 1
                
                # We can compute the gradient numerically or analytically.
                # Analytical is better.
                # Let S_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
                # dL/dU_{ij,r} = -2 * (1 - S_{ijk}) * V_{jk,r} * W_{ki,r} (summed over j,k? No, fixed i,j)
                # Wait, S depends on i,j,k.
                # For fixed i,j, the term in S is U_{ij,r} * (sum_k V_{jk,r} W_{ki,r})?
                # No, S_{ijk} = U_{ij,r} V_{jk,r} W_{ki,r}.
                # dL/dU_{ij,r} = -2 * sum_{k} (1 - S_{ijk}) * V_{jk,r} * W_{ki,r}
                
                # This is complex to vectorize easily without memory blowup.
                # Given small sizes, we can do a simple numerical gradient or coordinate descent.
                # Let's use coordinate descent on one variable at a time?
                # Or just use a simple update rule.
                
                # Let's try a simple stochastic gradient descent step
                # Pick random i, j, k
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                
                # Current value at (i, j, k)
                val = 0.0
                for r in range(target_rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    
                err = 1.0 - val
                
                # Update U, V, W
                for r in range(target_rank):
                    u_val = U[i*m+j, r]
                    v_val = V[j*p+k, r]
                    w_val = W[k*n+i, r]
                    
                    # Gradient for U
                    grad_U = -2 * err * v_val * w_val
                    # Gradient for V
                    grad_V = -2 * err * u_val * w_val
                    # Gradient for W
                    grad_W = -2 * err * u_val * v_val
                    
                    U[i*m+j, r] -= lr * grad_U
                    V[j*p+k, r] -= lr * grad_V
                    W[k*n+i, r] -= lr * grad_W
                
                # Periodic rounding check
                if step % 50 == 0:
                    # Compute loss
                    loss = 0.0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                val = 0.0
                                for r in range(target_rank):
                                    val += U[ii*m+jj, r] * V[jj*p+kk, r] * W[kk*n+ii, r]
                                loss += (1.0 - val)**2
                    if loss < best_loss:
                        best_loss = loss
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        
                    if loss < 1e-6:
                        # Try to round
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        if verify_decomp(U_r, V_r, W_r, n, m, p):
                            return U_r, V_r, W_r
                        
        # Final check on best found
        if best_U is not None and best_loss < 1e-3:
            U_r = np.round(best_U * 2) / 2
            V_r = np.round(best_V * 2) / 2
            W_r = np.round(best_W * 2) / 2
            if verify_decomp(U_r, V_r, W_r, n, m, p):
                return U_r, V_r, W_r
                
        return None

    # Search for 3x3x2 decomposition
    U_2, V_2, W_2 = None, None, None
    rank_2 = 18
    for r in [16, 17, 18]:
        if time_left() < 30:
            break
        res = search_decomp(3, 3, 2, r, 10)
        if res is not None:
            U_2, V_2, W_2 = res
            rank_2 = r
            break
            
    if U_2 is None:
        # Fallback to schoolbook for 3x3x2
        rank_2 = 18
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

    # Search for 3x3x3 decomposition
    U_3, V_3, W_3 = None, None, None
    rank_3 = 27
    for r in [23, 24, 25]:
        if time_left() < 40:
            break
        res = search_decomp(3, 3, 3, r, 15)
        if res is not None:
            U_3, V_3, W_3 = res
            rank_3 = r
            break
            
    if U_3 is None:
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

    # Combine decompositions
    # 3x3x8 = 3x3x3 (cols 0-2) + 3x3x3 (cols 3-5) + 3x3x2 (cols 6-7)
    
    # We need to map the indices correctly.
    # For the first 3x3x3 block (cols 0-2 of B):
    # U is same. V uses cols 0-2. W uses rows 0-2 of C (which are cols 0-2 of B in W index? No.)
    # W index is (k, i). k is col of B. i is row of C.
    # For block 1, k is 0,1,2.
    # For block 2, k is 3,4,5.
    # For block 3, k is 6,7.
    
    total_rank = rank_3 + rank_3 + rank_2
    
    U_final = np.zeros((9, total_rank))
    V_final = np.zeros((24, total_rank))
    W_final = np.zeros((24, total_rank))
    
    # Block 1: k=0,1,2
    U_final[:, 0:rank_3] = U_3
    # V_3 is (9, rank_3) for j=0..2, k=0..2
    # V_final index is j*p + k. p=8.
    # So V_3[j, k] goes to V_final[j*8+k]
    for j in range(3):
        for k in range(3):
            V_final[j*8+k, 0:rank_3] = V_3[j*3+k]
    # W_3 is (9, rank_3) for k=0..2, i=0..2
    # W_final index is k*n + i. n=3.
    # W_3[k, i] goes to W_final[k*3+i]
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
