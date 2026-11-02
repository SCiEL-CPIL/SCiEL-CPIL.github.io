import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # The problem decomposes into p/m = 3 blocks of 2x3x3 multiplication
    # We search for a low-rank decomposition of 2x3x3
    # Then replicate it 3 times
    
    # Target tensor for 2x3x3
    # T[i,j,k] = 1
    # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    
    # Try ranks from 11 upwards
    best_U = None
    best_V = None
    best_W = None
    best_rank = 18 # schoolbook rank for 2x3x3 is 18
    
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    for R in [11, 12, 13, 14, 15, 16, 17]:
        if time.time() > deadline - 10:
            break
            
        found = False
        for restart in range(50): # 50 restarts per rank
            if time.time() > deadline - 5:
                break
                
            np.random.seed(42 + restart)
            # Initialize with small random values
            U = np.random.randn(2*3, R) * 0.1
            V = np.random.randn(3*3, R) * 0.1
            W = np.random.randn(3*2, R) * 0.1
            
            # Gradient descent
            lr = 0.01
            for step in range(1000):
                if time.time() > deadline - 2:
                    break
                    
                # Compute error
                # T[i,j,k] = sum_r U[i*3+j, r] V[j*3+k, r] W[k*2+i, r]
                # We can vectorize this partially
                # For each r, compute outer product?
                # E[i,j,k] = sum_r ...
                
                # Let's compute the sum
                # U is 6xR, V is 9xR, W is 6xR
                # We need sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                
                # This is a tensor contraction.
                # We can loop over r
                # E = np.zeros((2,3,3))
                # for r in range(R):
                #     u_r = U[:, r].reshape(2, 3)
                #     v_r = V[:, r].reshape(3, 3)
                #     w_r = W[:, r].reshape(3, 2)
                #     # term is u_r[i,j] * v_r[j,k] * w_r[k,i]
                #     # This is not a standard matmul.
                #     # We can compute this as:
                #     # For fixed i, k: sum_j u[i,j] v[j,k] w[k,i]
                #     # This is (u * v)_{i,k} * w_{k,i}? No.
                #     # Let's just loop or use einsum
                #     E += np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
                
                # Using einsum for the whole sum is faster if we reshape
                # U_r is (2,3,1), V_r is (1,3,3), W_r is (3,1,2)?
                # Let's stick to a simple loop for clarity in the inner step, 
                # but optimize if needed.
                
                E = np.zeros((2, 3, 3))
                for r in range(R):
                    u_r = U[:, r].reshape(2, 3)
                    v_r = V[:, r].reshape(3, 3)
                    w_r = W[:, r].reshape(3, 2)
                    E += np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
                
                error = np.sum((E - 1)**2)
                if error < 1e-6:
                    found = True
                    break
                
                # Compute gradients
                # dE/dU[i,j,r] = V[j,k,r] W[k,i,r] (summed over k? No, E[i,j,k] depends on U[i,j,r])
                # E[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # dE[i,j,k]/dU[i,j,r] = V[j,k,r] W[k,i,r]
                # dJ/dU[i,j,r] = 2 * (E[i,j,k] - 1) * V[j,k,r] W[k,i,r] (summed over k)
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for r in range(R):
                    u_r = U[:, r].reshape(2, 3)
                    v_r = V[:, r].reshape(3, 3)
                    w_r = W[:, r].reshape(3, 2)
                    
                    # Reshape error to (2,3,3)
                    # dU[i,j,r] = sum_k (E[i,j,k]-1) * V[j,k,r] * W[k,i,r]
                    # dV[j,k,r] = sum_i (E[i,j,k]-1) * U[i,j,r] * W[k,i,r]
                    # dW[k,i,r] = sum_j (E[i,j,k]-1) * U[i,j,r] * V[j,k,r]
                    
                    # Use einsum
                    # dU_r = np.einsum('ijk,jk,ki->ij', E-1, v_r, w_r)
                    # dV_r = np.einsum('ijk,ij,ki->jk', E-1, u_r, w_r)
                    # dW_r = np.einsum('ijk,ij,jk->ki', E-1, u_r, v_r)
                    
                    dU[:, r] = np.einsum('ijk,jk,ki->ij', E-1, v_r, w_r).flatten()
                    dV[:, r] = np.einsum('ijk,ij,ki->jk', E-1, u_r, w_r).flatten()
                    dW[:, r] = np.einsum('ijk,ij,jk->ki', E-1, u_r, v_r).flatten()
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Adaptive learning rate?
                lr *= 0.999
            
            if found:
                best_U = U
                best_V = V
                best_W = W
                best_rank = R
                break
        
        if found:
            break

    if best_U is None:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    else:
        # Construct full decomposition by repeating the 2x3x3 solution 3 times
        # The blocks correspond to k in {0,1,2}, {3,4,5}, {6,7,8}
        # For block b (0,1,2), the k index is k' + 3*b
        # U is the same for all blocks?
        # In the decomposition of M_2,3,3, U depends on i,j.
        # In M_2,3,9, U depends on i,j.
        # The rank is 3 * best_rank.
        # The columns 0..best_rank-1 correspond to block 0 (k=0,1,2)
        # The columns best_rank..2*best_rank-1 correspond to block 1 (k=3,4,5)
        # The columns 2*best_rank..3*best_rank-1 correspond to block 2 (k=6,7,8)
        
        total_rank = 3 * best_rank
        U_full = [[0] * total_rank for _ in range(n * m)]
        V_full = [[0] * total_rank for _ in range(m * p)]
        W_full = [[0] * total_rank for _ in range(p * n)]
        
        for b in range(3):
            offset = b * best_rank
            for r in range(best_rank):
                col = offset + r
                # U is the same for all blocks
                for i in range(n):
                    for j in range(m):
                        U_full[i * m + j][col] = best_U[i * m + j, r]
                
                # V depends on j and k'
                for j in range(m):
                    for kp in range(3): # k' in 0..2
                        k = kp + b * 3
                        V_full[j * p + k][col] = best_V[j * 3 + kp, r]
                
                # W depends on k' and i
                for kp in range(3):
                    k = kp + b * 3
                    for i in range(n):
                        W_full[k * n + i][col] = best_W[kp * n + i, r]
        
        return {"U": U_full, "V": V_full, "W": W_full}
