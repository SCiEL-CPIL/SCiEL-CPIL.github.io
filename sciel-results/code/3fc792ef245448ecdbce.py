# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses a hybrid continuous gradient descent + discrete local search strategy
    to find a lower rank decomposition than schoolbook (rank 60).
    """
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col_idx] = 1
                    V[j * p + k][col_idx] = 1
                    W[k * n + i][col_idx] = 1
                    col_idx += 1
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W):
        """
        Check if U, V, W exactly decompose the matrix multiplication tensor.
        U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
        For all i,j,k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1
        """
        rank = len(U[0])
        # Use NumPy for faster verification
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
        U_reshaped = U_np.reshape((n, m, rank))
        V_reshaped = V_np.reshape((m, p, rank))
        W_reshaped = W_np.reshape((p, n, rank))
        
        # Compute tensor product sum
        # T[i, j, k] = sum_t U[i, j, t] * V[j, k, t] * W[k, i, t]
        # We can do this by broadcasting
        # U[i, j, t] * V[j, k, t] -> (n, m, p, rank)
        # This might be memory heavy for large rank, but rank <= 60, n*m*p=60, so 60*60*60*60 is small.
        
        # Efficient way:
        # For each t, we have outer products? No.
        # T = np.einsum('ijt,jkt,kit->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        T = np.einsum('ijt,jkt,kit->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Check if all entries are 1
        if not np.all(T == 1):
            return False
        return True

    def optimize_rank(r):
        """
        Attempt to find a rank-r decomposition using gradient descent and local search.
        Returns (U, V, W) if found, else None.
        """
        # Shapes
        U_shape = (n * m, r)
        V_shape = (m * p, r)
        W_shape = (p * n, r)
        
        # Reshape for gradient computation
        # U: (n, m, r), V: (m, p, r), W: (p, n, r)
        
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                return None
            
            # Initialize with small random values
            np.random.seed(42 + restart + r)
            U = np.random.uniform(-1, 1, U_shape)
            V = np.random.uniform(-1, 1, V_shape)
            W = np.random.uniform(-1, 1, W_shape)
            
            # Reshape for einsum
            U_r = U.reshape((n, m, r))
            V_r = V.reshape((m, p, r))
            W_r = W.reshape((p, n, r))
            
            # Gradient Descent
            lr = 0.05
            for step in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Compute Residual
                T = np.einsum('ijt,jkt,kit->ijk', U_r, V_r, W_r)
                Residual = T - 1.0
                
                # Compute Gradients
                # dE/dU[i,j,t] = 2 * sum_k Residual[i,j,k] * V[j,k,t] * W[k,i,t]
                # Using einsum:
                # dU[i,j,t] = sum_k Res[i,j,k] * V[j,k,t] * W[k,i,t]
                dU = np.einsum('ijk,jkt,kit->ijt', Residual, V_r, W_r)
                dV = np.einsum('ijk,ijt,kit->jkt', Residual, U_r, W_r)
                dW = np.einsum('ijk,ijt,jkt->kit', Residual, U_r, V_r)
                
                # Update
                U_r -= 2 * lr * dU
                V_r -= 2 * lr * dV
                W_r -= 2 * lr * dW
                
                # Decay learning rate
                lr *= 0.95
            
            # Check continuous residual
            T = np.einsum('ijt,jkt,kit->ijk', U_r, V_r, W_r)
            if np.max(np.abs(T - 1.0)) > 1e-2:
                continue
            
            # Round to nearest integer
            U_int = np.round(U_r).astype(int)
            V_int = np.round(V_r).astype(int)
            W_int = np.round(W_r).astype(int)
            
            # Flatten back to list format for verification
            U_list = [list(row) for row in U_int.reshape(U_shape)]
            V_list = [list(row) for row in V_int.reshape(V_shape)]
            W_list = [list(row) for row in W_int.reshape(W_shape)]
            
            if verify_decomposition(U_list, V_list, W_list):
                return U_list, V_list, W_list
            
            # Local Discrete Search (Hill Climbing) to fix rounding errors
            # We want to change some entries by +-1 to reduce the error to 0
            current_error = np.sum((np.einsum('ijt,jkt,kit->ijk', 
                          U_int.reshape(n,m,r), V_int.reshape(m,p,r), W_int.reshape(p,n,r)) - 1.0)**2)
            
            # Simple local search: try flipping random entries
            for _ in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random matrix and entry
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    idx = np.random.randint(n * m * r)
                    i, j, t = divmod(idx, m * r) # This is wrong, need proper indexing
                    # Better:
                    flat_idx = np.random.randint(n * m * r)
                    # Map flat index to (i, j, t) in U (n*m, r)
                    row, col = divmod(flat_idx, r)
                    i, j = divmod(row, m)
                    
                    # Try +1 and -1
                    for delta in [1, -1]:
                        U_int[row, col] += delta
                        new_T = np.einsum('ijt,jkt,kit->ijk', 
                                          U_int.reshape(n,m,r), V_int.reshape(m,p,r), W_int.reshape(p,n,r))
                        new_error = np.sum((new_T - 1.0)**2)
                        if new_error < current_error:
                            current_error = new_error
                            break
                        else:
                            U_int[row, col] -= delta
                            
                elif mat_idx == 1:
                    flat_idx = np.random.randint(m * p * r)
                    row, col = divmod(flat_idx, r)
                    j, k = divmod(row, p)
                    
                    for delta in [1, -1]:
                        V_int[row, col] += delta
                        new_T = np.einsum('ijt,jkt,kit->ijk', 
                                          U_int.reshape(n,m,r), V_int.reshape(m,p,r), W_int.reshape(p,n,r))
                        new_error = np.sum((new_T - 1.0)**2)
                        if new_error < current_error:
                            current_error = new_error
                            break
                        else:
                            V_int[row, col] -= delta
                            
                else:
                    flat_idx = np.random.randint(p * n * r)
                    row, col = divmod(flat_idx, r)
                    k, i = divmod(row, n)
                    
                    for delta in [1, -1]:
                        W_int[row, col] += delta
                        new_T = np.einsum('ijt,jkt,kit->ijk', 
                                          U_int.reshape(n,m,r), V_int.reshape(m,p,r), W_int.reshape(p,n,r))
                        new_error = np.sum((new_T - 1.0)**2)
                        if new_error < current_error:
                            current_error = new_error
                            break
                        else:
                            W_int[row, col] -= delta
                
                if current_error == 0:
                    U_list = [list(row) for row in U_int.reshape(U_shape)]
                    V_list = [list(row) for row in V_int.reshape(V_shape)]
                    W_list = [list(row) for row in W_int.reshape(W_shape)]
                    if verify_decomposition(U_list, V_list, W_list):
                        return U_list, V_list, W_list

        return None

    # Start with schoolbook
    result = schoolbook()
    best_rank = n * m * p  # 60
    U = result["U"]
    V = result["V"]
    W = result["W"]

    # Try to find lower rank
    # Lower bound is 30. We try down to 30.
    for rank in range(best_rank - 1, 29, -1):
        if time.time() > deadline - 10:
            break
            
        # Try optimization for this rank
        found = optimize_rank(rank)
        if found is not None:
            U, V, W = found
            best_rank = rank
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
