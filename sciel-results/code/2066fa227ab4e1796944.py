import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Initialize with schoolbook solution
    best_rank = rank_schoolbook
    best_U = [[0] * rank_schoolbook for _ in range(n * m)]
    best_V = [[0] * rank_schoolbook for _ in range(m * p)]
    best_W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        # Check dimensions
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        for row in U:
            if len(row) != rank:
                return False
        for row in V:
            if len(row) != rank:
                return False
        for row in W:
            if len(row) != rank:
                return False
        
        # Convert to numpy arrays
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For matrix multiplication, T[i,j,k] should be 1 if this corresponds to
        # the (i,k) entry of A*B where A is n x m and B is m x p
        # Actually, the tensor for matrix multiplication has T[i,j,k] = 1 for all i,j,k
        # because (AB)[i,k] = sum_j A[i,j] * B[j,k]
        # So each term A[i,j]*B[j,k] contributes to (AB)[i,k]
        
        # Let's verify by checking that the decomposition reproduces the tensor
        # The tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        # Compute the decomposed tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i * m + j]
                    v_val = V_arr[j * p + k]
                    w_val = W_arr[k * n + i]
                    tensor_val = np.sum(u_val * v_val * w_val)
                    if abs(tensor_val - 1.0) > 1e-9:
                        return False
        return True
    
    def try_rank(r):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline:
            return None
        
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Use gradient descent to minimize the error
            # The objective is: sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            # This is a non-convex problem, so we need multiple restarts
            
            lr = 0.01
            for step in range(1000):
                if time.time() > deadline:
                    break
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            
                            # Compute the current tensor value
                            tensor_val = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                            error = tensor_val - 1.0
                            
                            # Gradients
                            dU[u_idx] += 2 * error * V[v_idx] * W[w_idx]
                            dV[v_idx] += 2 * error * U[u_idx] * W[w_idx]
                            dW[w_idx] += 2 * error * U[u_idx] * V[v_idx]
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence
                max_grad = max(np.max(np.abs(dU)), np.max(np.abs(dV)), np.max(np.abs(dW)))
                if max_grad < 1e-8:
                    break
            
            # Check if we found a valid solution
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        tensor_val = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                        error += (tensor_val - 1.0) ** 2
            
            if error < 1e-6:
                # Round to nearest integers or half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify the rounded solution
                if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), r):
                    return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
        
        return None
    
    # Try ranks below schoolbook
    for r in range(rank_schoolbook - 1, 10, -1):
        if time.time() > deadline:
            break
        
        result = try_rank(r)
        if result is not None:
            best_rank = r
            best_U, best_V, best_W = result
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
