import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Function to create the target tensor T
    # T[a, b, c] = 1 if indices match, 0 otherwise
    # a = i*m + j, b = j*p + k, c = k*n + i
    # Match condition: j from a == j from b, k from b == k from c, i from c == i from a
    def get_target_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T = get_target_tensor(n, m, p)
    
    # Function to check if a decomposition is exact
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    def check_decomposition(U, V, W, T):
        # Compute sum_r U[:,r] V[:,r] W[:,r]
        # This is a tensor contraction
        # Result R[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        R = np.einsum('ar,br,cr->abc', U, V, W)
        # Check if R matches T
        # Since we snap to 0.5, we can check exact equality
        if np.allclose(R, T, atol=1e-9):
            return True
        return False

    # Function to snap to nearest 0.5
    def snap_to_half(X):
        return np.round(X * 2) / 2

    # Function to try to find a decomposition of a given rank
    def find_decomposition(rank, max_iters=5000, learning_rate=0.01):
        # Initialize with random values
        # Scale initialization to help convergence
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Gradient descent
        # Loss: || T - sum_r U_r V_r W_r ||_F^2
        # Gradient w.r.t U: 2 * (T - R) * (V W^T) ?
        # Let R = sum_r U[:,r] V[:,r] W[:,r]
        # dL/dU[a,r] = -2 * sum_{b,c} (T[a,b,c] - R[a,b,c]) * V[b,r] * W[c,r]
        
        for it in range(max_iters):
            R = np.einsum('ar,br,cr->abc', U, V, W)
            error = T - R
            if np.linalg.norm(error) < 1e-10:
                break
                
            # Compute gradients
            # dU[a,r] = -2 * sum_{b,c} error[a,b,c] V[b,r] W[c,r]
            # dV[b,r] = -2 * sum_{a,c} error[a,b,c] U[a,r] W[c,r]
            # dW[c,r] = -2 * sum_{a,b} error[a,b,c] U[a,r] V[b,r]
            
            # Efficient computation using einsum
            # dU = -2 * einsum('abc,br,cr->ar', error, V, W)
            # dV = -2 * einsum('abc,ar,cr->br', error, U, W)
            # dW = -2 * einsum('abc,ar,br->cr', error, U, V)
            
            dU = -2 * np.einsum('abc,br,cr->ar', error, V, W)
            dV = -2 * np.einsum('abc,ar,cr->br', error, U, W)
            dW = -2 * np.einsum('abc,ar,br->cr', error, U, V)
            
            # Update
            U -= learning_rate * dU
            V -= learning_rate * dV
            W -= learning_rate * dW
            
            # Optional: snap periodically? No, keep continuous for gradient flow.
            
        # Snap to half-integers
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        
        # Check if exact
        if check_decomposition(U_s, V_s, W_s, T):
            return U_s, V_s, W_s
        return None

    # Try ranks from 20 up to schoolbook_rank - 1
    # We know rank 20 is possible for 2x3x4
    for rank in range(20, schoolbook_rank):
        # Try a few times for each rank
        for attempt in range(3):
            start_time = time.time()
            result = find_decomposition(rank, max_iters=2000, learning_rate=0.05)
            if result is not None:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            # Check time
            if time.time() - start_time > 10:
                break

    # Fallback to schoolbook
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
