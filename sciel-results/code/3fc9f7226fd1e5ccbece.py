"""Program for matrix multiplication 3x3 by 3x6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]
DEADLINE = 120.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # First, create the schoolbook decomposition as a safe fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Build the multiplication tensor T[i,j,k] where output[i,k] = sum_j A[i,j]*B[j,k]
    # For matrix mult, T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is T_{i,j,k} where i is row of A/C, j is col of A/row of B, k is col of B/C
    
    # We want to find rank r such that T = sum_{t=1}^{r} u_t ⊗ v_t ⊗ w_t
    # where u_t has length n*m, v_t has length m*p, w_t has length p*n
    
    # Actually, let me think about this more carefully.
    # The matrix multiplication tensor T is defined by T[i,j,k] = delta(i,j,k) 
    # where the multiplication is C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 if we're at position (i,j,k) in the tensor, 0 otherwise
    # But actually T[i,j,k] is always 1 for the relevant indices...
    
    # Wait, the tensor T has T[i,j,k] = 1 for all i,j,k in the appropriate ranges
    # No, that's not right either. Let me think again.
    
    # The matrix multiplication tensor is T_{ijk} where the multiplication 
    # C_{ik} = sum_j A_{ij} B_{jk}
    # The tensor is defined such that when we contract T with A and B, we get C
    # T[i,j,k] = delta(i,j,k) doesn't make sense.
    
    # Actually, the multiplication tensor T is T[i,j,k] = 1 for all valid i,j,k
    # No wait. The tensor T satisfies:
    # C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # For standard matrix multiplication, T[i,j,k] = 1 for all i,j,k
    # where i in [0,n), j in [0,m), k in [0,p)
    
    # But that's just the schoolbook. The rank of this tensor is what we want to minimize.
    # A rank-1 decomposition is u ⊗ v ⊗ w where:
    # u[i*m+j] * v[j*p+k] * w[k*n+i] contributes to T[i,j,k]
    
    # For the tensor T[i,j,k] = 1 for all i,j,k (which it is for matrix mult),
    # we need sum_t u_t[i*m+j] * v_t[j*p+k] * w_t[k*n+i] = 1 for all i,j,k
    
    # The schoolbook uses T[i,j,k] = u[i,j] * v[j,k] * w[k,i] with u[i,j]=v[j,k]=w[k,i]=1
    # and each (i,j,k) gets its own term, giving rank n*m*p.
    
    # To reduce rank, we need to share terms across different (i,j,k) indices.
    
    # Let's try to use a known result. The rank of 3x3x3 matrix multiplication is known
    # to be between 19 and 23. Let's see if we can use a rank-23 decomposition for 3x3x3
    # and extend it to 3x3x6.
    
    # For 3x3x3, the tensor T3[i,j,k] = 1 for i,j,k in [0,3)
    # For 3x3x6, T[i,j,k] = 1 for i,j in [0,3), k in [0,6)
    
    # We can think of the 3x6 matrix B as [B1, B2] where B1, B2 are 3x3
    # Then C = A * B = A * [B1, B2] = [A*B1, A*B2]
    # So we need to compute two 3x3x3 multiplications.
    # If we have a rank-r decomposition for 3x3x3, we can use rank 2r for 3x3x6.
    
    # But we might do better by sharing some terms.
    
    # Let's try to find a low-rank decomposition using numerical methods.
    # We'll use ALS (Alternating Least Squares) or similar.
    
    def find_decomposition(target_rank, max_iters=200):
        """Try to find a rank-`target_rank` decomposition using gradient descent."""
        if target_rank >= n * m * p:
            return schoolbook()
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Scale to get initial fit
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Compute current approximation
            # Approx[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
            # We want this to equal 1 for all i,j,k
            
            # Compute the tensor approximation
            approx = np.zeros((n, m, p))
            for t in range(target_rank):
                u_t = U[:, t].reshape(n, m)
                v_t = V[:, t].reshape(m, p)
                w_t = W[:, t].reshape(p, n)
                # approx[i,j,k] += u_t[i,j] * v_t[j,k] * w_t[k,i]
                # This is a bit tricky to vectorize directly
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            approx[i, j, k] += u_t[i, j] * v_t[j, k] * w_t[k, i]
            
            loss = np.sum((approx - 1.0) ** 2)
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-6:
                break
            
            # Gradient descent step
            # d/dU[i*m+j, t] = sum_{j',k} (approx[i,j',k] - 1) * V[j'*p+k, t] * W[k*n+i, t]
            # Actually, let's compute gradients properly
            
            # For U[i*m+j, t]:
            # dLoss/dU[i*m+j, t] = 2 * sum_{j',k} (approx[i,j',k] - 1) * V[j'*p+k, t] * W[k*n+i, t]
            # Wait, no. approx[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
            # So d approx[i,j,k] / d U[i*m+j, t] = V[j*p+k, t] * W[k*n+i, t]
            
            # Let's compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        err = approx[i, j, k] - 1.0
                        for t in range(target_rank):
                            dU[i*m+j, t] += 2 * err * V[j*p+k, t] * W[k*n+i, t]
                            dV[j*p+k, t] += 2 * err * U[i*m+j, t] * W[k*n+i, t]
                            dW[k*n+i, t] += 2 * err * U[i*m+j, t] * V[j*p+k, t]
            
            # Learning rate
            lr = 0.01
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Check if we got a good solution
        if best_loss < 1e-4:
            # Try to find integer/half-integer values
            # Round to nearest 0.5 and check
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            # Verify
            approx = np.zeros((n, m, p))
            for t in range(target_rank):
                u_t = U_rounded[:, t].reshape(n, m)
                v_t = V_rounded[:, t].reshape(m, p)
                w_t = W_rounded[:, t].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            approx[i, j, k] += u_t[i, j] * v_t[j, k] * w_t[k, i]
            
            if np.max(np.abs(approx - 1.0)) < 1e-6:
                # Convert to lists
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
        
        return None
    
    # Try various ranks
    for rank in range(20, n * m * p + 1):
        if time.time() - start_time > DEADLINE - 5:
            break
        result = find_decomposition(rank)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
