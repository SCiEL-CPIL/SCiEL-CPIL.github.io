"""Strategy: Structured Alternating Minimization with Exact Linear Solves for W, and Gradient Updates for U, V."""
import time
import numpy as np

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin

    # 1. Schoolbook fallback (Rank 28)
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, col] = 1
                    V[k * p + j, col] = 1
                    W[j * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, r

    best_result, best_rank = schoolbook()
    
    # Target rank 25
    target_rank = 25
    found = False

    # Helper to reshape for tensor operations
    def get_tensor(U, V, W):
        r = U.shape[1]
        u_r = U.T.reshape(n, m, r)
        v_r = V.T.reshape(m, p, r)
        w_r = W.T.reshape(p, n, r)
        # T[i,j,k] = sum_r U[i,k,r] V[k,j,r] W[j,i,r]
        T = np.einsum('ikr,kjr,jir->ijk', u_r, v_r, w_r)
        return T

    def verify(U, V, W):
        T = get_tensor(U, V, W)
        return np.allclose(T, 1.0, atol=1e-8)

    # 2. Alternating Minimization
    # We will try to find U, V, W of rank 25.
    # Initialize U, V with random sparse patterns.
    
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 10: break
        
        np.random.seed(1000 + restart)
        
        # Initialize U and V with small random values
        # Using a mix of 0s and 1s to encourage sparsity
        U = np.random.choice([0, 1, -1], size=(n*m, target_rank)) * 0.5
        V = np.random.choice([0, 1, -1], size=(m*p, target_rank)) * 0.5
        
        # Ensure some non-zero columns to avoid trivial zero solutions
        for c in range(target_rank):
            if np.all(U[:, c] == 0):
                U[np.random.randint(n*m), c] = 1.0
            if np.all(V[:, c] == 0):
                V[np.random.randint(m*p), c] = 1.0
                
        # Initialize W to zeros
        W = np.zeros((p*n, target_rank))
        
        # Iterate
        for it in range(50):
            if time.time() > deadline - 5: break
            
            # Step 1: Update W given U, V
            # Equation: sum_r W[j*n+i, r] * (sum_k U[i*m+k, r] * V[k*p+j, r]) = 1
            # Let A_ji_r = sum_k U[i*m+k, r] * V[k*p+j, r]
            # We need sum_r W[j*n+i, r] * A_ji_r = 1 for all i, j.
            # This is a linear system for W.
            # Variables: W[j*n+i, r] for all j, i, r.
            # Equations: 1 for each (i, j). Note: the tensor identity requires it for all (i,j,k).
            # Wait, the tensor identity is T[i,j,k] = 1.
            # T[i,j,k] = sum_r U[i*m+k, r] V[k*p+j, r] W[j*n+i, r].
            # This must hold for ALL i, j, k.
            # So for a fixed (i, j), we have m equations (one for each k).
            # Let's build the linear system for W.
            
            # Number of equations: n * p * m = 28
            # Number of unknowns: (p*n) * target_rank = 14 * 25 = 350
            # We can solve using least squares or by finding a particular solution.
            
            # Matrix M of shape (28, 350)
            # Row index: (i * p + j) * m + k
            # Col index: (j * n + i) * target_rank + r
            # Entry: U[i*m+k, r] * V[k*p+j, r]
            
            M = np.zeros((n * p * m, p * n * target_rank))
            for i in range(n):
                for j in range(p):
                    for k in range(m):
                        row = (i * p + j) * m + k
                        for r in range(target_rank):
                            col = (j * n + i) * target_rank + r
                            M[row, col] = U[i * m + k, r] * V[k * p + j, r]
            
            b = np.ones(n * p * m)
            
            try:
                # Solve M W_flat = b
                # Since underdetermined, use lstsq to find minimum norm solution
                W_flat, residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
                W = W_flat.reshape(p * n, target_rank)
            except Exception:
                break
            
            # Step 2: Update U and V
            # We can use gradient descent on the residual error.
            # Error E = sum (T[i,j,k] - 1)^2
            # T[i,j,k] = sum_r U[i*m+k, r] V[k*p+j, r] W[j*n+i, r]
            
            # Compute gradient for U
            # dE/dU[i*m+k, r] = 2 * (T[i,j,k] - 1) * V[k*p+j, r] * W[j*n+i, r]
            # Summed over j? No, T is defined for each (i,j,k).
            # The error is sum over all i,j,k of (T[i,j,k]-1)^2.
            # So grad_U[i*m+k, r] = 2 * sum_j (T[i,j,k]-1) * V[k*p+j, r] * W[j*n+i, r]
            
            T = get_tensor(U, V, W)
            residual = T - 1.0
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            
            for i in range(n):
                for k in range(m):
                    for r in range(target_rank):
                        # Sum over j
                        # T[i,j,k] - 1
                        # V[k*p+j, r]
                        # W[j*n+i, r]
                        for j in range(p):
                            grad_U[i*m+k, r] += 2 * residual[i, j, k] * V[k*p+j, r] * W[j*n+i, r]
                            
            for k in range(m):
                for j in range(p):
                    for r in range(target_rank):
                        # Sum over i
                        for i in range(n):
                            grad_V[k*p+j, r] += 2 * residual[i, j, k] * U[i*m+k, r] * W[j*n+i, r]
            
            # Update U and V with small step
            lr = 0.01
            U -= lr * grad_U
            V -= lr * grad_V
            
            # Periodic sparsification/rounding to help convergence to half-integers
            if it % 5 == 0:
                # Snap values close to 0, 0.5, 1 to exact values
                U[np.abs(U) < 0.01] = 0.0
                U[np.abs(U - 0.5) < 0.01] = 0.5
                U[np.abs(U + 0.5) < 0.01] = -0.5
                U[np.abs(U - 1.0) < 0.01] = 1.0
                U[np.abs(U + 1.0) < 0.01] = -1.0
                
                V[np.abs(V) < 0.01] = 0.0
                V[np.abs(V - 0.5) < 0.01] = 0.5
                V[np.abs(V + 0.5) < 0.01] = -0.5
                V[np.abs(V - 1.0) < 0.01] = 1.0
                V[np.abs(V + 1.0) < 0.01] = -1.0

            # Check convergence
            if np.max(np.abs(residual)) < 1e-6:
                # Found a solution
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2.0
                V_rounded = np.round(V * 2) / 2.0
                W_rounded = np.round(W * 2) / 2.0
                
                if verify(U_rounded, V_rounded, W_rounded):
                    best_result = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
