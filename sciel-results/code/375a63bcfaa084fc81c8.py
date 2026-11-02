import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p  # 72
    
    def make_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # The decomposition condition: for all (i,j,k) in [n]x[m]x[p]:
    # sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # This is a system of n*m*p = 72 polynomial equations in R*(n*m + m*p + p*n) variables.
    #
    # Strategy: Try ranks from lower bound upward. For each rank, try to find a solution
    # using a structured search or optimization.
    
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    lower_bound = max(n * m, m * p, p * n)
    
    # Try a few candidate ranks with structured approaches
    # First, let's try to see if we can find a decomposition using a clever structure.
    
    # Approach: Use the fact that the tensor is "constant" (all ones).
    # If we set U[i*m+j, r] = a_r, V[j*p+k, r] = b_r, W[k*n+i, r] = c_r for all i,j,k,
    # then we need sum_r a_r * b_r * c_r = 1, but this only gives one equation, not 72.
    # So we need more structure.
    
    # Better approach: Try to find a decomposition where the rank-1 tensors have
    # specific patterns that cover the (i,j,k) space.
    
    # Let's try a randomized local search approach for various ranks.
    # We'll use a smooth optimization approach.
    
    def try_rank(R, max_iterations=5000, seed=42):
        """Try to find a rank-R decomposition using gradient descent on a smooth objective."""
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n * m, R) * 0.5
        V = np.random.randn(m * p, R) * 0.5
        W = np.random.randn(p * n, R) * 0.5
        
        # Objective: minimize sum over (i,j,k) of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
        # We'll use a simple gradient descent with momentum.
        
        lr = 0.01
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Compute gradients
            # L = sum_{i,j,k} (S[i,j,k] - 1)^2 where S[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            # dL/dU[i*m+j,r] = 2 * (S[i,j,k] - 1) * V[j*p+k,r] * W[k*n+i,r]
            # But S depends on all r, so we need to sum over k for each (i,j,r)
            
            # For efficiency, let's compute the residuals first
            # Residual[i,j,k] = S[i,j,k] - 1
            Residual = np.ones((n, m, p))
            for r in range(R):
                # U[:, r] has shape (n*m,), V[:, r] has shape (m*p,), W[:, r] has shape (p*n,)
                # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # This is a bit tricky to vectorize directly
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            Residual[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                Residual -= 1  # This is wrong, should be outside the r loop
            
            # Let me redo this more carefully
            Residual = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(R):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        Residual[i, j, k] = val - 1.0
            
            loss = np.sum(Residual ** 2)
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < 1e-12:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res = Residual[i, j, k]
                        for r in range(R):
                            dU[i * m + j, r] += 2 * res * V[j * p + k, r] * W[k * n + i, r]
                            dV[j * p + k, r] += 2 * res * U[i * m + j, r] * W[k * n + i, r]
                            dW[k * n + i, r] += 2 * res * U[i * m + j, r] * V[j * p + k, r]
            
            # Update with momentum
            vU = momentum * vU - lr * dU
            vV = momentum * vV - lr * dV
            vW = momentum * vW - lr * dW
            
            U += vU
            V += vV
            W += vW
        
        if best_loss < 1e-8:
            return best_U, best_V, best_W
        return None
    
    # Try ranks from lower bound upward
    for R in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline:
            break
        
        # Try a few random seeds
        for seed in range(5):
            if time.time() > deadline:
                break
            
            result = try_rank(R, max_iterations=2000, seed=seed + 100)
            if result is not None:
                U, V, W = result
                # Verify exactness
                ok = True
                for i in range(n):
                    if not ok:
                        break
                    for j in range(m):
                        if not ok:
                            break
                        for k in range(p):
                            val = 0.0
                            for r in range(R):
                                val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            if abs(val - 1.0) > 1e-6:
                                ok = False
                                break
                
                if ok:
                    # Round to nearest half-integer if close
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Check if rounded version still works
                    ok2 = True
                    for i in range(n):
                        if not ok2:
                            break
                        for j in range(m):
                            if not ok2:
                                break
                            for k in range(p):
                                val = 0.0
                                for r in range(R):
                                    val += U_rounded[i * m + j, r] * V_rounded[j * p + k, r] * W_rounded[k * n + i, r]
                                if abs(val - 1.0) > 1e-9:
                                    ok2 = False
                                    break
                    
                    if ok2:
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                    else:
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
    
    # Fallback to schoolbook
    return make_schoolbook()
# EVOLVE-BLOCK-END
