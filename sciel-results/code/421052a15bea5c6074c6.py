"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130  # 20s safety margin
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    def schoolbook():
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
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        # Tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For matrix mult: T should be 1 if (i,j,k) corresponds to A[i,j]*B[j,k] contributing to C[i,k]
        # Actually the tensor for matmul: for each (i,j,k), the coefficient is 1
        # We need: for all i,j,k: sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    def verify_numpy(U, V, W):
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        # Check: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    def round_to_half(vals):
        return [round(v * 2) / 2 for v in vals]
    
    def try_rank(rank, time_budget):
        """Try to find a rank decomposition with given rank."""
        start = time.time()
        U_init = np.zeros((n*m, rank))
        V_init = np.zeros((m*p, rank))
        W_init = np.zeros((p*n, rank))
        
        # Initialize with random values
        np.random.seed(42)
        U_init = np.random.randn(n*m, rank) * 0.5
        V_init = np.random.randn(m*p, rank) * 0.5
        W_init = np.random.randn(p*n, rank) * 0.5
        
        for attempt in range(20):
            if time.time() > deadline or time.time() - start > time_budget:
                break
            
            U_cur = U_init.copy()
            V_cur = V_init.copy()
            W_cur = W_init.copy()
            
            # Simple iterative refinement
            lr = 0.01
            for step in range(200):
                if time.time() > deadline:
                    break
                # Compute gradients (simplified)
                # We want sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
                U_grad = np.zeros_like(U_cur)
                V_grad = np.zeros_like(V_cur)
                W_grad = np.zeros_like(W_cur)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_cur[i*m+j], V_cur[j*p+k] * W_cur[k*n+i])
                            err = val - 1
                            U_grad[i*m+j] += err * (V_cur[j*p+k] * W_cur[k*n+i])
                            V_grad[j*p+k] += err * (U_cur[i*m+j] * W_cur[k*n+i])
                            W_grad[k*n+i] += err * (U_cur[i*m+j] * V_cur[j*p+k])
                
                U_cur -= lr * U_grad
                V_cur -= lr * V_grad
                W_cur -= lr * W_grad
                
                # Check convergence
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_cur[i*m+j], V_cur[j*p+k] * W_cur[k*n+i])
                            max_err = max(max_err, abs(val - 1))
                if max_err < 1e-8:
                    break
            
            # Round to half-integers and verify
            U_rounded = [[round(U_cur[r,c]*2)/2 for c in range(rank)] for r in range(n*m)]
            V_rounded = [[round(V_cur[r,c]*2)/2 for c in range(rank)] for r in range(m*p)]
            W_rounded = [[round(W_cur[r,c]*2)/2 for c in range(rank)] for r in range(p*n)]
            
            if verify(U_rounded, V_rounded, W_rounded):
                return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
            
            # Re-initialize for next attempt
            np.random.seed(42 + attempt * 1000)
            U_init = np.random.randn(n*m, rank) * 0.5
            V_init = np.random.randn(m*p, rank) * 0.5
            W_init = np.random.randn(p*n, rank) * 0.5
        
        return None
    
    # Try to find lower rank decompositions
    # Known bounds: M_{2,4,5} rank is between 34 and 40
    # Try ranks from 34 to 39
    for rank in range(34, 40):
        if time.time() > deadline:
            break
        time_budget = min(15, deadline - time.time())
        if time_budget < 3:
            break
        result = try_rank(rank, time_budget)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
