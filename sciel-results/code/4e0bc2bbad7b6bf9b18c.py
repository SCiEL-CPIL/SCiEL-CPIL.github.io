"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 100
    
    def schoolbook():
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
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0])
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        
        # Build the multiplication tensor from the decomposition
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We need T[i,j,k] = 1 if this is a valid multiplication term, else 0
        # Actually for matrix multiplication: (AB)[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor T_{i,j,k} should be 1 for all valid (i,j,k) in the multiplication
        
        # Let's verify by checking that the decomposition reproduces matrix multiplication
        # For random matrices A (n x m) and B (m x p), check AB = sum_c U_c * V_c * W_c
        # where U_c is extracted from U, etc.
        
        # Actually, the standard verification: 
        # (AB)[i,k] = sum_j A[i,j] B[j,k]
        # Decomposition: (AB)[i,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) * (sum_j V[j*p+k,c] B[j,k]) ... 
        # No, that's not right either.
        
        # The correct interpretation:
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # The multiplication tensor M[i,j,k] = 1 for all i,j,k (it's the full multiplication tensor)
        # Decomposition: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # So we just need to check that for all (i,j,k):
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        
        # This is expensive for large tensors, so let's use a smarter check
        # with random sampling or direct computation for small sizes
        
        if n * m * p * rank > 1000000:
            # Random sampling
            np.random.seed(42)
            for _ in range(1000):
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
            return True
        else:
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                        if abs(val - 1.0) > 1e-9:
                            return False
            return True
    
    def try_rank_decomposition(rank, max_iters=50, time_budget=60.0):
        """Try to find a rank-`rank` decomposition using iterative refinement."""
        start_time = time.time()
        deadline = start_time + time_budget
        
        # Initialize with random values
        for attempt in range(3):
            np.random.seed(attempt * 12345 + rank)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            best_err = float('inf')
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Compute the current tensor approximation
                # T_approx[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # Target: T_target[i,j,k] = 1 for all i,j,k
                
                # Use alternating least squares approach
                # Fix V and W, solve for U; fix U and W, solve for V; fix U and V, solve for W
                
                # For U: for each (i,j), we need sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c] summed over k... )
                # This is tricky because the coupling is complex.
                
                # Alternative: use gradient descent or a simpler iterative method
                
                # Let's use a simple approach: compute the residual and update
                # Actually, let's try a different strategy: start from schoolbook and try to merge terms
                
                if it % 10 == 0:
                    # Compute error
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                                err += (val - 1.0) ** 2
                    if err < best_err:
                        best_err = err
                    
                    if best_err < 1e-6:
                        # Found a solution, verify exactly
                        U_list = [[float(U[i*m+j][c]) for c in range(rank)] for i in range(n) for j in range(m)]
                        V_list = [[float(V[j*p+k][c]) for c in range(rank)] for j in range(m) for k in range(p)]
                        W_list = [[float(W[k*n+i][c]) for c in range(rank)] for k in range(p) for i in range(n)]
                        if verify(U_list, V_list, W_list):
                            return {"U": U_list, "V": V_list, "W": W_list}
                
                # Simple gradient step (very basic)
                lr = 0.01 / (1 + it * 0.1)
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        for c in range(rank):
                            grad = 0.0
                            for k in range(p):
                                val = sum(U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc] for cc in range(rank))
                                grad += 2 * (val - 1.0) * V[j*p+k][c] * W[k*n+i][c]
                            U[i*m+j][c] -= lr * grad
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            grad = 0.0
                            for i in range(n):
                                val = sum(U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc] for cc in range(rank))
                                grad += 2 * (val - 1.0) * U[i*m+j][c] * W[k*n+i][c]
                            V[j*p+k][c] -= lr * grad
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        for c in range(rank):
                            grad = 0.0
                            for j in range(m):
                                val = sum(U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc] for cc in range(rank))
                                grad += 2 * (val - 1.0) * U[i*m+j][c] * V[j*p+k][c]
                            W[k*n+i][c] -= lr * grad
            
            if best_err < 1e-6:
                U_list = [[float(U[i*m+j][c]) for c in range(rank)] for i in range(n) for j in range(m)]
                V_list = [[float(V[j*p+k][c]) for c in range(rank)] for j in range(m) for k in range(p)]
                W_list = [[float(W[k*n+i][c]) for c in range(rank)] for k in range(p) for i in range(n)]
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try various ranks, starting from lower
    # The theoretical lower bound for 4x5x5 is at least max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # But it could be higher. Let's try from 25 upward.
    
    elapsed = time.time()
    remaining = 120.0 - elapsed - 5.0  # 5 second safety margin
    
    for rank in range(25, min(schoolbook_rank, 60)):
        if time.time() - elapsed > remaining:
            break
        
        budget = min(remaining * 0.5, 30.0)
        result = try_rank_decomposition(rank, max_iters=30, time_budget=budget)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
