"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    N = n * m
    M = m * p
    P = p * n
    schoolbook_rank = n * m * p
    
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 140.0  # 10s safety margin
    
    # Build the matrix multiplication tensor as a linear system
    # T[i, j, k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For rank r, we need to find U, V, W such that for all (i,j,k):
    # sum_{a=0}^{r-1} sum_{b=0}^{r-1} sum_{c=0}^{r-1} 
    #   U[i*m+j, a] * V[j*p+k, b] * W[k*n+i, c] = delta_{i,j,k}
    
    # The tensor has 1s at positions where i,j,k correspond to valid matrix mult indices
    # For 2x3 * 3x3: n=2, m=3, p=3
    # Tensor T[i,j,k] = 1 if valid, else 0
    # But actually the matrix multiplication tensor is:
    # T[i,j,k] = 1 for all (i,j,k) in [0,n) x [0,m) x [0,p)
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The decomposition requires:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is a trilinear form. For small ranks, we can try to find exact decompositions.
    # Known result: rank of 2x3 * 3x3 matrix multiplication is 17 (not 18).
    # Let's try to find a rank-17 decomposition.
    
    # Strategy: Use the known structure. The rank-17 decomposition for 2x3*3x3
    # can be constructed using the fact that it's related to the 2x2*2x2 case.
    
    # For now, let's implement a search for low-rank decompositions using 
    # alternating least squares or direct optimization.
    
    # First, try to verify if rank 17 is achievable by attempting to find
    # a decomposition with rank 17.
    
    def try_rank(r, max_iters=500, tol=1e-10):
        """Try to find a rank-r decomposition using alternating minimization."""
        if time.time() > deadline - 5:
            return None
        
        # Initialize randomly
        np.random.seed(42)
        U = np.random.randn(N, r) * 0.1
        V = np.random.randn(M, r) * 0.1
        W = np.random.randn(P, r) * 0.1
        
        # Tensor target
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0
        
        best_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (i,j), solve for U[i*m+j, :]
            # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # For fixed (i,j): T[i,j,:] = U[i*m+j,:] @ diag(W[k*n+i,:]) @ V[j*p,:]^T ... 
            # This is complex. Let's use a simpler approach.
            
            # Alternative: Use the fact that for small ranks, we can set up
            # a system of equations.
            pass
        
        return None
    
    # Given the complexity, let's use a known construction for rank 17.
    # The 2x3 * 3x3 matrix multiplication tensor has rank 17.
    # We can construct it from the 2x2 * 2x2 tensor (rank 7) by extension.
    
    # Actually, let's just return the schoolbook rank 18 for now, since
    # finding the exact rank-17 decomposition requires careful construction.
    # The score for rank 18 is 18/18 = 1.0, same as current.
    # But if we can find rank 17, score would be 17/18 ≈ 0.944, which is better.
    
    # Let me try a direct approach: enumerate possible sparse patterns.
    
    # For rank r, we need r triples (u_r, v_r, w_r) where
    # u_r is a vector in R^N, v_r in R^M, w_r in R^P
    # such that T = sum_r u_r ⊗ v_r ⊗ w_r
    
    # For integer/half-integer entries, the values are in {0, ±1/2, ±1, ±3/2, ...}
    
    # Let's try rank 17 with a structured approach.
    
    # Known: The rank of matrix multiplication <n,m,p> satisfies
    # R(2,3,3) <= 17 (Landsberg's bound or explicit construction)
    
    # Let me construct the rank-17 decomposition explicitly.
    # Based on the literature, the 2x3*3x3 tensor can be decomposed as follows:
    
    # Use the fact that 2x3*3x3 is related to 3x3*3x3 which has rank 23.
    # Actually, let's just use a numerical search.
    
    def numerical_search(r, timeout=30):
        """Numerical search for rank-r decomposition."""
        t0 = time.time()
        np.random.seed(123)
        
        # Initialize with small random values
        U = np.random.randn(N, r) * 0.05
        V = np.random.randn(M, r) * 0.05
        W = np.random.randn(P, r) * 0.05
        
        # Target tensor
        T = np.ones((n, m, p))
        
        for it in range(2000):
            if time.time() - t0 > timeout:
                break
            
            # Compute current tensor
            T_cur = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_cur[i, j, k] = np.sum(
                            U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                        )
            
            err = np.linalg.norm(T_cur - T)
            if err < 1e-12:
                return U, V, W, err
            
            # Gradient step
            lr = 0.01 / (1 + it * 0.001)
            
            # Gradient of 0.5 * ||T_cur - T||^2 with respect to U, V, W
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        diff = T_cur[i, j, k] - T[i, j, k]
                        grad_U[i*m+j, :] += diff * V[j*p+k, :] * W[k*n+i, :]
                        grad_V[j*p+k, :] += diff * U[i*m+j, :] * W[k*n+i, :]
                        grad_W[k*n+i, :] += diff * U[i*m+j, :] * V[j*p+k, :]
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        T_cur = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_cur[i, j, k] = np.sum(
                        U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    )
        err = np.linalg.norm(T_cur - T)
        return U, V, W, err
    
    # Try rank 17
    for r in [17, 16, 15]:
        if time.time() > deadline - 10:
            break
        
        result = numerical_search(r, timeout=min(30, deadline - time.time() - 10))
        if result is not None:
            U, V, W, err = result
            if err < 1e-10:
                # Convert to exact fractions if possible
                # Round to nearest half-integer
                U_exact = np.round(U * 2) / 2
                V_exact = np.round(V * 2) / 2
                W_exact = np.round(W * 2) / 2
                
                # Verify
                T_check = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_check[i, j, k] = np.sum(
                                U_exact[i*m+j, :] * V_exact[j*p+k, :] * W_exact[k*n+i, :]
                            )
                
                if np.allclose(T_check, np.ones((n, m, p)), atol=1e-10):
                    return {
                        "U": U_exact.tolist(),
                        "V": V_exact.tolist(),
                        "W": W_exact.tolist()
                    }
    
    # Fallback to schoolbook
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
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
