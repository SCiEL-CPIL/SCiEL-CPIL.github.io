"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, trials=1000, seed=42):
    """Verify that the decomposition exactly reconstructs matrix multiplication."""
    rng = np.random.RandomState(seed)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C = A @ B
        
        # Reconstruct using tensor decomposition
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # C[i,k] = sum_j T[i,j,k] = sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        
        C_recon = np.zeros((n, p))
        for c in range(rank):
            # For each c, we have vectors u_c (n*m), v_c (m*p), w_c (p*n)
            # u_c[i*m+j], v_c[j*p+k], w_c[k*n+i]
            # Contribution to C[i,k] from this c: sum_j u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    C_recon[i,k] += s
        
        if not np.allclose(C, C_recon, atol=1e-10):
            return False
    return True

def _try_rank(r, n, m, p, deadline):
    """Try to find a decomposition of given rank using structured approach."""
    # For small ranks, try specific structured decompositions
    
    # Strategy: Use the fact that for each j, we have A[:,j] (n-vector) and B[j,:] (p-vector)
    # The tensor is sum_j (A[:,j] ⊗ e_j ⊗ B[j,:]) in some sense
    
    # Try rank = n*p + m - 1 type constructions
    # Or try to find if rank r works with a parameterized family
    
    # Simple approach: try to build U, V, W with specific patterns
    # For rank r, we can try:
    # U[i*m+j, c] = a_{i,j,c}
    # V[j*p+k, c] = b_{j,k,c}  
    # W[k*n+i, c] = w_{k,i,c}
    
    # The constraint is: for all i,j,k: sum_c a_{i,j,c} * b_{j,k,c} * w_{k,i,c} = 1 (if we normalize)
    # Actually T[i,j,k] = delta structure... 
    
    # Let me try a different approach: use known results
    # For (2,3,5), try rank 11, 12, etc.
    
    # Constructive attempt for rank 11:
    if r >= 11:
        # Try: use 10 terms for the "main" part and 1 for correction
        # Or use a bilinear decomposition approach
        
        # Let's try to construct explicitly
        # For n=2, m=3, p=5:
        # We need sum_c U[i*3+j,c] * V[j*5+k,c] * W[k*2+i,c] to give the MM tensor
        
        # Try rank 11 with specific structure
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        
        # Idea: For each pair (i,k), we want to capture the dependence on j
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] should satisfy: the "fiber" in j direction encodes A[i,j]*B[j,k]
        
        # Try: use r = n*p + m - 1 = 10 + 3 - 1 = 12? Or n*p + 1 = 11?
        
        # Let me try a greedy/structured approach
        # For now, return None to try other ranks
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # First, try to find minimal rank with known constructions
    # For (2,3,5), the schoolbook rank is 30
    # Known lower bounds and constructions suggest rank around 11-15
    
    # Try ranks from low to high
    for r in range(10, 31):
        if time.time() > deadline - 5:
            break
        result = _try_rank(r, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p, trials=500):
                return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
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
# EVOLVE-BLOCK-END
