"""Optimized tensor decomposition for 3x3x5 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W decompose the matmul tensor exactly."""
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = 1 for all i,j,k (matmul tensor is just 1s)
    # Actually the matmul tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The decomposition requires: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, the standard form is: C_ik = sum_j A_ij B_jk
    # Tensor form: T[i,j,k] such that C_ik = sum_j A_ij T[i,j,k] B_jk, so T[i,j,k]=1
    # Decomposition: T[i,j,k] = sum_c u_ic * v_jc * w_kc where:
    #   U row index = i*m+j (so U[i*m+j, c] corresponds to u for input A)
    #   V row index = j*p+k (so V[j*p+k, c] corresponds to v for input B)  
    #   W row index = k*n+i (so W[k*n+i, c] corresponds to w for output C)
    # But the convention says W[k*n+i, r], so output index is (k, i) meaning C_ik
    # Let me re-derive: C_ik = sum_j A_ij B_jk
    # In tensor form with indices (i, j, k): the bilinear form is sum_{i,j,k} A_ij B_jk C_ik^*
    # The tensor is T[i,j,k] = 1 for all valid i,j,k
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Check a few entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x3x5 matmul."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # 130s safety margin before 150s kill
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school), dtype=np.float64)
    V_school = np.zeros((m*p, rank_school), dtype=np.float64)
    W_school = np.zeros((p*n, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find lower rank decompositions using numerical optimization
    # We'll attempt ranks from 27 down to 20 (below schoolbook's 45)
    # Strategy: use random initialization + gradient descent on the residual
    
    def try_rank(rank):
        """Try to find a decomposition of given rank via optimization."""
        nonlocal best_rank, best_U, best_V, best_W
        if rank >= best_rank:
            return False
        
        # Multiple random restarts
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Random initialization
            np.random.seed(42 + restart)
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Use simple iterative refinement
            # The constraint is: for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is a system of n*m*p = 45 equations in 3*rank variables (with symmetry breaking)
            
            # Let's use a simpler approach: fix some variables and solve linear systems
            # Alternate: fix V and W, solve for U (linear), etc.
            
            lr = 0.01
            for it in range(200):
                if time.time() > deadline - 5.0:
                    break
                
                # Compute residual
                res = 0.0
                # Compute gradient-like update
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            r = 0.0
                            for c in range(rank):
                                r += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err = r - 1.0
                            for c in range(rank):
                                dU[i*m+j, c] += err * V[j*p+k, c] * W[k*n+i, c]
                                dV[j*p+k, c] += U[i*m+j, c] * err * W[k*n+i, c]
                                dW[k*n+i, c] += U[i*m+j, c] * V[j*p+k, c] * err
                            res += err * err
                
                if res < 1e-18:
                    break
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            # Check if we found a valid decomposition
            if _verify(U, V, W, n, m, p, tol=1e-8):
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                return True
        
        return False
    
    # Try decreasing ranks
    for rank in range(44, 19, -1):
        if time.time() > deadline - 10.0:
            break
        try_rank(rank)
    
    # Convert to list-of-lists format as required
    def to_list(mat):
        return [[float(mat[i, j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
