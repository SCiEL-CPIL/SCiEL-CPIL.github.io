"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 3]

def _build_tensor(n, m, p):
    """Return the matrix multiplication tensor T[i,j,k] with shape (n, m, p)."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matmul tensor."""
    rank = U.shape[1]
    # Contract: sum_c U[i*m+j, c] * V[j*p+k, c] should equal 1 for all i,j,k
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = np.array(U).reshape(n, m, rank)
    V3 = np.array(V).reshape(m, p, rank)
    W3 = np.array(W).reshape(p, n, rank)
    # For each (i,j,k): sum_c U3[i,j,c] * V3[j,k,c] should be 1
    # And W3[k,i,c] is the output coefficient (should be 1 for the standard decomposition)
    # Actually, the tensor decomposition means:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's not right. Let me re-read the task.
    # The task says: "exactly decompose the matrix multiplication tensor"
    # Standard tensor decomposition: T_{ijk} = sum_r U_{i,r} V_{j,r} W_{k,r}
    # But here the indices are mapped differently.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The contraction should give the identity mapping for matrix multiplication.
    # For C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] = 1 if this term appears in the sum.
    # So T[i,j,k] = 1 for all valid i,j,k.
    # The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # Wait, that's a 3-way decomposition but the task gives 3 matrices.
    # Let me re-read: "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # So the decomposition is: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is a rank-r decomposition of the all-ones tensor of shape (n*m, m*p, p*n).
    # But that's not quite right either. Let me think again.
    # Actually, in tensor decomposition for matrix multiplication, we have:
    # (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # A rank-r algorithm writes: (A*B)[i,k] = sum_c (sum_j A[i,j] U_{j,c}) * (sum_k B[j,k] V_{k,c}) * W_{i,k,c}
    # Hmm, the indexing is tricky. Let me just check: does sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1?
    # For the schoolbook: U[i*m+j, c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1 for c corresponding to (i,j,k).
    # Then sum_c ... = 1 for the matching c, 0 otherwise. So yes, it equals 1.
    # So the check is: for all i,j,k, sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Start with schoolbook decomposition
    rank0 = n * m * p
    U0 = [[0] * rank0 for _ in range(n * m)]
    V0 = [[0] * rank0 for _ in range(m * p)]
    W0 = [[0] * rank0 for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j][col] = 1
                V0[j*p+k][col] = 1
                W0[k*n+i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank0
    
    # Try to find lower rank decompositions using random search + optimization
    # We'll use a simple approach: try ranks from 18 down to 16, and use random initialization
    # with local optimization.
    
    # Theoretical lower bound for 2x3x3 is at least 17 (from literature), but let's try 18, 17, 16
    # Known: 2x3x3 matmul can be done in 18 multiplications.
    
    target_ranks = [18, 17, 16]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 1000 + target_rank)
            np.random.seed(attempt * 1000 + target_rank)
            
            # Random initialization with small values
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Simple gradient-free optimization: try to minimize the error
            # Error = sum_{i,j,k} (sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] - 1)^2
            
            best_err = float('inf')
            for step in range(200):
                if time.time() > deadline:
                    break
                
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err += (s - 1.0) ** 2
                err = err / (n*m*p)
                
                if err < 1e-10:
                    # Found a solution!
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U = U.tolist()
                        best_V = V.tolist()
                        best_W = W.tolist()
                    break
                
                if err < best_err:
                    best_err = err
                
                # Simple random perturbation
                if random.random() < 0.1:
                    # Large step
                    scale = 0.5
                else:
                    scale = 0.05
                
                U += np.random.randn(n*m, target_rank) * scale
                V += np.random.randn(m*p, target_rank) * scale
                W += np.random.randn(p*n, target_rank) * scale
            
            # Check if we found a valid decomposition
            if best_err < 1e-10:
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                break  # Found for this rank, move to next
    
    # Verify the best solution
    if best_rank < rank0:
        if not _check_decomposition(best_U, best_V, best_W, n, m, p):
            # Fallback to schoolbook
            best_U, best_V, best_W = U0, V0, W0
            best_rank = rank0
    
    return {"U": best_U, "V": best_V, "W": best_W}
