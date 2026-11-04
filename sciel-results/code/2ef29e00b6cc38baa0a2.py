"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x3 matrix by a 3x6 matrix.
    
    Strategy:
    1. The tensor for (3x3)*(3x6) multiplication has dimensions n=3, m=3, p=6.
       The multiplication tensor T_{i,j,k} = 1 if the product C[i,k] = sum_{j} A[i,j]*B[j,k].
       In the standard convention: T[i, j, k] corresponds to the term A[i,j] * B[j,k] contributing to C[i,k].
       
       The interface specifies:
       U: shape (n*m, rank) = (9, rank)
       V: shape (m*p, rank) = (18, rank)
       W: shape (p*n, rank) = (18, rank) with W indexed as W[k*n + i, r]
       
       The decomposition satisfies:
       sum_r U[i*m + j, r] * V[j*p + k, r] = W[k*n + i, r] ... wait, no.
       
       The standard matrix multiplication tensor decomposition is:
       C[i,k] = sum_j A[i,j] B[j,k]
       
       The tensor T is a 3-way tensor of size (n, m, p) where T[i,j,k] = 1.
       We want to decompose T as a sum of rank-1 tensors:
       T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
       
       But the interface uses flattened indices:
       U has shape (n*m, rank), so U[i*m + j, r]
       V has shape (m*p, rank), so V[j*p + k, r]
       W has shape (p*n, rank), so W[k*n + i, r]
       
       The condition is: for all i, j, k:
       T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
       
       Since T[i,j,k] = 1 for all i,j,k in the multiplication tensor (every term A[i,j]*B[j,k] contributes to C[i,k]),
       we need:
       sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i in [0,n), j in [0,m), k in [0,p)
       
       Wait, that's not right. The multiplication tensor T_{ijk} is 1 if the product C[i,k] includes the term A[i,j]*B[j,k].
       For matrix multiplication C = A*B, we have C[i,k] = sum_j A[i,j]*B[j,k].
       So the tensor T[i,j,k] = 1 for all i,j,k.
       
       Actually, let me reconsider. The tensor for matrix multiplication is defined such that:
       T[i,j,k] = 1 if the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
       For standard matrix multiplication, C[i,k] = sum_j A[i,j] B[j,k], so T[i,j,k] = 1 for all i,j,k.
       
       So we need to find rank-r decomposition:
       sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k
       
       This is equivalent to finding a rank-R decomposition of the all-ones tensor of size (n*m, m*p, p*n) ... no, that's not right either.
       
       Let me think again. The multiplication tensor is a 3-tensor T of size (n, m, p) where T[i,j,k] = 1.
       We want to express T as:
       T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
       
       where U is (n x m x R), V is (m x p x R), W is (n x p x R) ... but the indices need to match.
       
       Actually, the standard formulation is:
       The matrix multiplication tensor T_{n,m,p} is a 3-way tensor of dimensions (n, m, p) with T[i,j,k] = 1.
       A rank-R decomposition is a set of R triples (u_r, v_r, w_r) where:
       u_r in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n} (or some permutation)
       
       such that:
       sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1  for all i, j, k
       
       Let me verify with the schoolbook: for each (i,j,k), we have one term with u[i*m+j]=1, v[j*p+k]=1, w[k*n+i]=1, and all others 0.
       So sum over r gives 1 for each (i,j,k). Yes, that's correct.
       
       Now, the rank of the (3,3,3) matrix multiplication tensor is 7 (Strassen).
       The rank of the (3,3,6) tensor... we can think of it as a block structure.
       
       Let's try to construct a rank-21 decomposition using Strassen's algorithm for the 3x3 core and a simple decomposition for the 3x6 part.
       
       Actually, let's try a different approach: use the fact that we can decompose the 3x6 multiplication by treating the 3x6 matrix as three 3x2 blocks, or use a rank-3 decomposition for the rectangular part.
       
       For now, let's try to find a good decomposition using numerical optimization.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Leave 20s safety margin for 120s limit
    
    # Start with schoolbook rank = n*m*p = 54
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
    
    # Try to find a lower rank decomposition using ALS (Alternating Least Squares)
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact (within tolerance)."""
        # For each (i, j, k), check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    # Try ranks from 21 up to 54 (Strassen rank for 3x3 is 7, and we have p=6)
    # The theoretical lower bound is at least max(n*m, m*p, p*n) / something... 
    # Actually, the rank is at least max(n*m, m*p, p*n) in some cases, but not always.
    # For 3x3x3, rank is 7 < 9.
    
    # Let's try to use a structured approach:
    # Decompose the 3x3x6 tensor using the Kronecker product of 3x3x3 (rank 7) and a 3x1x2 tensor.
    # But this might not be straightforward.
    
    # Alternative: use numerical search with ALS for small ranks.
    # Start with rank = 21 (7 * 3, since 3x3x3 has rank 7 and we need to handle the 3x6 part)
    
    for target_rank in range(21, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Initialize with random values
        np.random.seed(42)
        U_init = np.random.randn(n*m, target_rank) * 0.1
        V_init = np.random.randn(m*p, target_rank) * 0.1
        W_init = np.random.randn(p*n, target_rank) * 0.1
        
        # Scale to make the product approximately 1
        # sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] should be 1
        # Let's scale each factor by (n*m*p)**(-1/3) or similar
        
        # Use ALS to refine
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        # Precompute indices
        indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        for iteration in range(100):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (i, j), solve for U[i*m+j, :]
            for i in range(n):
                for j in range(m):
                    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                    # This is a linear system: U[i*m+j, :] * (V[j*p+k, :] * W[k*n+i, :]) = 1 for each k
                    # Let M be a (p x target_rank) matrix where M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # Then U[i*m+j, :] * M^T = [1, 1, ..., 1]
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve U[i*m+j, :] * M^T = ones(p)
                    # This is (M^T)^T * U[i*m+j, :]^T = ones(p)
                    # i.e., M * U[i*m+j, :]^T = ones(p)
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j, k), solve for V[j*p+k, :]
            for j in range(m):
                for k in range(p):
                    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                    # Let M be a (n x target_rank) matrix where M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    # Then V[j*p+k, :] * M^T = [1, 1, ..., 1]
                    M = np.zeros((n, target_rank))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (i, k), solve for W[k*n+i, :]
            for i in range(n):
                for k in range(p):
                    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                    # Let M be a (m x target_rank) matrix where M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    # Then W[k*n+i, :] * M^T = [1, 1, ..., 1]
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        pass
        
        # Check if the decomposition is valid
        if verify(U.tolist(), V.tolist(), W.tolist(), target_rank):
            best_rank = target_rank
            best_U = U.tolist()
            best_V = V.tolist()
            best_W = W.tolist()
            break  # Found the lowest rank, stop
    
    # Convert to lists if needed
    if isinstance(best_U, np.ndarray):
        best_U = best_U.tolist()
    if isinstance(best_V, np.ndarray):
        best_V = best_V.tolist()
    if isinstance(best_W, np.ndarray):
        best_W = best_W.tolist()
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
