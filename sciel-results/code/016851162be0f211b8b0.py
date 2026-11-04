"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120s wall-clock limit
    safety = 10  # seconds before limit to stop
    
    # Build the matrix multiplication tensor T[i,j,k] = 1
    # Indexing: U has rows indexed by (i,j) -> i*m+j
    #           V has rows indexed by (j,k) -> j*p+k
    #           W has rows indexed by (k,i) -> k*n+i
    # T is a 3D tensor of shape (n, m, p) with all entries 1
    # We need to find rank-minimal decomposition:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Flatten T into a matrix for rank analysis
    # The tensor T of shape (n*m, p) can be viewed as a matrix
    # Actually, the matrix multiplication tensor M_{(i,j),(j,k),(k,i)} 
    # The standard approach: consider the tensor as a 3-way array
    # T[i,j,k] = 1 for all i,j,k
    
    # For matrix multiplication, the tensor is:
    # M[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # M[i,j,k] = 1
    
    # The rank of this tensor is what we seek.
    # For small dimensions, we can try to find exact decompositions.
    
    # Strategy: 
    # 1. Start with schoolbook (rank = n*m*p = 72)
    # 2. Try to find lower rank decompositions using known techniques
    # 3. For 3x4x6, try various structured decompositions
    
    # Schoolbook decomposition (rank 72)
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to improve using various strategies
    # For 3x4x6, we can try:
    # 1. Block-structured decompositions
    # 2. Search for lower rank using optimization
    
    # Let's try a systematic search for rank < 72
    # The tensor is T[i,j,k] = 1 for all i,j,k
    # This is actually the all-ones tensor, which has a special structure
    
    # Wait, the matrix multiplication tensor is NOT all ones.
    # Let me reconsider. The matrix multiplication tensor for multiplying
    # an n x m matrix by an m x p matrix is:
    # T[i, j, k] where i indexes rows of first matrix (0..n-1),
    # j indexes columns of first matrix / rows of second (0..m-1),
    # k indexes columns of second matrix (0..p-1)
    # T[i,j,k] = 1
    
    # Actually, the standard matrix multiplication tensor has entries:
    # M_{i,j,k} = 1 if we're computing element (i,k) of the product
    # using element (i,j) of A and element (j,k) of B
    # So yes, M[i,j,k] = 1 for all valid i,j,k
    
    # This is the all-ones tensor of shape (n, m, p)
    # The rank of the all-ones tensor of shape (n, m, p) is known to be
    # at most n*m*p (schoolbook), but can be lower.
    
    # For the all-ones tensor, we can use:
    # T[i,j,k] = 1 = (1)(1)(1)
    # This gives rank 1! But wait, that's not right for matrix multiplication.
    
    # Let me re-read the problem. The tensor decomposition is for matrix multiplication.
    # The standard formulation: given A (n x m) and B (m x p), compute C = A @ B
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor T has entries T[i,j,k] = 1, and the multiplication is:
    # C[i,k] = sum_{j} sum_{c} U[i,j,c] * V[j,k,c] * W[i,k,c] * A[i,j] * B[j,k]
    
    # Wait, I need to be more careful. The standard tensor decomposition for
    # matrix multiplication uses:
    # C[i,k] = sum_c (sum_j U[i,j,c] * A[i,j]) * (sum_j V[j,k,c] * B[j,k]) * W[i,k,c]
    
    # No, that's not right either. Let me think again.
    
    # The correct formulation: we want to compute C[i,k] = sum_j A[i,j] B[j,k]
    # using rank-r decomposition:
    # C[i,k] = sum_{c=1}^r (sum_j U_c[i,j] A[i,j]) (sum_j V_c[j,k] B[j,k]) W_c[i,k]
    
    # Hmm, that doesn't look standard. Let me use the standard tensor formulation.
    
    # The matrix multiplication tensor is a 3-tensor M of shape (n*m, m*p, n*p)
    # where M[(i,j), (j,k), (i,k)] = 1
    # 
    # A rank-1 decomposition is: M = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^{n*m}, v_c is in R^{m*p}, w_c is in R^{n*p}
    # 
    # So M[(i,j), (j,k), (i,k)] = sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[i*n+k]
    # 
    # Wait, the W indexing in the problem is W[k*n+i, r], so w_c[k*n+i]
    # This means the third index is (k, i) not (i, k).
    
    # So the tensor is: M[(i,j), (j,k), (k,i)] = 1
    # And we decompose: M[(i,j), (j,k), (k,i)] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For this to give matrix multiplication, we need:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] A[i,j] B[j,k]
    
    # This works if for each c, the terms U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # combine to give the correct product.
    
    # The schoolbook decomposition has rank n*m*p with U[i*m+j, c] = 1 only when
    # c corresponds to (i,j,k), similarly for V and W.
    
    # Now, for the all-ones-like structure, can we do better?
    
    # Let's try a simple approach: since the tensor entries are all 1,
    # we can try low-rank approximations and check for exactness.
    
    # Actually, let me just try various ranks and see if we can find exact decompositions.
    
    # For now, let's try a structured approach.
    # One idea: use the fact that 1 = 1*1*1, but we need the indices to work out.
    
    # Let's try rank = n + m + p - 2 or similar small values.
    
    # Actually, let me just return the schoolbook for now, but try to improve.
    
    # Try a simple optimization: random initialization + gradient descent
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    # Wait, that's not right. The decomposition is:
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    # Try small ranks with random initialization
    np.random.seed(42)
    for target_rank in range(1, min(best_rank, 30)):
        if time.time() > deadline - safety:
            break
        for trial in range(20):
            if time.time() > deadline - safety:
                break
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            # Simple gradient descent
            lr = 0.01
            for step in range(500):
                if time.time() > deadline - safety:
                    break
                # Compute gradient
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err = val - 1
                            grad_U[i*m+j, :] += err * V[j*p+k, :] * W[k*n+i, :]
                            grad_V[j*p+k, :] += err * U[i*m+j, :] * W[k*n+i, :]
                            grad_W[k*n+i, :] += err * U[i*m+j, :] * V[j*p+k, :]
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            if verify(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
    
    # Convert to lists for return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
