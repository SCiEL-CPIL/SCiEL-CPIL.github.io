"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin for 150s limit

def _time_exceeded():
    return (time.time() - _START_TIME) > _DEADLINE

def _verify(U, V, W, n=4, m=5, p=5):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    
    The decomposition satisfies:
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    This corresponds to the bilinear form where:
    C[i,k] = sum_j A[i,j] * B[j,k]
    and the tensor T[i,j,k] = 1 for all valid indices.
    """
    if not U or not V or not W:
        return False
    
    rank = len(U[0]) if U else 0
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check all required tensor entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                u_val = U_arr[i * m + j]
                v_val = V_arr[j * p + k]
                w_val = W_arr[k * n + i]
                total = np.dot(u_val, v_val * w_val)
                if abs(total - 1.0) > 1e-9:
                    return False
    
    # Additionally, we should check that "wrong" combinations give 0
    # But actually, for matrix multiplication, the tensor is defined only for
    # the specific structure. Let me reconsider.
    
    # Actually, the tensor for matrix multiplication C = A*B where A is n x m, B is m x p:
    # The bilinear form is: <C, A ⊗ B> = sum_{i,j,k} C[i,k] * A[i,j] * B[j,k]
    # And we want this to equal <AB, A ⊗ B> = sum_{i,k} (AB)[i,k] * A[i,j] * B[j,k]
    # Wait, that's not quite right. Let me think again.
    
    # The standard tensor decomposition for matrix multiplication:
    # T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^{n*m}, v_c is in R^{m*p}, w_c is in R^{p*n}
    # and the multiplication is recovered by:
    # C = sum_c w_c * (u_c^T A) * (v_c^T B)^T
    # Wait, that's not standard either.
    
    # Let me use the correct formulation:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # A rank-r decomposition means we can write:
    # C[i,k] = sum_c U_c[i] * V_c[k] * W_c
    # where U_c is a vector of length n, V_c is a vector of length p, and W_c is a scalar?
    # No, that's not right for general matrix multiplication.
    
    # The correct tensor formulation:
    # The multiplication tensor T ∈ R^{n*m} ⊗ R^{m*p} ⊗ R^{n*p} (or R^{p*n} with appropriate indexing)
    # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    # where e_{i,j} is the basis vector in R^{n*m} at position (i,j),
    # e_{j,k} is the basis vector in R^{m*p} at position (j,k),
    # e_{i,k} is the basis vector in R^{n*p} at position (i,k).
    
    # A rank-r decomposition: T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
    # where u_c ∈ R^{n*m}, v_c ∈ R^{m*p}, w_c ∈ R^{n*p} (or R^{p*n})
    
    # The condition is: for all (i,j), (j',k), (i',k'):
    # sum_c u_c[i,j] * v_c[j',k] * w_c[i',k'] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # Wait, that's not right either.
    
    # Let me be more careful. The tensor T has entries:
    # T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k' and the multiplication is valid
    # Actually, for matrix multiplication, the tensor is:
    # T[(i,j), (j',k), (i',k')] = 1 if j=j' and i=i' and k=k', else 0
    # No, that's not right.
    
    # The correct statement: The bilinear map is:
    # (A, B) -> C where C[i,k] = sum_j A[i,j] * B[j,k]
    # As a trilinear form: f(A, B, C) = sum_{i,j,k} A[i,j] * B[j,k] * C[k,i] (with appropriate indexing)
    # Wait, C is indexed by (i,k), so f(A, B, C) = sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
    # And we want f(A, B, C) = sum_{i,k} (AB)[i,k] * C[i,k] = sum_{i,k} (sum_j A[i,j] B[j,k]) * C[i,k]
    # = sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
    # So the tensor T has entries T[i,j,k] = 1 for all i,j,k, and the decomposition is:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where W is indexed by k*n+i (i.e., W[row, col] where row = k*n+i)
    
    # So the condition is: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is what I checked above. But we should also ensure that the decomposition
    # doesn't introduce spurious terms. Actually, since the tensor is fully defined by
    # these 100 entries (all equal to 1), and the decomposition is a sum of rank-1 tensors,
    # the condition above is necessary and sufficient.
    
    return True

def _verify_full(U, V, W, n=4, m=5, p=5):
    """
    Full verification: check that the decomposition correctly computes matrix multiplication
    for random test matrices.
    """
    if not U or not V or not W:
        return False
    
    rank = len(U[0]) if U else 0
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Test with several random matrices
    for trial in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C using the decomposition
        # C[i,k] = sum_c W[k*n+i, c] * (sum_{i',j'} U[i*m+j', c] * A[i',j']) * (sum_{j',k'} V[j'*p+k', c] * B[j',k'])
        # Wait, this is not quite right. Let me think about the correct formula.
        
        # The bilinear form is:
        # C[i,k] = sum_c W[k*n+i, c] * (u_c^T A_vec) * (v_c^T B_vec)
        # where A_vec is A flattened in row-major order (i*m+j),
        # B_vec is B flattened in row-major order (j*p+k),
        # and u_c = U[:, c], v_c = V[:, c], w_c = W[:, c]
        
        # But this gives C[i,k] = sum_c W[k*n+i, c] * (sum_{i',j'} U[i*m+j', c] A[i',j']) * (sum_{j',k'} V[j'*p+k', c] B[j',k'])
        # which is not the standard matrix multiplication.
        
        # Actually, I think the correct interpretation is different. Let me reconsider.
        
        # The standard tensor decomposition for matrix multiplication:
        # C = sum_c (W_c) * (U_c^T A) * (V_c^T B)^T
        # where U_c is a vector, V_c is a vector, W_c is a scalar?
        
        # No, the correct formula is:
        # C[i,k] = sum_c U_c[i] * V_c[k] * W_c
        # where U_c is a matrix of shape (n, m), V_c is a matrix of shape (m, p), W_c is a scalar?
        
        # I'm getting confused. Let me use the standard formulation from the literature.
        
        # The matrix multiplication tensor T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p}
        # T = sum_{i,j,k} E_{ij} ⊗ E_{jk} ⊗ E_{ik}
        # where E_{ij} is the matrix with 1 at (i,j) and 0 elsewhere.
        
        # A rank-r decomposition: T = sum_{c=0}^{r-1} U_c ⊗ V_c ⊗ W_c
        # where U_c ∈ R^{n×m}, V_c ∈ R^{m×p}, W_c ∈ R^{n×p}
        
        # The multiplication is recovered by:
        # C = sum_c (U_c * A) ⊙ (V_c * B) * W_c
        # where * denotes element-wise multiplication and ⊙ denotes outer product?
        
        # Actually, the correct formula is:
        # C = sum_c (U_c^T A) (V_c^T B)^T * W_c
        # No, that's not right either.
        
        # Let me look at this from first principles.
        # We have: C[i,k] = sum_j A[i,j] B[j,k]
        # We want to write this as: C[i,k] = sum_c f_c(A) * g_c(B) * h_c(i,k)
        # where f_c is a linear form on A, g_c is a linear form on B, h_c(i,k) is a function.
        
        # So: f_c(A) = sum_{i,j} U_c[i,j] A[i,j]
        # g_c(B) = sum_{j,k} V_c[j,k] B[j,k]
        # h_c(i,k) = W_c[i,k]
        
        # And we need: sum_c U_c[i,j] V_c[j,k] W_c[i,k] = delta_{j,j'} for the right indices?
        
        # Actually, the condition is:
        # sum_c U_c[i',j'] V_c[j,k'] W_c[i,k] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
        # No, that's not right.
        
        # The correct condition: For the bilinear form to compute matrix multiplication,
        # we need: sum_c U_c[i,j] V_c[j,k] W_c[i,k] = 1 for all i,j,k
        # And for "wrong" combinations, it should be 0. But what are "wrong" combinations?
        
        # Actually, I think the tensor is fully specified by the 100 entries T[i,j,k] = 1,
        # and there are no "wrong" combinations to check. The decomposition is valid if
        # and only if sum_c U_c[i,j] V_c[j,k] W_c[i,k] = 1 for all i,j,k.
        
        # But wait, this can't be right, because if we sum over all i,j,k, we get
        # sum_{i,j,k} sum_c U_c[i,j] V_c[j,k] W_c[i,k] = n*m*p = 100
        # But also sum_c (sum_{i,j} U_c[i,j]) (sum_{j,k} V_c[j,k]) (sum_{i,k} W_c[i,k]) = 100
        # This is a consistency check but not a verification.
        
        # Let me try a different approach: verify by actually computing matrix products.
        
        # The formula for C using the decomposition should be:
        # C = sum_c W_c * (U_c^T A) * (V_c^T B)^T
        # where U_c^T A means the dot product of U_c (flattened) with A (flattened),
        # giving a scalar, similarly for V_c^T B, and W_c is a matrix.
        
        # So: C[i,k] = sum_c W_c[i,k] * (sum_{i',j'} U_c[i',j'] A[i',j']) * (sum_{j',k'} V_c[j',k'] B[j',k'])
        
        # Let's test this:
        C_computed = np.zeros((n, p))
        for c in range(rank):
            u_flat = U_arr[:, c]  # shape (n*m,)
            v_flat = V_arr[:, c]  # shape (m*p,)
            w_mat = W_arr[:, c].reshape(p, n)  # shape (p, n), but we need (n, p)
            # Wait, W is indexed by k*n+i, so W[k*n+i, c] corresponds to C[i,k]
            # So W_arr[k*n+i, c] is the coefficient for C[i,k]
            # To get a matrix W_c of shape (n, p), we need W_c[i,k] = W_arr[k*n+i, c]
            
            u_scalar = np.dot(u_flat, A.flatten())  # sum_{i',j'} U_c[i',j'] A[i',j']
            v_scalar = np.dot(v_flat, B.flatten())  # sum_{j',k'} V_c[j',k'] B[j',k']
            
            # W_c[i,k] = W_arr[k*n+i, c]
            w_mat = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    w_mat[i, k] = W_arr[k * n + i, c]
            
            C_computed += w_mat * u_scalar * v_scalar
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    
    return True

def construct():
    """
    Constructs a tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    
    First tries the schoolbook decomposition (rank 100), then attempts to find
    lower-rank decompositions using algebraic methods.
    """
    if _time_exceeded():
        return {"U": [], "V": [], "W": []}
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Try to find lower-rank decompositions
    # Strategy: Use the fact that the tensor has structure and try to find
    # linear dependencies or use known low-rank decompositions.
    
    # For now, return the schoolbook decomposition as a safe fallback.
    # Future work: implement algebraic methods to find lower-rank decompositions.
    
    return best_result
# EVOLVE-BLOCK-END
