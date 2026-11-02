import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def _build_bclr_3x3x3():
    # Returns U, V, W for 3x3x3 multiplication using BCLR 23-term algorithm
    # U: 9x23, V: 9x23, W: 9x23
    # Coefficients are integers
    # Note: I will define the 23 terms here.
    # If I don't have them, I will return schoolbook 27 terms.
    # For now, I will use schoolbook 27 terms to ensure correctness, 
    # but the prompt asks for a STRONGER algorithm.
    # I will implement a search for a lower rank decomposition with integer coefficients.
    pass

def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition"""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact"""
    # Check that U @ V @ W^T reconstructs the tensor
    # The tensor T has T[i,j,k] = 1 if i,j,k are valid indices
    # In our case, T[i,j,k] = 1 for all i,j,k (matrix multiplication)
    
    # Reconstruct: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 for all i,j,k
    
    # More efficient: check using matrix multiplication structure
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] contributes to C[i,k]
    
    # Let's verify by checking a few random matrices
    rng = np.random.RandomState(42)
    for _ in range(10):
        A = rng.randint(-5, 6, size=(n, m))
        B = rng.randint(-5, 6, size=(m, p))
        C_expected = A @ B
        
        # Reconstruct using decomposition
        # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
        # Wait, that's not right. Let me think again.
        
        # The tensor decomposition is:
        # T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
        
        # And matrix multiplication is:
        # C_{i,k} = sum_j A_{i,j} * B_{j,k} * T_{i,j,k}
        
        # So: C_{i,k} = sum_j sum_r A_{i,j} * B_{j,k} * U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
        
        C_reconstructed = np.zeros((n, p), dtype=float)
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    for r in range(U.shape[1]):
                        total += A[i, j] * B[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_reconstructed[i, k] = total
        
        if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
            return False
    return True

def _greedy_sparse_decomposition(n, m, p, max_rank, deadline):
    """
    Greedy sparse linear regression to find low-rank decomposition.
    
    The idea: we want to find U, V, W such that:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    This is a bilinear problem. We can approach it by:
    1. Start with rank 1
    2. At each step, find the best rank-1 term to add that minimizes residual
    
    For a rank-1 term (u, v, w), the contribution to T[i,j,k] is:
    u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    We want this to approximate the residual.
    """
    start_time = time.time()
    
    # Target tensor: T[i,j,k] = 1 for all i,j,k
    # This is the all-ones tensor of shape (n*m, m*p, p*n)
    
    # We'll work with the vectorized form
    # Residual starts as all 1s
    residual = np.ones((n, m, p), dtype=float)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Candidate values for entries: small integers
    candidate_vals = [-2, -1, 0, 1, 2]
    
    current_rank = 0
    while current_rank < max_rank:
        if time.time() > deadline - 5:
            break
        
        # Find best rank-1 term (u, v, w) to minimize ||residual - u⊗v⊗w||_F^2
        # This is equivalent to maximizing <residual, u⊗v⊗w> - 0.5 * ||u⊗v⊗w||_F^2
        # But this is still hard. Let's use a greedy approach.
        
        # Strategy: try all combinations of small integer vectors
        # But that's 5^9 * 5^12 * 5^12 which is too large.
        
        # Better: use random search + local improvement
        best_score = -np.inf
        best_u, best_v, best_w = None, None, None
        
        n_attempts = 500
        for attempt in range(n_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            u = np.random.choice(candidate_vals, size=n*m)
            v = np.random.choice(candidate_vals, size=m*p)
            w = np.random.choice(candidate_vals, size=p*n)
            
            # Compute score: how well does u⊗v⊗w match residual?
            # We want to maximize <residual, u⊗v⊗w>
            # which is sum_{i,j,k} residual[i,j,k] * u[i*m+j] * v[j*p+k] * w[k*n+i]
            
            # This is a tensor contraction
            # Let's compute it efficiently
            # First, contract u with residual over i,j: 
            # temp[j*p+k, i] = sum_j residual[i,j,k] * u[i*m+j]
            # No, let me think differently.
            
            # <residual, u⊗v⊗w> = sum_{i,j,k} residual[i,j,k] * u[i*m+j] * v[j*p+k] * w[k*n+i]
            
            # We can compute this as:
            # 1. For each i, compute: sum_j sum_k residual[i,j,k] * u[i*m+j] * v[j*p+k] * w[k*n+i]
            # 2. Sum over i
            
            score = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        score += residual[i, j, k] * u[i*m+j] * v[j*p+k] * w[k*n+i]
            
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
        
        if best_u is None:
            break
        
        # Add this term
        U_cols.append(best_u)
        V_cols.append(best_v)
        W_cols.append(best_w)
        
        # Update residual
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    residual[i, j, k] -= best_u[i*m+j] * best_v[j*p+k] * best_w[k*n+i]
        
        current_rank += 1
        
        # Check if residual is close to zero
        if np.max(np.abs(residual)) < 1e-10:
            break
    
    if current_rank == 0:
        return None
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        return None
    
    return U, V, W, current_rank

def _local_search_improve(U, V, W, n, m, p, deadline):
    """
    Try to improve the decomposition by removing terms or replacing them.
    """
    start_time = time.time()
    current_rank = U.shape[1]
    
    # Try to remove each term and see if we can compensate
    for col in range(current_rank - 1, -1, -1):
        if time.time() > deadline - 10:
            break
        
        # Try removing this column
        U_test = np.delete(U, col, axis=1)
        V_test = np.delete(V, col, axis=1)
        W_test = np.delete(W, col, axis=1)
        
        if _verify_decomposition(U_test, V_test, W_test, n, m, p):
            U, V, W = U_test, V_test, W_test
            current_rank -= 1
    
    return U, V, W, current_rank

def construct():
    start = time.time()
    deadline = start + 110.0
    
    n, m, p = DIMENSIONS
    
    # First, build schoolbook as fallback
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n * m * p
    
    # Try greedy sparse decomposition
    # Start with a reasonable max rank
    # We know rank 23 is possible for 3x3x3, so for 3x3x4 maybe around 27-30
    for max_rank in [30, 28, 26, 24, 22, 20]:
        if time.time() > deadline - 20:
            break
        
        result = _greedy_sparse_decomposition(n, m, p, max_rank, deadline)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                # Try to improve further
                U, V, W, rank = _local_search_improve(U, V, W, n, m, p, deadline)
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
    
    return best
