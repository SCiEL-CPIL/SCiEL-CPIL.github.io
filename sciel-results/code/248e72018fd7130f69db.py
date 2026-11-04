"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def _schoolbook(n, m, p):
    """
    Standard schoolbook matrix multiplication tensor decomposition.
    Rank = n * m * p.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    Indexing:
    U[i*m+j, c]
    V[j*p+k, c]
    W[k*n+i, c]
    """
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

def _try_reduce_rank(n, m, p, deadline):
    """
    Attempt to find a lower-rank decomposition by checking if any schoolbook term
    can be expressed as a linear combination of the others using the structured tensor form.
    
    For matrix multiplication, the tensor T has components T[i,j,k] where:
    - i in [0,n), j in [0,m), k in [0,p)
    - T[i,j,k] = 1 if we're computing (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    
    Each schoolbook term c=(i,j,k) contributes:
    - U[i*m+j, c] = 1
    - V[j*p+k, c] = 1  
    - W[k*n+i, c] = 1
    
    To check if term t can be removed, we need to find coefficients a_c for c != t
    such that for all (i,j,k):
    T[i,j,k] = sum_{c != t} a_c * U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    This is equivalent to checking if the "vector" for term t is in the span of
    vectors for other terms in the tensor space.
    """
    base_rank = n * m * p
    if base_rank <= 20:  # Lower bound
        return None
    
    # Build the schoolbook terms
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append((i, j, k))
    
    num_terms = len(terms)
    
    # For each candidate term to remove, set up the linear system
    # The constraint: for all (i,j,k), 
    #   T[i,j,k] = sum_{c != t} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # where U, V, W are the schoolbook values for terms c != t
    # and we need to find if there exist W values that satisfy this.
    
    # Actually, the proper approach: each term defines a rank-1 tensor
    # T_c = U_c ⊗ V_c ⊗ W_c where U_c, V_c, W_c are vectors with single 1s.
    # The sum of all rank-1 tensors should equal the matrix multiplication tensor.
    
    # To check if term t is redundant, we ask: is T_t in the span of {T_c : c != t}?
    # This is equivalent to checking if the vectorized T_t is in the column space
    # of the matrix formed by vectorized T_c for c != t.
    
    # Vectorize each rank-1 tensor into a vector of length n*m*p
    def vec_tensor(i, j, k):
        """Create the vectorized form of the rank-1 tensor for term (i,j,k)."""
        vec = [0] * (n * m * p)
        # The rank-1 tensor has non-zero entries at positions (i,j,k)
        # In the flattened form, position is i*m*p + j*p + k
        vec[i * m * p + j * p + k] = 1
        return vec
    
    # Build the matrix of all term vectors
    term_vectors = []
    for i, j, k in terms:
        term_vectors.append(vec_tensor(i, j, k))
    
    # Convert to numpy for linear algebra
    import numpy as np
    A = np.array(term_vectors, dtype=float)  # Shape: (num_terms, n*m*p)
    
    # For each candidate term t, check if its vector is in the span of others
    # This is equivalent to checking if removing that row doesn't change the rank
    for t in range(num_terms):
        if time.time() > deadline - 10.0:
            break
        
        # Create matrix without row t
        A_reduced = np.delete(A, t, axis=0)
        
        # Check if A[t] is in the column space of A_reduced
        # This is equivalent to: rank([A_reduced; A[t]]) == rank(A_reduced)
        A_full = np.vstack([A_reduced, A[t]])
        
        rank_reduced = np.linalg.matrix_rank(A_reduced)
        rank_full = np.linalg.matrix_rank(A_full)
        
        if rank_reduced == rank_full:
            # Term t is redundant! We can remove it.
            # Build the reduced decomposition
            new_terms = [terms[c] for c in range(num_terms) if c != t]
            rank = len(new_terms)
            
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            for col, (i, j, k) in enumerate(new_terms):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
            
            return {"U": U, "V": V, "W": W}
    
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly reproduces matrix multiplication.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all (i,j,k)
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill switch
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # 1. Establish the safe baseline (Schoolbook)
    best = _schoolbook(n, m, p)
    best_rank = base_rank
    
    # 2. Structural search: attempt rank reduction via linear algebra
    # Try to find if any term can be eliminated
    if time.time() < deadline - 30.0:
        result = _try_reduce_rank(n, m, p, deadline)
        if result is not None:
            # Verify before accepting
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = len(result["U"][0])
    
    # 3. Try iterative reduction: keep removing terms as long as possible
    current_rank = best_rank
    max_iterations = 10
    for iteration in range(max_iterations):
        if time.time() > deadline - 15.0:
            break
        if current_rank <= 20:  # Lower bound
            break
        
        result = _try_reduce_rank(n, m, p, deadline)
        if result is not None and len(result["U"][0]) < current_rank:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                best = result
                current_rank = len(result["U"][0])
            else:
                break
        else:
            break
    
    # 4. Return the best valid decomposition found
    return best
# EVOLVE-BLOCK-END
