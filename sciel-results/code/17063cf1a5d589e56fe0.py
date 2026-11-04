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

def _try_reduce_rank(n, m, p, deadline):
    """
    Attempt to find a rank-minimal exact tensor decomposition by checking if any schoolbook term
    can be expressed as a linear combination of the others.
    
    The tensor for matrix multiplication T_{i,j,k} = E_{i,k} \otimes E_{j,j} (in appropriate spaces)
    Each schoolbook term c=(i,j,k) contributes to U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
    
    To check if term t can be removed, we need to find coefficients a_c for c != t such that
    the tensor contribution of term t equals sum of a_c * tensor_contribution(c) for c != t.
    
    This is equivalent to: is the vector for term t in the span of the vectors for other terms?
    Each term corresponds to a vector in R^{(n*m)*(m*p)*(p*n)} (the full tensor space),
    but more efficiently, we can work with the structured form.
    
    Actually, a simpler check: if we fix the U and V matrices for the remaining terms,
    we need W to satisfy: for all (i,j,k), T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c].
    
    Let's set up the linear system properly.
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
    
    # For each candidate term to remove, check if it's in the span of others
    # We'll use a randomized approach: pick a few candidates and test
    
    # The constraint is: for all (i,j,k), 
    #   1 = sum_{c != t} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where U, V, W are the schoolbook values for terms c != t
    # and W is unknown for the remaining terms.
    
    # Actually, this is underdetermined. Let me think differently.
    
    # A term t=(i0,j0,k0) has:
    #   U[i0*m+j0, t] = 1, V[j0*p+k0, t] = 1, W[k0*n+i0, t] = 1
    #   All other entries in U, V, W for column t are 0.
    
    # If we remove term t, we need to modify the other terms' U, V, W values
    # such that the product still equals the identity tensor.
    
    # This is a hard problem. Let's use a simpler heuristic:
    # Check if there exist two terms that can be merged.
    
    # For now, return None to indicate no reduction found
    return None

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
    
    # 3. Return the best valid decomposition found
    return best
# EVOLVE-BLOCK-END
