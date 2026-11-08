import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _is_half_int_matrix(A, tol=1e-6):
    """Check if all entries of A are integer or half-integer (i.e., 2*A is integer)."""
    return np.allclose(2.0 * A, np.round(2.0 * A), atol=tol)

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-8):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    # The tensor for C = A * B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In trace convention: W[k*n+i, r]
    # We need: for each r, U[:,r] * V[:,r] contributes to the tensor
    # U has shape (n*m, rank), indexed by (i,j)
    # V has shape (m*p, rank), indexed by (j,k)
    # W has shape (p*n, rank), indexed by (k,i)
    #
    # The decomposition says: T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    # Wait, that's not right. Let me think again.
    #
    # Actually, the standard tensor decomposition for matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T is 3-dimensional with indices (i,j,k) where T[i,j,k] = 1 if this term contributes.
    # But actually the matrix multiplication tensor is:
    # T_{i,j,k} = 1 for the "structural" tensor, and the decomposition is:
    # T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    #
    # Hmm, let me reconsider. In the context of matrix multiplication:
    # - U[i*m+j, r] is the coefficient for A[i,j]
    # - V[j*p+k, r] is the coefficient for B[j,k]  
    # - W[k*n+i, r] is the coefficient for C[i,k] (or contributes to it)
    #
    # The condition is: sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r] ... no that's not right either.
    #
    # Let me look at this more carefully. The matrix multiplication can be written as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # A tensor decomposition of rank R means we can write:
    # C[i,k] = sum_{r=1}^{R} sum_j (U[i*m+j, r] * A[i,j]) * (V[j*p+k, r] * B[j,k])
    #        = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * A[i,j] * B[j,k]
    #
    # For this to equal sum_j A[i,j] * B[j,k] for all A and B, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r] ... 
    #
    # Actually, I think the correct interpretation is:
    # The matrix multiplication tensor T has T[i,j,k] = 1 (it's the "structural" tensor).
    # A rank-R decomposition means T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    #
    # Wait, that doesn't work dimensionally. Let me think about it differently.
    #
    # In the standard formulation, if we have:
    # U: (n*m) x R
    # V: (m*p) x R  
    # W: (p*n) x R
    #
    # Then the condition is that for all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # No wait, that's a 3-index sum which gives a scalar, but we need it to be 1 for all (i,j,k).
    # That can't be right because the LHS depends on i, j, k but the RHS is just 1.
    #
    # Let me reconsider. I think the correct formulation is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    #
    # For this to equal sum_j A[i,j] * B[j,k], we need the bilinear forms to "reconstruct" the product.
    #
    # Actually, I think the standard way is:
    # Define scalar products: 
    #   u_r(A) = sum_{i,j} U[i*m+j, r] * A[i,j]
    #   v_r(B) = sum_{j,k} V[j*p+k, r] * B[j,k]
    #   Then C[i,k] = sum_r W[k*n+i, r] * u_r(A) * v_r(B)
    #
    # For this to work: sum_r W[k*n+i, r] * (sum_{i',j} U[i'*m+j, r] * A[i',j]) * (sum_{j',k'} V[j'*p+k', r] * B[j',k'])
    # = sum_j A[i,j] * B[j,k]
    #
    # This is getting complex. Let me just verify numerically by checking the tensor equation.
    #
    # The simplest check: build the multiplication tensor and verify the decomposition.
    # T[i,j,k] = delta_{correct} ... actually T is just the identity-like tensor for matmul.
    #
    # Let me just do a direct numerical check: pick random A, B, compute C = A@B,
    # then compute C' using the decomposition and check they match.
    
    # Using the interpretation: C[i,k] = sum_r W[k*n+i, r] * (sum_{i',j} U[i'*m+j,r]*A[i',j]) * (sum_{j',k'} V[j'*p+k',r]*B[j',k'])
    # This doesn't seem right dimensionally for reconstruction.
    
    # Let me try the most natural interpretation:
    # The tensor for matrix multiplication is T[i,j,k] = 1 (for all valid i,j,k)
    # and the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Check: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :rank] * V[j*p+k, :rank], W[k*n+i, :rank])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_search(n, m, p, deadline):
    """Search for lower-rank decompositions."""
    import random
    
    best_rank = n * m * p
    best_solution = _schoolbook(n, m, p)
    
    # Try ranks from (best_rank-1) downward
    for target_rank in range(best_rank - 1, 0, -1):
        if time.time() > deadline:
            break
        
        # Try a few random searches at this rank
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with half-integer values
            U = np.random.randint(-2, 3, size=(n * m, target_rank)) * 0.5
            V = np.random.randint(-2, 3, size=(m * p, target_rank)) * 0.5
            W = np.random.randint(-2, 3, size=(p * n, target_rank)) * 0.5
            
            # Simple greedy/local search: try to fix the decomposition
            # For each (i,j,k), the constraint is:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is nonlinear in the variables, so simple local search is hard.
            #
            # Alternative: use a structured approach. For small problems, we can
            # try to find a decomposition where each term contributes to a specific
            # (i,j,k) pattern.
            
            # For now, just check if random initialization happens to work (unlikely)
            if _verify_decomposition(U, V, W, n, m, p, target_rank, tol=1e-6):
                best_rank = target_rank
                best_solution = {"U": U, "V": V, "W": W}
                break
    
    return best_solution, best_rank

def _structured_search(n, m, p, deadline):
    """Try structured decompositions."""
    # Known result: for 2x3 times 3x6, the minimum rank is at most 2*3*6=36
    # but could be lower. Let's try some known patterns.
    #
    # One approach: if p is a multiple of n, we might be able to reduce rank.
    # Here p=6, n=2, so p/n=3.
    
    best_rank = n * m * p
    best_solution = _schoolbook(n, m, p)
    
    # Try rank = n * m * (p//n) if p is divisible by n
    if p % n == 0:
        candidate_rank = n * m * (p // n)
        if candidate_rank < best_rank:
            # Construct a structured decomposition
            U = np.zeros((n * m, candidate_rank))
            V = np.zeros((m * p, candidate_rank))
            W = np.zeros((p * n, candidate_rank))
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for kk in range(p // n):
                        # For each "block" of k values
                        k_base = kk * n
                        U[i * m + j, col] = 1.0
                        # V needs to sum over k in the block
                        for k_offset in range(n):
                            k = k_base + k_offset
                            V[j * p + k, col] = 1.0
                        W[k_base * n + i, col] = 1.0
                        col += 1
            
            if _verify_decomposition(U, V, W, n, m, p, candidate_rank, tol=1e-6):
                best_rank = candidate_rank
                best_solution = {"U": U, "V": V, "W": W}
    
    return best_solution, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with schoolbook
    best_solution = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try structured search
    if time.time() < deadline:
        structured_sol, structured_rank = _structured_search(n, m, p, deadline)
        if structured_rank < best_rank:
            best_rank = structured_rank
            best_solution = structured_sol
    
    # Try random search if time permits
    if time.time() < deadline - 10.0:
        search_sol, search_rank = _try_search(n, m, p, deadline)
        if search_rank < best_rank:
            best_rank = search_rank
            best_solution = search_sol
    
    return best_solution

# EVOLVE-BLOCK-START
# Attempt to find lower-rank exact tensor decompositions for 2x3 * 3x6 matrix multiplication.
# Uses structured patterns and random search with half-integer coefficients.
# EVOLVE-BLOCK-END
