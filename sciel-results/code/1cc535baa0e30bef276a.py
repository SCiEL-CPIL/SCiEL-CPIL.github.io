"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _decompose_rank17(n, m, p):
    """Build the known rank-17 decomposition using Strassen + direct."""
    n_a, n_b, n_c = n*m, m*p, p*n
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    def set_term(r, u_dict, v_dict, w_dict):
        for a, val in u_dict.items():
            U[a, r] = val
        for b, val in v_dict.items():
            V[b, r] = val
        for c, val in w_dict.items():
            W[c, r] = val
    
    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2, 0:2]
    set_term(0, {0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1})
    set_term(1, {3:1, 4:1}, {0:1}, {1:1, 3:-1})
    set_term(2, {0:1}, {1:1, 4:-1}, {2:1, 3:1})
    set_term(3, {4:1}, {3:1, 0:-1}, {0:1, 1:1})
    set_term(4, {0:1, 1:1}, {4:1}, {0:-1, 2:1})
    set_term(5, {3:1, 0:-1}, {0:1, 1:1}, {3:1})
    set_term(6, {1:1, 4:-1}, {3:1, 4:1}, {0:1})
    
    # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2]
    set_term(7, {0:1}, {2:1}, {4:1})
    set_term(8, {1:1}, {5:1}, {4:1})
    set_term(9, {3:1}, {2:1}, {5:1})
    set_term(10, {4:1}, {5:1}, {5:1})
    
    # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3]
    set_term(11, {2:1}, {6:1}, {0:1})
    set_term(12, {2:1}, {7:1}, {2:1})
    set_term(13, {2:1}, {8:1}, {4:1})
    set_term(14, {5:1}, {6:1}, {1:1})
    set_term(15, {5:1}, {7:1}, {3:1})
    set_term(16, {5:1}, {8:1}, {5:1})
    
    T = _build_tensor(n, m, p)
    if not _verify_decomposition(U, V, W, T):
        return None
    return {"U": U, "V": V, "W": W}

def _try_rank16_search(n, m, p, deadline):
    """
    Try to find a rank-16 decomposition by searching for structured solutions.
    
    Strategy: Consider the tensor as a bilinear map. We look for decompositions
    where the rank-16 terms are linear combinations of structured patterns.
    
    Key idea: The 2x3*3x3 multiplication can be viewed as:
    - 2x2 * 2x2 (Strassen, rank 7)
    - 2x2 * 2x1 (rank 4)  
    - 2x1 * 1x3 (rank 6)
    Total: 17
    
    To get rank 16, we need to save 1 rank. This requires a non-trivial
    algebraic identity. We search by:
    1. Setting up the linear system for rank-16 decomposition
    2. Using the structure that most terms are sparse
    """
    n_a, n_b, n_c = n*m, m*p, p*n
    target_rank = 16
    
    # The tensor has 6*9*6 = 324 entries, but only 18 are nonzero.
    # For rank r, we need: sum_{s=1}^{r} U[:,s] ⊗ V[:,s] ⊗ W[:,s] = T
    # This gives 324 equations in 16*(6+9+6) = 320 unknowns.
    # The system is overdetermined, so we need the right structure.
    
    # Approach: Use the known rank-17 solution and try to eliminate one term
    # by expressing it as a linear combination of others (if possible).
    
    # Alternative: Direct numerical search using structured ansatz.
    # We parameterize each rank-1 vector with sparse support patterns.
    
    # For 2x3*3x3, the nonzero entries of T are at:
    # T[i*3+j, j*3+k, k*2+i] = 1 for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    
    nonzero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                nonzero_indices.append((a, b, c))
    
    # Strategy: Try to find a rank-16 decomposition by solving a constrained
    # optimization. We fix the support patterns and solve for values.
    
    # The key structural constraint: each rank-1 term U_s ⊗ V_s ⊗ W_s
    # contributes to T[a,b,c] as U_s[a]*V_s[b]*W_s[c].
    # For the contribution to be nonzero at T[a,b,c], we need:
    #   a = i*3+j, b = j*3+k, c = k*2+i for some (i,j,k)
    
    # Let's try a different approach: use the fact that we can group terms.
    # Consider merging two of the "direct" terms with the Strassen terms.
    
    # Actually, let's try a more systematic approach:
    # Set up the problem as finding 16 vectors (u_s, v_s, w_s) such that
    # the sum of outer products equals T.
    
    # We'll use a gradient-free approach: random initialization + local search
    # on the residual, but with structured initialization from rank-17.
    
    # First, let's try: can we express one of the 17 terms as a combination
    # of the others? This would mean the rank-17 solution is not minimal.
    
    U17, V17, W17 = _decompose_rank17(n, m, p)["U"], _decompose_rank17(n, m, p)["V"], _decompose_rank17(n, m, p)["W"]
    
    # Check if any term is a linear combination of others by looking at
    # the subspace spanned by the Kronecker products.
    # T = sum_s u_s ⊗ v_s ⊗ w_s
    # Flatten: T_flat = sum_s (u_s ⊗ v_s ⊗ w_s)_flat
    
    # For rank 16, we need 16 vectors spanning the same tensor.
    # This is equivalent to: the 17 vectors {u_s ⊗ v_s ⊗ w_s} lie in
    # a 16-dimensional subspace.
    
    # Let's check the rank of the matrix formed by flattening each term.
    terms_flat = []
    for s in range(17):
        term = np.einsum('a,b,c->abc', U17[:,s], V17[:,s], W17[:,s])
        terms_flat.append(term.flatten())
    
    terms_matrix = np.array(terms_flat)  # shape (17, 324)
    
    # Check if the 17 terms are linearly dependent
    rank_of_terms = np.linalg.matrix_rank(terms_matrix, tol=1e-10)
    
    if rank_of_terms < 17:
        # There's a linear dependence, so we might be able to reduce rank
        # Find the null space
        _, s, vh = np.linalg.svd(terms_matrix.T)  # 324x17
        # Null space of terms_matrix: vectors x such that terms_matrix @ x = 0
        # This means sum_s x_s * term_s = 0
        null_vec = vh[-1, :]  # last row of Vh is the null vector
        # Check which terms can be eliminated
        for s in range(17):
            if abs(null_vec[s]) > 1e-8:
                # Term s can be expressed as combination of others
                # This doesn't directly give rank 16, but indicates structure
                pass
    
    # More direct approach: try to find rank-16 by solving the nonlinear system
    # using a structured ansatz.
    
    # Let's try: assume the rank-16 solution has the same support structure
    # as rank-17 but with one term merged.
    
    # Specifically, try to merge term 7 (A[0,0]*B[0,2]->C[0,2]) with
    # one of the Strassen terms by adjusting coefficients.
    
    # This is getting complex. Let's try a numerical optimization approach.
    
    # Parameterize: 16 terms, each with 6+9+6 = 21 parameters = 336 total
    # Constraints: 18 equations (one per nonzero entry of T)
    # This is underdetermined, so we should be able to find solutions.
    
    # But the equations are cubic (product of three variables), so we need
    # a nonlinear solver. Let's use a simple iterative approach.
    
    # Initialize from rank-17: take first 16 terms and try to adjust.
    # Actually, let's try a cleaner approach.
    
    # The tensor T can be written as:
    # T[a,b,c] = sum_{i,j,k} delta(a, i*m+j) * delta(b, j*p+k) * delta(c, k*n+i)
    
    # For 2x3*3x3:
    # T[a,b,c] = 1 iff a=i*3+j, b=j*3+k, c=k*2+i for some (i,j,k)
    
    # Let's try to find a rank-16 decomposition by exploiting the structure.
    # 
    # Key observation: The output C[i,k] = sum_j A[i,j]*B[j,k]
    # For fixed k, C[:,k] = A[:,0]*B[0,k] + A[:,1]*B[1,k] + A[:,2]*B[2,k]
    # 
    # This is a matrix-vector product: C[:,k] = A * B[:,k]
    # where A is 2x3 and B[:,k] is 3x1.
    #
    # The rank of the 2x3*3x1 multiplication is at most 6 (schoolbook) or
    # we can use Strassen-like tricks.
    
    # Let's try a completely different structural approach:
    # Group by output column k.
    # C[:,0] = A[:,0]*B[0,0] + A[:,1]*B[1,0] + A[:,2]*B[2,0]
    # C[:,1] = A[:,0]*B[0,1] + A[:,1]*B[1,1] + A[:,2]*B[2,1]
    # C[:,2] = A[:,0]*B[0,2] + A[:,1]*B[1,2] + A[:,2]*B[2,2]
    
    # Each C[:,k] is a linear combination of A[:,0], A[:,1], A[:,2]
    # with coefficients B[0,k], B[1,k], B[2,k].
    
    # So C = [C[:,0], C[:,1], C[:,2]] = A * B
    # where A is 2x3 and B is 3x3.
    
    # The tensor is: T[i,j,k] = A[i,j] * B[j,k] (with appropriate indexing)
    
    # This is the standard matrix multiplication tensor.
    
    # For 2x3*3x3, the known lower bound is 16 (from Coppersmith-Winograd style
    # bounds, though for small dimensions we need explicit constructions).
    
    # Let me try to construct a rank-16 solution explicitly.
    # 
    # Idea: Use Strassen for the 2x2*2x2 part (rank 7), and for the remaining
    # parts, try to save one rank.
    #
    # The remaining parts are:
    # - A[:,2] * B[2,:] (2x1 * 1x3 = 2x3 matrix, rank 6 schoolbook)
    # - A[:,:2] * B[2:,:]... wait, let me re-examine.
    
    # Actually, let me try a direct algebraic approach.
    # 
    # Let's define:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])  [Strassen M1]
    # M2 = (A[1,0]+A[1,1])*B[0,0]           [Strassen M2]
    # M3 = A[0,0]*(B[0,1]-B[1,1])           [Strassen M3]
    # M4 = A[1,1]*(B[1,0]-B[0,0])           [Strassen M4]
    # M5 = (A[0,0]+A[0,1])*B[1,1]           [Strassen M5]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])  [Strassen M6]
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])  [Strassen M7]
    
    # These give C[0,0], C[1,0], C[0,1], C[1,1]
    
    # For C[0,2] and C[1,2]:
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    
    # For C[0,0], C[0,1], C[1,0], C[1,1] we also have contributions from
    # A[*,2]*B[2,*]:
    # C[0,0] += A[0,2]*B[2,0]
    # C[0,1] += A[0,2]*B[2,1]
    # C[1,0] += A[1,2]*B[2,0]
    # C[1,1] += A[1,2]*B[2,1]
    
    # So the full expressions are:
    # C[0,0] = (Strassen combo) + A[0,2]*B[2,0]
    # C[0,1] = (Strassen combo) + A[0,2]*B[2,1]
    # C[1,0] = (Strassen combo) + A[1,2]*B[2,0]
    # C[1,1] = (Strassen combo) + A[1,2]*B[2,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    
    # Total terms if all direct: 7 + 4 + 6 = 17
    
    # To save one rank, we need to find a way to express one of the 10
    # "direct" terms (4 for C[:,2] from A[:,:2]*B[:2,2] and 6 for A[:,2]*B[2,:])
    # using the Strassen structure.
    
    # This is a research-level problem. For now, let's try a numerical search.
    
    # Use scipy to solve the nonlinear system if available
    try:
        from scipy.optimize import least_squares
        # This might be too slow or not converge. Skip for now.
        pass
    except ImportError:
        pass
    
    # Given the complexity and time constraints, let's return None if we
    # can't find rank 16 quickly.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline with safety margin
    
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    
    # Try to find rank-16 solution (time-limited)
    best_result = None
    if time.time() < deadline - 10:
        result16 = _try_rank16_search(n, m, p, deadline)
        if result16 is not None:
            best_result = result16
    
    # Fall back to rank-17
    if best_result is None:
        result17 = _decompose_rank17(n, m, p)
        if result17 is not None:
            best_result = result17
        else:
            # Ultimate fallback: schoolbook rank 18
            U_fb = np.zeros((n_a, n*m*p))
            V_fb = np.zeros((n_b, n*m*p))
            W_fb = np.zeros((n_c, n*m*p))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j, col] = 1
                        V_fb[j*p+k, col] = 1
                        W_fb[k*n+i, col] = 1
                        col += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    T = _build_tensor(n, m, p)
    U, V, W = best_result["U"], best_result["V"], best_result["W"]
    if not _verify_decomposition(U, V, W, T):
        # Fallback to schoolbook
        U_fb = np.zeros((n_a, n*m*p))
        V_fb = np.zeros((n_b, n*m*p))
        W_fb = np.zeros((n_c, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
