import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _trivial_decomp(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_decomposition(n, m, p, rank, deadline, max_iters=200):
    """
    Try to find a rank-r decomposition by random search.
    For a given rank r, we need U (n*m x r), V (m*p x r), W (p*n x r)
    such that for all i, k: sum_j A[i,j]*B[j,k] = sum_t U[i*m+j][t]*V[j*p+k][t]*W[k*n+i][t]
    
    This is equivalent to: for each (i,k), the vector of coefficients
    sum_t U[i*m+j][t]*V[j*p+k][t] over j must equal W[k*n+i][t] for each t.
    
    Actually, the condition is:
    For each t, define U_t (n x m), V_t (m x p), W_t (p x n) where
    U_t[i,j] = U[i*m+j][t], V_t[j,k] = V[j*p+k][t], W_t[k,i] = W[k*n+i][t]
    
    Then we need: sum_t U_t ⊗ V_t ⊗ W_t = T (the matrix multiplication tensor)
    
    More precisely: for each (i,k): sum_j A[i,j]*B[j,k] = sum_t (U_t[i,j] * V_t[j,k]) * W_t[k,i]
    Wait, that's not quite right either. Let me re-derive.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    With rank-r decomposition:
    C[i,k] = sum_{t=1}^r (sum_j U[i,j,t] * A[i,j]) * (sum_k V[j,k,t] * B[j,k]) ... no.
    
    Actually the standard form is:
    C[i,k] = sum_t (sum_j U[i*m+j, t] * A[i,j]) * (sum_j V[j*p+k, t] * B[j,k]) * W[k*n+i, t]
    
    No, that's not right either. Let me think more carefully.
    
    The tensor decomposition is:
    T = sum_t u_t ⊗ v_t ⊗ w_t
    
    where u_t is a vector of size n*m, v_t is of size m*p, w_t is of size p*n.
    
    And the multiplication is:
    C[i,k] = sum_{j} A[i,j] * B[j,k] = sum_t (sum_j u_t[i*m+j] * A[i,j]) * (sum_j v_t[j*p+k] * B[j,k]) * w_t[k*n+i]
    
    Hmm, that's still not quite standard. Let me look at this differently.
    
    Actually, the correct formulation is:
    C[i,k] = sum_t U_t[i,j] * A[i,j] * V_t[j,k] * B[j,k] * W_t[k,i]
    
    No wait. The standard tensor decomposition for matrix multiplication is:
    
    C[i,k] = sum_t (sum_j U[i,j,t] * A[i,j]) * (sum_j V[j,k,t] * B[j,k]) ... 
    
    I think I'm overcomplicating this. Let me use the formulation from the seed:
    
    The tensor T has T[i,j,k] = 1 for all valid (i,j,k).
    The multiplication is: C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    
    A rank-r decomposition means:
    T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    Wait, that would mean T[i,j,k] is decomposed as a sum of outer products, but T[i,j,k] = 1 for all (i,j,k), so:
    
    1 = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] for all i, j, k
    
    But that's a constraint on U, V, W. However, this doesn't seem right because the "1" is the same for all (i,j,k).
    
    Let me reconsider. The matrix multiplication tensor is actually:
    T_{i,j,k} where the multiplication is C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor is defined such that:
    C[i,k] = sum_{j} A[i,j] * B[j,k] * T[i,j,k]
    
    where T[i,j,k] = 1 for all valid indices.
    
    A rank-r decomposition of this tensor means we can write:
    T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    So we need: for all i, j, k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    This is a system of n*m*p equations in r*(n*m + m*p + p*n) unknowns.
    
    For 4x5x6: 120 equations, r*(20+30+24) = 74r unknowns.
    For r=120: 74*120 = 8880 unknowns, 120 equations - very underdetermined.
    For r=30: 74*30 = 2220 unknowns, 120 equations - still underdetermined.
    
    But wait, this formulation seems too simple. The issue is that T[i,j,k] = 1 for all (i,j,k) is a very specific tensor. Let me verify with the seed code.
    
    Looking at the seed:
    U[i*m+j][column] = 1
    V[j*p+k][column] = 1  
    W[k*n+i][column] = 1
    
    So for each (i,j,k) triple, there's one rank-1 term with all entries 1.
    Then sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1 (the term where t corresponds to (i,j,k)) + 0 (other terms) = 1.
    
    Yes, so the constraint is indeed: for all i, j, k: sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1.
    
    This is a system of 120 cubic equations. We can try to solve it using random initialization and iterative methods.
    
    One approach: fix U and V randomly, then for each k*n+i, we need:
    W[k*n+i, t] = 1 / (sum_j U[i*m+j,t] * V[j*p+k,t]) ... no, that's not right.
    
    Actually, for a fixed t, the product U[i*m+j,t] * V[j*p+k,t] is a bilinear form in j. Let me define:
    For fixed i, k: sum_j U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] should contribute to the sum.
    
    Wait, I think I need to be more careful. The constraint is:
    For all (i,j,k): sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    
    This is a single equation for each (i,j,k). Let me fix i and k, and vary j:
    For fixed i, k: sum_t W[k*n+i,t] * (sum_j ... ) ... no, j is also an index in U and V.
    
    Let me rewrite: for fixed i, k, the constraint for all j is:
    sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1 for all j
    
    This is a system of m equations in the variables W[k*n+i,t] for t=1..r, given U and V.
    
    But U[i*m+j,t] * V[j*p+k,t] is a product, so it's not linear in W.
    
    Actually, if we fix U and V, then for each (i,k), the constraint becomes:
    For all j: sum_t (U[i*m+j,t] * V[j*p+k,t]) * W[k*n+i,t] = 1
    
    This IS linear in W! For fixed i, k, we have m equations:
    sum_t M[j,t] * W[k*n+i,t] = 1 for j=0..m-1
    
    where M[j,t] = U[i*m+j,t] * V[j*p+k,t]
    
    So for each (i,k), we have a linear system M_{i,k} * W_{k*n+i} = 1 (vector of ones of length m).
    
    If M_{i,k} is m x r, and we need it to have rank at least m (or at least the system to be consistent), we can solve for W.
    
    So the algorithm is:
    1. Randomly initialize U (n*m x r) and V (m*p x r)
    2. For each (i,k), form M_{i,k}[j,t] = U[i*m+j,t] * V[j*p+k,t] (m x r matrix)
    3. Solve M_{i,k} * W_{k*n+i} = 1 (m equations, r unknowns)
    4. If all systems are consistent, we have a valid decomposition.
    
    The system is consistent if the vector of ones is in the column space of M_{i,k}.
    
    Let me implement this.
    """
    n, m, p = 4, 5, 6
    np.random.seed(42)  # For reproducibility within a call
    
    best_W = None
    best_U = None
    best_V = None
    
    for iter in range(max_iters):
        if time.time() > deadline:
            break
        
        # Random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        
        # For each (i,k), solve for W[k*n+i, :]
        W = np.zeros((p * n, rank))
        success = True
        
        for i in range(n):
            for k in range(p):
                # Form M[j,t] = U[i*m+j, t] * V[j*p+k, t] for j=0..m-1, t=0..rank-1
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                # Solve M * w = ones(m)
                # M is m x rank, w is rank x 1
                try:
                    # Use least squares
                    w, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    # Check if the solution is valid
                    if np.allclose(M @ w, np.ones(m), atol=1e-10):
                        W[k * n + i, :] = w
                    else:
                        success = False
                        break
                except np.linalg.LinAlgError:
                    success = False
                    break
            
            if not success:
                break
        
        if success:
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140 second deadline, 10 second safety margin
    
    # Trivial decomposition as fallback
    trivial = _trivial_decomp(n, m, p)
    best_rank = n * m * p
    best_result = trivial
    
    # Lower bound
    lb = max(n * p, n * m, m * p)
    
    # Try ranks from a promising range
    # The rank is at least lb = 30 and at most 120
    # Try some candidate ranks
    candidate_ranks = list(range(lb + 10, lb - 1, -1))  # 40 down to 30
    for r in [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115]:
        if lb <= r < best_rank and r not in candidate_ranks:
            candidate_ranks.append(r)
    
    candidate_ranks = sorted(set(candidate_ranks), reverse=True)
    
    for rank in candidate_ranks:
        if time.time() > deadline:
            break
        if rank >= best_rank:
            continue
        
        result = _try_decomposition(n, m, p, rank, deadline, max_iters=50)
        if result is not None:
            best_rank = rank
            best_result = result
    
    return best_result
# EVOLVE-BLOCK-END
