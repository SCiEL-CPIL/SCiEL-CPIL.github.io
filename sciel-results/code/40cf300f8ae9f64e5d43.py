"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product, combinations

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_linear_system(n, m, p, r):
    """
    Build the linear system for finding a rank-r decomposition.
    
    We seek U (n*m x r), V (m*p x r), W (p*n x r) such that:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    This is a system of n*m*p equations. We can fix W to be identity-like
    and solve for U and V, or use a different parameterization.
    
    Strategy: Fix W to have a specific structure and solve the resulting
    bilinear system.
    """
    # Number of equations
    num_eqs = n * m * p
    
    # We'll use the following approach:
    # For each column c, we have vectors u_c (n*m), v_c (m*p), w_c (p*n)
    # The constraint is: for each (i,j,k), sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
    
    # To make this tractable, we'll use a structured approach:
    # Fix w_c to be standard basis vectors (or close to it)
    # Then the system becomes linear in u_c and v_c
    
    # Actually, let's try a different approach: use the fact that we can
    # parameterize the decomposition using the exterior algebra structure.
    
    # For now, let's set up the full linear system and use numpy's solver
    # with a fixed W matrix.
    
    # Fix W to be a matrix where each column has a single 1
    W = np.zeros((p*n, r))
    for c in range(r):
        idx = c % (p*n)
        W[idx, c] = 1.0
    
    # Now the constraint becomes:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Since W[k*n+i, c] is 1 only when c % (p*n) == k*n+i, this simplifies
    
    # For each (i,j,k), the constraint is:
    # U[i*m+j, c0] * V[j*p+k, c0] = 1 where c0 is the column with W[k*n+i, c0] = 1
    # But we need c0 to be consistent...
    
    # This approach is getting complicated. Let's try a different strategy:
    # Use a random W and solve the resulting system.
    
    return None  # Placeholder

def _search_with_fixed_W(n, m, p, r, deadline):
    """
    Search for a rank-r decomposition by fixing W and solving for U, V.
    
    For each column c, if W[k*n+i, c] = w_cik, then:
    sum_c U[i*m+j, c] * V[j*p+k, c] * w_cik = 1
    
    If we fix W, this becomes a system of equations in U and V.
    """
    np.random.seed(12345)
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Generate a random W with sparse structure
        W = np.zeros((p*n, r))
        for c in range(r):
            # Each column of W has a few non-zero entries
            num_nonzero = 1 + (c % 2)  # 1 or 2 non-zero entries
            indices = np.random.choice(p*n, size=num_nonzero, replace=False)
            for idx in indices:
                W[idx, c] = 1.0
        
        # Now solve for U and V
        # The constraint is: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # This is still bilinear. Let's try to linearize by fixing U or V.
        
        # Fix U to be random and solve for V
        U = np.random.randn(n*m, r)
        
        # For each (i,j,k), we need:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # For fixed i, this gives us equations in V[j*p+k, c] for all j,k
        # Let's collect these equations
        
        # Actually, let's try a different approach: use alternating minimization
        # or a direct algebraic method.
        
        # For now, let's just try to verify if a random assignment works
        V = np.random.randn(m*p, r)
        
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return (U.tolist(), V.tolist(), W.tolist())
    
    return None

def _algebraic_search(n, m, p, deadline):
    """
    Use algebraic structure to find low-rank decompositions.
    
    Key insight: For matrix multiplication, we can use the following identity:
    (a+b)(c+d) = ac + ad + bc + bd
    
    This can be rewritten as:
    (a+b)(c+d) = (a+c)(b+d) - ab - cd + ... 
    
    More generally, Strassen's trick uses:
    (a+b)(c+d) = ac + (a+b-c)d = ac + ad + bd
    
    For our 2x4x7 case, let's try to find such identities.
    
    Strategy: Start from schoolbook (rank 56) and try to merge terms
    using algebraic identities.
    """
    best_rank = n * m * p
    best_result = None
    
    # Schoolbook decomposition
    def make_schoolbook(n, m, p):
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, r
    
    U, V, W, r = make_schoolbook(n, m, p)
    best_result = (U, V, W)
    best_rank = r
    
    # Try to reduce rank by finding linear dependencies
    # For each pair of terms, check if they can be merged
    
    # A term is identified by (i, j, k) with coefficient 1
    # Two terms (i1, j1, k1) and (i2, j2, k2) can potentially be merged
    # if there's an algebraic identity that combines them
    
    # For now, let's try a systematic search for rank r < 56
    # by checking if a valid decomposition exists
    
    for target_rank in range(50, best_rank):
        if time.time() > deadline:
            break
        
        # Try to find a decomposition with this rank
        # Use a simple heuristic: try random sparse patterns
        found = _try_random_sparse(n, m, p, target_rank, deadline)
        if found:
            U, V, W = found
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = target_rank
                best_result = (U, V, W)
                break
    
    return best_result

def _try_random_sparse(n, m, p, target_rank, deadline, max_trials=100):
    """
    Try random sparse patterns to find a valid decomposition.
    """
    np.random.seed(999)
    
    for _ in range(max_trials):
        if time.time() > deadline:
            break
        
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Each column corresponds to one "term" in the decomposition
        # For term c, we pick indices (i_c, j_c, k_c) and set
        # U[i_c*m+j_c, c] = a_c, V[j_c*p+k_c, c] = b_c, W[k_c*n+i_c, c] = c_c
        # such that a_c * b_c * c_c contributes to the sum
        
        for c in range(target_rank):
            i = np.random.randint(0, n)
            j = np.random.randint(0, m)
            k = np.random.randint(0, p)
            
            # Choose coefficients that might help with cancellation
            # For simplicity, use 1 for now
            U[i*m+j, c] = 1
            V[j*p+k, c] = 1
            W[k*n+i, c] = 1
        
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return (U.tolist(), V.tolist(), W.tolist())
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Try algebraic search
    result = _algebraic_search(n, m, p, deadline)
    
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
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
# EVOLVE-BLOCK-END
