"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return U, V, W as lists for the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly satisfies:
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    Returns True if valid (within numerical tolerance), False otherwise.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _is_half_integer(x):
    """Check if x is an integer or half-integer (i.e., 2*x is an integer)."""
    return abs(2 * x - round(2 * x)) < 1e-9

def _verify_entries(U, V, W):
    """Check all entries are integers or half-integers."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_integer(val):
                    return False
    return True

def _try_sparse_rank(n, m, p, rank, deadline, max_iters=5000):
    """
    Try to find a rank-`rank` decomposition with sparse half-integer entries.
    Uses iterative refinement starting from schoolbook and attempting to
    zero out entries while maintaining the decomposition.
    """
    if time.time() > deadline:
        return None
    
    # Start from schoolbook and try to reduce rank by finding
    # linear dependencies and eliminating terms.
    
    # For [2, m, p], let's try a construction that exploits
    # the structure of having 2 output rows.
    
    # Key insight: For each j, we can combine the two products
    # A[0,j]*B[j,k] and A[1,j]*B[j,k] using bilinear identities.
    
    # Let's try a systematic search over small rank values.
    # For rank r, we need to find U (10 x r), V (30 x r), W (12 x r)
    # such that for all i,j,k: sum_r U[i*5+j,r]*V[j*6+k,r]*W[k*2+i,r] = 1
    
    # This is a system of 60 cubic equations in 52*r variables.
    
    # For small r, we can try random initialization and Newton's method.
    
    import numpy as np
    
    best_result = None
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        np.random.seed(attempt * 12345 + 67890)
        
        # Random initialization with half-integers
        U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank))
        V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank))
        W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, rank))
        
        # Check if this works
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_result = (U.tolist(), V.tolist(), W.tolist())
            break
    
    return best_result

def _try_winograd_like(n, m, p, deadline):
    """
    Try a Winograd-like scheme for [2, m, p].
    
    For each group of terms, we can use the identity:
    x1*y1 + x2*y2 = ((x1+x2)*(y1+y2) - (x1-x2)*(y1-y2))/2
    
    This reduces the number of multiplications from 2 to 3, but with
    half-integer coefficients.
    """
    if time.time() > deadline:
        return None
    
    # For [2, 5, 6], let's try to pair up terms.
    # We have 5*6 = 30 products for each row, so 60 total.
    
    # One approach: for each k, pair the two rows' contributions.
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    
    # We can write:
    # C[0,k] + C[1,k] = sum_j (A[0,j]+A[1,j])*B[j,k]
    # C[0,k] - C[1,k] = sum_j (A[0,j]-A[1,j])*B[j,k]
    
    # Then:
    # C[0,k] = 0.5*((C[0,k]+C[1,k]) + (C[0,k]-C[1,k]))
    # C[1,k] = 0.5*((C[0,k]+C[1,k]) - (C[0,k]-C[1,k]))
    
    # This doesn't reduce rank directly, but suggests a structure.
    
    # Let's try a different pairing: for each j, pair k values.
    # For fixed j, we have 6 products: A[i,j]*B[j,k] for k=0..5
    # We can pair them: (k=0,k=1), (k=2,k=3), (k=4,k=5)
    
    # For each pair (k1, k2):
    # A[i,j]*B[j,k1] + A[i,j]*B[j,k2] = A[i,j]*(B[j,k1]+B[j,k2])
    # This doesn't help directly.
    
    # Let's try a more sophisticated approach using the fact that
    # we have 2 rows.
    
    # For each j, define:
    # s_j = A[0,j] + A[1,j]
    # d_j = A[0,j] - A[1,j]
    
    # Then:
    # A[0,j] = (s_j + d_j)/2
    # A[1,j] = (s_j - d_j)/2
    
    # So:
    # C[0,k] = sum_j ((s_j + d_j)/2)*B[j,k] = 0.5*sum_j s_j*B[j,k] + 0.5*sum_j d_j*B[j,k]
    # C[1,k] = sum_j ((s_j - d_j)/2)*B[j,k] = 0.5*sum_j s_j*B[j,k] - 0.5*sum_j d_j*B[j,k]
    
    # This is a valid decomposition, but it still has rank 2*m*p = 60.
    
    # To reduce rank, we need to find linear dependencies.
    
    # For now, return None to indicate no improvement found.
    return None

def _try_groebner_search(n, m, p, max_rank, deadline):
    """
    Try to find a low-rank decomposition by solving the polynomial system.
    This is a simplified approach that tries small ranks.
    """
    if time.time() > deadline:
        return None
    
    import numpy as np
    
    # For [2, 5, 6], let's try ranks from 40 to 59.
    # The system is: for all i,j,k, sum_r U[i*5+j,r]*V[j*6+k,r]*W[k*2+i,r] = 1
    
    # This is a system of 60 cubic equations.
    
    # For each rank r, we have 10*r + 30*r + 12*r = 52*r unknowns.
    
    # Let's try a random search with local refinement.
    
    best_result = None
    best_rank = max_rank + 1
    
    for r in range(max(20, max_rank - 20), max_rank):
        if time.time() > deadline:
            break
        
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            np.random.seed(r * 1000 + attempt)
            
            # Initialize with random half-integers
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, r))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, r))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, r))
            
            # Check if valid
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for rr in range(r):
                            s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_result = (U.tolist(), V.tolist(), W.tolist())
                best_rank = r
                break
        
        if best_result is not None:
            break
    
    return best_result if best_result is not None else None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U, V, W)
    
    # Verify schoolbook
    if _verify_decomposition(U, V, W, n, m, p) and _verify_entries(U, V, W):
        pass
    
    # Try to find a better decomposition
    # First, try the Groebner-style search for low ranks
    result = _try_groebner_search(n, m, p, best_rank, deadline)
    if result is not None:
        best_result = result
        best_rank = len(result[0][0])
    
    # Try Winograd-like schemes
    result = _try_winograd_like(n, m, p, deadline)
    if result is not None:
        r = len(result[0][0])
        if r < best_rank:
            best_result = result
            best_rank = r
    
    # Try sparse rank search
    for r in range(max(20, best_rank - 10), best_rank):
        if time.time() > deadline:
            break
        result = _try_sparse_rank(n, m, p, r, deadline)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
