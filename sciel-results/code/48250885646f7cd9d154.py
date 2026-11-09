import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _is_halfint(val, tol=1e-9):
    """Check if a value is an integer or half-integer."""
    v = val * 2
    return np.isclose(v, np.round(v), atol=tol)

def _clean_halfint(val, tol=1e-9):
    """Convert a value to the nearest half-integer."""
    v = val * 2
    return np.round(v) / 2.0

def _find_decomp(n, m, p, max_rank, min_rank, time_limit):
    """
    Attempts to find a rank-minimal decomposition for n x m x p.
    Returns (U, V, W) or None.
    """
    start_time = time.time()
    # U: n*m x r, V: m*p x r, W: p*n x r
    # Equations: n*m*p. Variables: p*n*r.
    
    for r in range(max_rank, min_rank - 1, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Number of random attempts per rank
        # Reduce attempts for larger r if time is tight, but r is small here.
        max_attempts = 10000
        
        for _ in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Generate random U and V with entries in {-1, -0.5, 0, 0.5, 1}
            # Using sparse random matrices might be better?
            # Let's use dense for simplicity, but restricted values.
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            
            U = np.random.choice(vals, size=(n*m, r))
            V = np.random.choice(vals, size=(m*p, r))
            
            # We need to solve for W.
            # The condition is: for all i, j, k:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This is a linear system for W.
            # Let's reshape U and V to facilitate the linear combination.
            # For a fixed (i, k), the equation is:
            # sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1?
            # No, the sum is over r for each i,j,k.
            # The unknowns are W[k*n+i, r].
            
            # We can set up a matrix A such that A * w = b.
            # w is the flattened W matrix (size p*n*r).
            # b is the vector of 1s (size n*m*p).
            
            # For each index (i, j, k), the coefficient of W[k*n+i, r] is U[i*m+j, r] * V[j*p+k, r].
            
            # Constructing the full matrix A (n*m*p) x (p*n*r) might be large.
            # n=2, m=3, p=7 -> 42 x (14*r).
            # For r=20, 42 x 280. This is small.
            
            num_eqs = n * m * p
            num_vars = p * n * r
            
            A = np.zeros((num_eqs, num_vars))
            b = np.ones(num_eqs)
            
            # Fill A
            # Index mapping:
            # Eq index: idx = i * m * p + j * p + k
            # Var index: vidx = k * n * r + i * r + r_idx  (Wait, W is (p*n, r))
            # W index in matrix: row = k*n + i, col = r_idx.
            # So vidx = (k*n + i) * r + r_idx.
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        # Coefficients for this equation
                        for r_idx in range(r):
                            u_val = U[i * m + j, r_idx]
                            v_val = V[j * p + k, r_idx]
                            if u_val == 0 or v_val == 0:
                                continue
                            var_idx = (k * n + i) * r + r_idx
                            A[eq_idx, var_idx] = u_val * v_val
            
            # Solve A * w = b
            # Use least squares if underdetermined or inconsistent.
            try:
                w, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Check if the solution is close to integer/half-integer
            w_cleaned = np.array([_clean_halfint(val) for val in w])
            
            # Verify the solution
            # Reconstruct W
            W = w_cleaned.reshape((p * n, r))
            
            # Check exactness
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        # Compute sum
                        s_val = 0.0
                        for r_idx in range(r):
                            s_val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                        if not np.isclose(s_val, 1.0, atol=1e-9):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to lists of floats/ints as required
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Ensure entries are clean
                for i in range(len(U_list)):
                    for j in range(len(U_list[i])):
                        U_list[i][j] = _clean_halfint(U_list[i][j])
                for i in range(len(V_list)):
                    for j in range(len(V_list[i])):
                        V_list[i][j] = _clean_halfint(V_list[i][j])
                for i in range(len(W_list)):
                    for j in range(len(W_list[i])):
                        W_list[i][j] = _clean_halfint(W_list[i][j])
                
                return U_list, V_list, W_list

    return None

def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication.
    Tries to find a rank lower than schoolbook (42) using random search.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    # Try to find a better decomposition
    # We can try to split p=7 into 3+4
    # Search for 2x3x3 and 2x3x4
    # If we find them, we can combine?
    # Combining decompositions is not trivial if they share indices.
    # However, if we find a decomposition for the full 2x3x7 directly, that's best.
    
    # Direct search for 2x3x7
    # Start from rank 41 down to 20
    # Limit time for this search
    search_time_limit = 60.0
    
    result = _find_decomp(n, m, p, schoolbook_rank - 1, 20, search_time_limit)
    
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
