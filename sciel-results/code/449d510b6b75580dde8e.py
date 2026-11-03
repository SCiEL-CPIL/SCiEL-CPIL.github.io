"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]
START_TIME = time.time()
DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Must satisfy: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c]
    Actually the tensor condition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    Wait, let me re-read the convention.
    
    The tensor for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p.
    T[i, j, k] = 1 if i is row of A, j is col of A / row of B, k is col of B.
    Actually standard: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    So the tensor has indices (i, j, k) and T[i,j,k] = 1 for valid triples.
    
    The decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    With the trace convention W[k*n+i, r], W has shape (p*n, rank).
    
    Let me verify: U shape (n*m, rank) indexed by [i*m+j, c]
    V shape (m*p, rank) indexed by [j*p+k, c]  
    W shape (p*n, rank) indexed by [k*n+i, c]
    
    Condition: for all i in [0,n), j in [0,m), k in [0,p):
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    For invalid (i,j,k) where no such triple exists... wait, all (i,j,k) triples are valid
    in the sense that T[i,j,k] = 1 for all valid index combinations.
    
    Actually for matrix multiplication, T[i,j,k] = 1 for ALL i in [0,n), j in [0,m), k in [0,p).
    The tensor is just the all-ones tensor of shape (n, m, p).
    
    Wait no. Let me reconsider. The matrix multiplication tensor T has:
    T[i, j, k] = 1 if and only if the term A[i,j] * B[j,k] contributes to C[i,k].
    Since every (i,j,k) triple with i in [0,n), j in [0,m), k in [0,p) is a valid term,
    T[i,j,k] = 1 for all valid (i,j,k). So it IS the all-ones tensor.
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    rank = U.shape[1]
    # Build the full tensor check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m+j]
                v_val = V[j*p+k]
                w_val = W[k*n+i]
                prod = u_val * v_val * w_val
                if abs(prod.sum() - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(r, n, m, p, time_budget, rng):
    """Try to find a decomposition of rank r. Returns (U, V, W) or None."""
    deadline = time.time() + time_budget
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Initialize with random small values
    U = rng.uniform(-1, 1, size=(nm, r))
    V = rng.uniform(-1, 1, size=(mp, r))
    W = rng.uniform(-1, 1, size=(pn, r))
    
    best_err = float('inf')
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Compute current error
        # For each (i,j,k), compute sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # and measure deviation from 1.
        
        # Update W: for each (k,i), we want sum_c W[k*n+i,c] * (U[i*m+j,c]*V[j*p+k,c]) = 1
        # This is a system: for fixed (i,j), sum_k W[k*n+i, :] * (U[i*m+j,:] * V[j*p+k,:]) = 1 for each k
        # Actually: for fixed i,j, and varying k:
        #   sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1
        # This is a linear system in W[k*n+i, :] for each k, but they're coupled through c.
        
        # Alternative: use a greedy approach. For each (i,j,k) triple, try to fix the residual.
        
        # Let's use a different approach: alternating minimization on the residual.
        # Error = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        
        # Gradient-based update is complex. Let's try a simpler approach:
        # For each column c, assign it to help satisfy specific (i,j,k) triples.
        
        # Actually, let me try a different strategy: 
        # Start from schoolbook and try to merge columns.
        pass
    
    # Fallback: try a structured approach
    # For rank r, try to find U, V, W with entries in {0, 0.5, 1, -0.5, -1}
    
    # Let's try a greedy column-by-column construction
    U = np.zeros((nm, r))
    V = np.zeros((mp, r))
    W = np.zeros((pn, r))
    
    # Precompute which triples need to be satisfied
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    for c in range(r):
        if time.time() > deadline:
            break
        
        # Choose a random subset of triples to assign to this column
        # For this column, we need to pick u_c in R^{nm}, v_c in R^{mp}, w_c in R^{pn}
        # such that for assigned triples (i,j,k): u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] contributes appropriately
        
        # Simple heuristic: assign this column to one (i,j,k) triple
        idx = rng.randint(0, len(triples) - 1)
        i, j, k = triples[idx]
        
        # Set U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
        # This contributes 1 to triple (i,j,k)
        U[i*m+j, c] = 1.0
        V[j*p+k, c] = 1.0
        W[k*n+i, c] = 1.0
    
    # Check if this works
    if _verify_fast(U, V, W, n, m, p):
        return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Known lower bounds and upper bounds for matrix multiplication rank
    # For 2x3 * 3x9, let's try various ranks
    # The rank of matrix multiplication tensor for n x m times m x p is at most n*m*p
    # and at least max(n*p, m*p, n*m) (from slicing arguments)
    
    # Lower bound: max(n*p, m*p, n*m) = max(18, 27, 6) = 27
    # Upper bound: n*m*p = 54
    
    # Try to find the minimum rank
    # Let's try ranks from lower bound upward
    
    lower_bound = max(n * p, m * p, n * m)
    
    rng = random.Random(42)
    
    # Try schoolbook first as a safe fallback
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = (U_sb, V_sb, W_sb, schoolbook_rank)
    
    # Try to improve
    for r in range(lower_bound, schoolbook_rank):
        if time.time() > DEADLINE:
            break
        
        # Try a few random initializations
        found = False
        for attempt in range(10):
            if time.time() > DEADLINE:
                break
            
            result = _try_rank(r, n, m, p, 2.0, rng)
            if result is not None:
                U, V, W = result
                if _verify_fast(U, V, W, n, m, p):
                    best_result = (U, V, W, r)
                    found = True
                    break
        
        if found:
            break
    
    U, V, W, rank = best_result
    
    # Convert to lists of lists with integer/half-integer entries
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                # Round to nearest half-integer
                rounded = round(val * 2) / 2
                if abs(rounded - int(rounded)) < 1e-9:
                    row_list.append(int(rounded))
                else:
                    row_list.append(rounded)
            result.append(row_list)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
