"""Matrix multiplication 3x3 by 3x6: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x3 matrix by a 3x6 matrix.
    
    The decomposition uses the structure of matrix multiplication:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For 3x3 * 3x6, the naive schoolbook algorithm uses 3*3*6 = 54 multiplications.
    We can do better by exploiting the tensor structure.
    
    The key insight: we can decompose the 3x6 matrix B into two 3x3 blocks,
    B = [B1 | B2], and compute A*B1 and A*B2 separately.
    
    For 3x3 * 3x3, the minimum rank is known to be 19 (from Coppersmith-Winograd
    and subsequent improvements, though the exact minimum for 3x3 is 19).
    
    However, a simpler approach: use the fact that we're multiplying by a 3x6 matrix.
    We can view this as computing 6 column vectors of the result, each being a
    linear combination of the columns of A weighted by B.
    
    A practical approach with rank < 54:
    
    Method 1: Block decomposition
    Split B into two 3x3 blocks. For each 3x3 multiplication, use a rank-19 
    decomposition (if we had one). Total rank = 2*19 = 38.
    
    Method 2: Direct optimization
    Try to find a rank-r decomposition directly for the 3x3*3x6 tensor.
    
    Method 3: Use the structure more cleverly
    Note that A is 3x3 and B is 3x6. The product C is 3x6.
    C[:, 0:3] = A * B[:, 0:3]
    C[:, 3:6] = A * B[:, 3:6]
    
    For each 3x3 * 3x3 multiplication, we can use the following rank-19 algorithm
    (based on known results):
    
    Actually, let me use a simpler but still improved approach.
    
    The standard result: 3x3 * 3x3 can be done in 19 multiplications.
    So 3x3 * 3x6 = 2 * (3x3 * 3x3) can be done in 2*19 = 38 multiplications.
    
    But we need an explicit rank-19 decomposition for 3x3*3x3.
    
    Let me use a known explicit construction. One such construction is based on
    the following idea (from Bini et al. or similar):
    
    Actually, for practical purposes within the time limit, let me try a different
    approach: use random initialization and gradient-based optimization to find
    a low-rank decomposition, then verify and round.
    
    Given the time constraint (150s), let me try:
    1. Start with schoolbook rank 54
    2. Try to reduce rank using optimization
    3. Return the best valid decomposition found
    
    Actually, let me think about this more carefully. The tensor for 3x3*3x6 has
    dimensions (3*3) x (3*6) x (6*3) = 9 x 18 x 18.
    
    A rank-r decomposition means:
    T[i, j, k] = sum_{c=1}^{r} U[i,c] * V[j,c] * W[k,c]
    
    where T[i, j, k] = 1 if i = a*m+j and k = p*n+a for some valid a, else 0.
    Wait, let me be more careful about the indexing.
    
    For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p:
    C[i, k] = sum_j A[i, j] * B[j, k]
    
    The tensor T has entries T[i, j, k] = 1 if the (i,k) entry of C depends on
    A[i,j] and B[j,k], which is always true for valid i, j, k.
    
    More precisely, the multiplication tensor is:
    T = sum_{i,j,k} e_i (in output) ⊗ e_i (in A) ⊗ e_{j,k} (in B)
    
    Wait, I need to be more careful. Let me define:
    - U has shape (n*m, rank): indexed by (i, j) for A
    - V has shape (m*p, rank): indexed by (j, k) for B  
    - W has shape (p*n, rank): indexed by (k, i) for C (output)
    
    The condition is:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k
    where i in [0,n), j in [0,m), k in [0,p)
    
    And for all other combinations of indices in U, V, W, the sum should be 0
    (but actually, the tensor only has non-zero entries for valid (i,j,k) triples,
    and the decomposition must reproduce exactly those entries).
    
    Actually, the tensor is defined as:
    T[a, b, c] where a indexes A (n*m elements), b indexes B (m*p elements), 
    c indexes C (p*n elements).
    
    T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i, j, k
    T[a, b, c] = 0 otherwise
    
    The rank-r decomposition satisfies:
    T[a, b, c] = sum_{r=1}^{rank} U[a, r] * V[b, r] * W[c, r]
    
    Let me implement a search for a low-rank decomposition.
    
    Given the complexity, let me try a practical approach:
    1. Use the block decomposition: split into two 3x3*3x3 multiplications
    2. For each, use a known low-rank algorithm
    
    For 3x3*3x3, I'll use a rank-19 algorithm if I can construct one, 
    otherwise fall back to schoolbook.
    
    Let me try to construct a rank-19 decomposition for 3x3*3x3.
    This is a well-known result but the explicit coefficients are complex.
    
    Alternative: use the following trick.
    For 3x3*3x3, we can write:
    C = A * B
    
    Using the identity:
    (a+b)(c+d) = ac + ad + bc + bd - (a-b)(c-d)/2 - (a+b+c+d)^2/4 + ...
    
    This gets complicated. Let me just use a numerical optimization approach.
    
    Strategy:
    1. Initialize with a random rank-r decomposition
    2. Use gradient descent to minimize the error
    3. Check if the error is small enough, then round to rationals
    4. Try decreasing r until we find a valid decomposition
    
    Given time constraints, let me limit the search.
    """
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Build the target tensor
    # T[a, b, c] where a in [0, n*m), b in [0, m*p), c in [0, p*n)
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i
    
    nm = n * m  # 9
    mp = m * p  # 18
    pn = p * n  # 18
    
    # Build sparse representation of T
    # For each (i, j, k), T[i*m+j, j*p+k, k*n+i] = 1
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c, 1.0))
    
    # Try to find a low-rank decomposition
    # Start with schoolbook rank and try to reduce
    
    def make_schoolbook():
        """Create the schoolbook rank-54 decomposition."""
        rank = n * m * p  # 54
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    def compute_error(U, V, W):
        """Compute the maximum absolute error of the decomposition."""
        max_err = 0.0
        for (a, b, c, val) in tensor_entries:
            s = np.dot(U[a, :], V[b, :] * W[c, :])
            err = abs(s - val)
            if err > max_err:
                max_err = err
        return max_err
    
    def try_rank(r, max_iters=500, lr=0.01, seed=42):
        """Try to find a rank-r decomposition using gradient descent."""
        np.random.seed(seed)
        
        # Initialize randomly
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        
        best_err = compute_error(U, V, W)
        
        for it in range(max_iters):
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for (a, b, c, val) in tensor_entries:
                # s = sum_c U[a,c] * V[b,c] * W[c,c]
                # ds/dU[a,c'] = V[b,c'] * W[c,c']
                # ds/dV[b,c'] = U[a,c'] * W[c,c']
                # ds/dW[c,c'] = U[a,c'] * V[b,c']
                error = np.dot(U[a, :], V[b, :] * W[c, :]) - val
                
                dU[a, :] += error * V[b, :] * W[c, :]
                dV[b, :] += error * U[a, :] * W[c, :]
                dW[c, :] += error * U[a, :] * V[b, :]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            err = compute_error(U, V, W)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-10:
                break
        
        return best_U, best_V, best_W, best_err
    
    # First, get the schoolbook solution as a fallback
    U_school, V_school, W_school = make_schoolbook()
    err_school = compute_error(U_school, V_school, W_school)
    
    # Try to find lower-rank decompositions
    # Start from rank 38 (2*19 for two 3x3 blocks) and go down
    best_result = (U_school, V_school, W_school, err_school, n * m * p)
    
    deadline = time.time() + 140  # Leave 10s margin
    
    # Try various ranks
    for r in range(38, 54):
        if time.time() > deadline:
            break
        
        # Try multiple seeds
        for seed in range(5):
            if time.time() > deadline:
                break
            
            U_try, V_try, W_try, err = try_rank(r, max_iters=200, lr=0.05, seed=seed)
            
            if err < 1e-10 and r < best_result[4]:
                best_result = (U_try, V_try, W_try, err, r)
                break  # Found a valid decomposition for this rank, move to lower rank
    
    # Use the best result
    U_best, V_best, W_best, err_best, rank_best = best_result
    
    # Convert to list format as required
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
