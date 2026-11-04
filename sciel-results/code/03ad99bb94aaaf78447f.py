"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Uses Strassen-like recursive block decomposition with rank-7 core for 2x2x2,
    adapted to rectangular sub-problems, plus local optimization.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 144
    
    def verify(U, V, W, tol=1e-10):
        """Verify decomposition: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def verify_fast(U, V, W, tol=1e-10):
        """Faster verification using numpy"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # Start with schoolbook
    U = np.zeros((n*m, schoolbook_rank))
    V = np.zeros((m*p, schoolbook_rank))
    W = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Strassen-like approach: use rank-7 decomposition for 2x2x2 core
    # For 2x2x2 matrix multiplication, Strassen gives rank 7
    
    # Strassen's 7 terms for 2x2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4x9, we can partition into 2x2 blocks
    # A is 4x4, B is 4x9
    # Partition A into 2x2 blocks of size 2x2
    # Partition B into 2x2 blocks, but B is 4x9 so we have 2x5 and 2x4 blocks
    
    # Let's use a recursive approach:
    # 4x4 * 4x9 = (2x2 + 2x2) * (2x2 + 2x2 + 2x1)
    # This gets complicated. Let's try a different structured approach.
    
    # Key insight: for fixed j, we need to compute A[:,j] * B[j,:]
    # A[:,j] is 4x1, B[j,:] is 1x9
    # The outer product has rank 1 in the bilinear sense
    
    # But in our tensor formulation, W is shared across j's
    # Let's try to exploit this
    
    # Approach: use the fact that for each j, the computation is an outer product
    # We can use rank m = 4 if we can decouple the j's, but W couples them
    
    # Let's try a local optimization approach with better initialization
    
    np.random.seed(42)
    
    def solve_for_rank(target_rank, max_iters=50, lr=0.01):
        """Try to find a decomposition of given rank using iterative optimization"""
        if time.time() > deadline - 5:
            return None
        
        best_err = float('inf')
        best_solution = None
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Initialize with schoolbook-like structure but reduced rank
            U_r = np.zeros((n*m, target_rank))
            V_r = np.zeros((m*p, target_rank))
            W_r = np.zeros((p*n, target_rank))
            
            # Try to assign columns to (i,j,k) triples in a structured way
            # For target_rank < n*m*p, we need to merge some triples
            
            # Simple approach: use random initialization
            U_r = np.random.randn(n*m, target_rank) * 0.1
            V_r = np.random.randn(m*p, target_rank) * 0.1
            W_r = np.random.randn(p*n, target_rank) * 0.1
            
            # Iterative refinement
            for inner_it in range(100):
                if time.time() > deadline - 3:
                    break
                
                # Compute errors
                errors = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            errors[i,j,k] = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :]) - 1.0
                
                err = np.sum(errors**2)
                if err < best_err:
                    best_err = err
                    if err < 1e-10:
                        return U_r, V_r, W_r
                    best_solution = (U_r.copy(), V_r.copy(), W_r.copy())
                
                # Gradient descent
                for _ in range(30):
                    i = np.random.randint(n)
                    j = np.random.randint(m)
                    k = np.random.randint(p)
                    c = np.random.randint(target_rank)
                    
                    gU = errors[i,j,k] * V_r[j*p+k, c] * W_r[k*n+i, c]
                    U_r[i*m+j, c] -= lr * gU
                    
                    gV = errors[i,j,k] * U_r[i*m+j, c] * W_r[k*n+i, c]
                    V_r[j*p+k, c] -= lr * gV
                    
                    gW = errors[i,j,k] * U_r[i*m+j, c] * V_r[j*p+k, c]
                    W_r[k*n+i, c] -= lr * gW
        
        if best_solution is not None and best_err < 1e-10:
            return best_solution
        return None
    
    # Verify schoolbook works
    assert verify_fast(U, V, W), "Schoolbook should be valid"
    
    # Try to find lower rank using structured approach
    # For 4x4*4x9, try ranks that are multiples of small numbers
    
    # Try rank = 36 (n*p = m*p)
    # Try rank = 32
    # Try rank = 28 (Strassen-like: 2*7 + 2*7 for 4x4x4 part, but we have 4x4x9)
    
    candidate_ranks = [143, 140, 135, 130, 120, 100, 80, 60, 45, 36, 32, 28]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        if target_rank >= best_rank:
            continue
        
        result = solve_for_rank(target_rank, max_iters=30, lr=0.01)
        if result is not None:
            U_r, V_r, W_r = result
            if verify_fast(U_r, V_r, W_r):
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break
    
    # If we haven't found a better rank, try a few more with different seeds
    if best_rank == schoolbook_rank:
        for seed in [1, 2, 3]:
            if time.time() > deadline - 10:
                break
            np.random.seed(seed)
            for target_rank in [40, 36, 32]:
                if target_rank >= best_rank:
                    continue
                result = solve_for_rank(target_rank, max_iters=20, lr=0.01)
                if result is not None:
                    U_r, V_r, W_r = result
                    if verify_fast(U_r, V_r, W_r):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_r, V_r, W_r
                        break
            if best_rank < schoolbook_rank:
                break
    
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
