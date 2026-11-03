"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 by 4x9 matrix.
    Uses a deterministic combinatorial search to find a lower-rank decomposition
    by leveraging the rectangular structure and trying structured patterns.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 144
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > tol:
                        return False
        return True
    
    def verify_numpy(U, V, W, tol=1e-9):
        """Faster verification using numpy."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        # Vectorize over c
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j, :]
                for k in range(p):
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    s = np.dot(u_row, v_row * w_row)
                    if abs(s - 1) > tol:
                        return False
        return True
    
    # Try to find a lower-rank decomposition using structured approaches
    
    # Approach 1: Try to use the fact that for 4x4*4x9, we might exploit
    # the rectangular structure by grouping operations
    
    # Let's try a systematic search for sparse decompositions
    # Start with ranks that are likely achievable
    
    best_result = None
    
    # Try rank 130 first (as suggested by the strategy)
    target_ranks = [130, 132, 134, 136, 138, 140, 142]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 15:
            break
        
        # Try random initialization with structured sparsity
        found = _try_structured_search(n, m, p, target_rank, deadline)
        if found is not None:
            U, V, W = found
            if verify_numpy(U, V, W):
                # Convert to list of lists
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                best_result = (U_list, V_list, W_list, target_rank)
                break
    
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = schoolbook_rank
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


def _try_structured_search(n, m, p, target_rank, deadline):
    """
    Try to find a decomposition of given rank using structured random search.
    """
    np.random.seed(42)
    random.seed(42)
    
    max_iters = 500
    
    for it in range(max_iters):
        if time.time() > deadline - 10:
            break
        
        # Initialize with random values, then try to optimize
        U = np.random.uniform(-1, 1, (n*m, target_rank))
        V = np.random.uniform(-1, 1, (m*p, target_rank))
        W = np.random.uniform(-1, 1, (p*n, target_rank))
        
        # Try a few iterations of local optimization
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            
            # Compute errors for each (i,j,k)
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(target_rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        errors.append((i, j, k, s - 1.0))
            
            # Check if we're close enough
            max_err = max(abs(e[3]) for e in errors)
            if max_err < 1e-6:
                return U, V, W
            
            # Try to reduce error by adjusting one random entry
            i, j, k, err = random.choice(errors)
            c = random.randint(0, target_rank - 1)
            
            # Compute gradient-like adjustment
            # d(sum)/dU[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c]
            grad_u = V[j*p+k, c] * W[k*n+i, c]
            grad_v = U[i*m+j, c] * W[k*n+i, c]
            grad_w = U[i*m+j, c] * V[j*p+k, c]
            
            # Simple gradient step
            lr = 0.01
            if abs(grad_u) > 1e-12:
                U[i*m+j, c] -= lr * err * grad_u
            if abs(grad_v) > 1e-12:
                V[j*p+k, c] -= lr * err * grad_v
            if abs(grad_w) > 1e-12:
                W[k*n+i, c] -= lr * err * grad_w
        
        # Final check
        if verify_numpy(U, V, W, tol=1e-6):
            return U, V, W
    
    return None
# EVOLVE-BLOCK-END
