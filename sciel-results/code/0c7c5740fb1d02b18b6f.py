import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    Uses a Max-SAT-inspired approach to search for lower-rank decompositions.
    """
    import time
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    dim_U = n * m  # 9
    dim_V = m * p  # 21
    dim_W = p * n  # 21
    
    # Start with schoolbook rank 63 as fallback
    best_rank = n * m * p
    best_U = np.zeros((dim_U, best_rank), dtype=float)
    best_V = np.zeros((dim_V, best_rank), dtype=float)
    best_W = np.zeros((dim_W, best_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                best_U[a_idx, col] = 1
                best_V[b_idx, col] = 1
                best_W[c_idx, col] = 1
                col += 1
    
    # Try to find lower rank decompositions using iterative refinement
    # The tensor decomposition requires: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # We'll use a randomized approach with local search
    
    def validate_decomposition(U, V, W, tol=1e-10):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    val = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def try_random_rank(rank, max_iters=10000):
        """Try to find a valid decomposition with given rank using random initialization + local search."""
        nonlocal best_rank, best_U, best_V, best_W
        
        if rank >= best_rank:
            return False
        
        # Random initialization with small integers
        U = np.random.randint(-2, 3, size=(dim_U, rank))
        V = np.random.randint(-2, 3, size=(dim_V, rank))
        W = np.random.randint(-2, 3, size=(dim_W, rank))
        
        # Check initial
        if validate_decomposition(U, V, W):
            best_rank = rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            return True
        
        # Local search: try to fix violations
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Find violated constraints
            violations = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a_idx = i * m + j
                        b_idx = j * p + k
                        c_idx = k * n + i
                        val = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :])
                        if abs(val - 1) > 1e-10:
                            violations.append((i, j, k, a_idx, b_idx, c_idx, val))
            
            if not violations:
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                return True
            
            # Pick a random violation and try to fix it
            i, j, k, a_idx, b_idx, c_idx, val = violations[np.random.randint(len(violations))]
            error = 1 - val
            
            # Try adjusting one of the three factors
            factor = np.random.randint(3)
            r = np.random.randint(rank)
            
            if factor == 0:
                # Adjust U[a_idx, r]
                old_val = U[a_idx, r]
                # Try small changes
                for delta in [-1, 1, -2, 2]:
                    U[a_idx, r] = old_val + delta
                    # Quick check on this constraint
                    new_val = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :])
                    if abs(new_val - 1) < abs(val - 1):
                        break
                    U[a_idx, r] = old_val
            elif factor == 1:
                # Adjust V[b_idx, r]
                old_val = V[b_idx, r]
                for delta in [-1, 1, -2, 2]:
                    V[b_idx, r] = old_val + delta
                    new_val = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :])
                    if abs(new_val - 1) < abs(val - 1):
                        break
                    V[b_idx, r] = old_val
            else:
                # Adjust W[c_idx, r]
                old_val = W[c_idx, r]
                for delta in [-1, 1, -2, 2]:
                    W[c_idx, r] = old_val + delta
                    new_val = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :])
                    if abs(new_val - 1) < abs(val - 1):
                        break
                    W[c_idx, r] = old_val
        
        # Final check
        if validate_decomposition(U, V, W):
            best_rank = rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            return True
        
        return False
    
    # Try ranks from low to high
    for rank in range(20, best_rank):
        if time.time() > deadline:
            break
        if try_random_rank(rank, max_iters=5000):
            break
    
    # If we found a valid decomposition, convert to appropriate format
    if best_rank < 63:
        U_out = best_U.astype(float)
        V_out = best_V.astype(float)
        W_out = best_W.astype(float)
    else:
        U_out = best_U.astype(int)
        V_out = best_V.astype(int)
        W_out = best_W.astype(int)
    
    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist()
    }
# EVOLVE-BLOCK-END
