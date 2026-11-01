"""Matrix multiplication 2x5x5 - rank minimization via structured search."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition of rank n*m*p
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    U_sb_np = np.array(U_sb, dtype=float)
    V_sb_np = np.array(V_sb, dtype=float)
    W_sb_np = np.array(W_sb, dtype=float)
    
    best_rank = rank_schoolbook
    best_U = U_sb_np
    best_V = V_sb_np
    best_W = W_sb_np
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], V[b, :]) * W[c, :]
                    # Wait, that's not right. The tensor entry is sum over c of U[a,c]*V[b,c]*W[c_row,c]
                    # Let me recompute properly
                    val = 0.0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_fast(U, V, W, rank):
        """Faster verification using matrix operations."""
        # For each (i,j,k), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        # Let's check all entries efficiently
        for i in range(n):
            for k in range(p):
                # For fixed i,k, we need: for all j, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
                w_row = W[k * n + i, :]
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    val = np.dot(U[a, :] * V[b, :], w_row)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find lower rank decompositions using structured approaches
    
    # Approach 1: Try ranks from 16 up to schoolbook-1
    # Use random initialization with small integer/half-integer values and local search
    
    def try_rank(r, max_attempts=50, seed_offset=0):
        """Try to find a rank-r decomposition."""
        if r >= best_rank:
            return None
        
        rng = np.random.RandomState(42 + seed_offset)
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Initialize with random values from {-1, -0.5, 0, 0.5, 1}
            U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, r))
            V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, r))
            W = rng.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, r))
            
            # Local search: try to improve by adjusting individual entries
            for it in range(200):
                if time.time() > deadline:
                    return None
                
                # Pick a random factor and entry to adjust
                factor_idx = rng.randint(3)
                if factor_idx == 0:
                    row = rng.randint(n * m)
                    col = rng.randint(r)
                    old_val = U[row, col]
                    # Try nearby values
                    for new_val in [-1, -0.5, 0, 0.5, 1]:
                        if new_val == old_val:
                            continue
                        U[row, col] = new_val
                        if verify_fast(U, V, W, r):
                            return (U, V, W, r)
                    U[row, col] = old_val
                elif factor_idx == 1:
                    row = rng.randint(m * p)
                    col = rng.randint(r)
                    old_val = V[row, col]
                    for new_val in [-1, -0.5, 0, 0.5, 1]:
                        if new_val == old_val:
                            continue
                        V[row, col] = new_val
                        if verify_fast(U, V, W, r):
                            return (U, V, W, r)
                    V[row, col] = old_val
                else:
                    row = rng.randint(p * n)
                    col = rng.randint(r)
                    old_val = W[row, col]
                    for new_val in [-1, -0.5, 0, 0.5, 1]:
                        if new_val == old_val:
                            continue
                        W[row, col] = new_val
                        if verify_fast(U, V, W, r):
                            return (U, V, W, r)
                    W[row, col] = old_val
        
        return None
    
    # Try various ranks in increasing order
    for r in range(16, rank_schoolbook):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_attempts=30, seed_offset=r)
        if result is not None:
            U, V, W, rank = result
            if verify_fast(U, V, W, rank):
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break  # Found the minimum, stop
    
    # If no better rank found, return schoolbook
    if best_rank == rank_schoolbook:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    else:
        # Convert numpy arrays to lists for the return interface
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
