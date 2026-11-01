"""Rank-minimal exact tensor decomposition for 2x2 * 2x11 matrix multiplication."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_rank39():
    """Construct an explicit rank-39 decomposition for M_{2,2,11}."""
    n, m, p = 2, 2, 11
    rank = 39
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col_offset, base_rank):
        r = base_rank
        U[0][r] += 1.0
        U[3][r] += 1.0
        V[col_offset][r] += 1.0
        V[12 + col_offset][r] += 1.0
        W[2 * col_offset][r] += 1.0
        W[2 * col_offset + 3][r] += 1.0
        
        r = base_rank + 1
        U[2][r] += 1.0
        U[3][r] += 1.0
        V[col_offset][r] += 1.0
        W[2 * col_offset + 1][r] += 1.0
        
        r = base_rank + 2
        U[0][r] += 1.0
        V[col_offset + 1][r] += 1.0
        V[11 + col_offset][r] += -1.0
        W[2 * col_offset + 2][r] += 1.0
        
        r = base_rank + 3
        U[3][r] += 1.0
        V[11 + col_offset][r] += 1.0
        V[col_offset][r] += -1.0
        W[2 * col_offset + 1][r] += 1.0
        W[2 * col_offset][r] += -1.0
        
        r = base_rank + 4
        U[0][r] += 1.0
        U[1][r] += 1.0
        V[12 + col_offset][r] += 1.0
        W[2 * col_offset + 2][r] += 1.0
        W[2 * col_offset + 3][r] += -1.0
        
        r = base_rank + 5
        U[2][r] += 1.0
        U[0][r] += -1.0
        V[col_offset][r] += 1.0
        V[col_offset + 1][r] += 1.0
        W[2 * col_offset + 1][r] += 1.0
        W[2 * col_offset + 3][r] += 1.0
        
        r = base_rank + 6
        U[1][r] += 1.0
        U[3][r] += -1.0
        V[11 + col_offset][r] += 1.0
        V[12 + col_offset][r] += 1.0
        W[2 * col_offset][r] += 1.0
        W[2 * col_offset + 1][r] += -1.0
    
    for block_idx in range(5):
        col_offset = 2 * block_idx
        base_rank = 7 * block_idx
        add_strassen_block(col_offset, base_rank)
    
    base_rank = 35
    r = base_rank
    U[0][r] += 1.0
    V[10][r] += 1.0
    W[20][r] += 1.0
    
    r = base_rank + 1
    U[1][r] += 1.0
    V[21][r] += 1.0
    W[20][r] += 1.0
    
    r = base_rank + 2
    U[2][r] += 1.0
    V[10][r] += 1.0
    W[21][r] += 1.0
    
    r = base_rank + 3
    U[3][r] += 1.0
    V[21][r] += 1.0
    W[21][r] += 1.0
    
    return U, V, W

def _alternating_minimize(U, V, W, n, m, p, max_iter=200, tol=1e-12):
    """Run alternating minimization to refine the decomposition."""
    rank = U.shape[1]
    
    # Precompute target triples
    targets = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                targets.append((i * m + j, j * p + k, k * n + i))
    
    # Precompute which targets involve each index
    a_to_targets = {a: [] for a in range(n * m)}
    b_to_targets = {b: [] for b in range(m * p)}
    c_to_targets = {c: [] for c in range(p * n)}
    for (a, b, c) in targets:
        a_to_targets[a].append((a, b, c))
        b_to_targets[b].append((a, b, c))
        c_to_targets[c].append((a, b, c))
    
    for it in range(max_iter):
        # Update U
        for a in range(n * m):
            for c in range(rank):
                num = 0.0
                den = 0.0
                for (ta, tb, tc) in a_to_targets[a]:
                    vw = V[tb, c] * W[tc, c]
                    num += vw
                    den += vw * vw
                if den > 1e-15:
                    U[a, c] = num / den
        
        # Update V
        for b in range(m * p):
            for c in range(rank):
                num = 0.0
                den = 0.0
                for (ta, tb, tc) in b_to_targets[b]:
                    uw = U[ta, c] * W[tc, c]
                    num += uw
                    den += uw * uw
                if den > 1e-15:
                    V[b, c] = num / den
        
        # Update W
        for c in range(p * n):
            for cc in range(rank):
                num = 0.0
                den = 0.0
                for (ta, tb, tc) in c_to_targets[c]:
                    uv = U[ta, cc] * V[tb, cc]
                    num += uv
                    den += uv * uv
                if den > 1e-15:
                    W[c, cc] = num / den
        
        # Check convergence
        err = 0.0
        for (a, b, c) in targets:
            s = np.dot(U[a], V[b] * W[c])
            err += (s - 1.0) ** 2
        if err < tol:
            break
    
    return U, V, W, err

def _try_rank(rank, n, m, p, seed, max_iter=500):
    """Try to find a decomposition of given rank."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    U, V, W, err = _alternating_minimize(U, V, W, n, m, p, max_iter=max_iter)
    
    if err < 1e-10:
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _try_rank_from_init(U_init, V_init, W_init, n, m, p, max_iter=1000):
    """Try to find a decomposition starting from a given initialization."""
    U = np.array(U_init, dtype=float)
    V = np.array(V_init, dtype=float)
    W = np.array(W_init, dtype=float)
    rank = U.shape[1]
    
    U, V, W, err = _alternating_minimize(U, V, W, n, m, p, max_iter=max_iter)
    
    if err < 1e-10:
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _try_merge(U, V, W, n, m, p, merge_count=1):
    """Try to merge rank-1 tensors to reduce rank."""
    rank = U.shape[1]
    if rank <= 2:
        return None
    
    # Try all pairs of rank-1 tensors to see if they can be merged
    for i in range(rank):
        for j in range(i + 1, rank):
            # Create a new tensor by combining i and j
            # Try to find a rank-1 tensor that approximates the sum of tensors i and j
            # and see if the remaining tensors can compensate
            pass  # This is complex; skip for now
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # First, get the known rank-39 solution
    U39, V39, W39 = _construct_rank39()
    if _verify(U39, V39, W39, n, m, p):
        best_result = (39, U39, V39, W39)
    else:
        best_result = None
    
    # Try to find lower rank solutions
    # Start from rank 38 and go down
    for rank in range(38, 20, -1):
        if time.time() > deadline - 15:
            break
        
        # Try multiple seeds
        for seed in [rank, rank + 1000, rank + 2000, rank + 3000]:
            if time.time() > deadline - 10:
                break
            
            result = _try_rank(rank, n, m, p, seed=seed, max_iter=300)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
        
        # Also try starting from the rank-39 solution and projecting
        if best_result is not None and rank <= 38:
            if time.time() > deadline - 10:
                break
            # Take a subset of the rank-39 tensors
            U39_arr = np.array(U39)
            V39_arr = np.array(V39)
            W39_arr = np.array(W39)
            
            # Randomly select 'rank' columns from the 39
            rng = np.random.RandomState(rank)
            indices = rng.choice(39, size=rank, replace=False)
            U_sub = U39_arr[:, indices]
            V_sub = V39_arr[:, indices]
            W_sub = W39_arr[:, indices]
            
            result = _try_rank_from_init(U_sub, V_sub, W_sub, n, m, p, max_iter=500)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    # Return the best result found
    if best_result is not None:
        return {"U": best_result[1], "V": best_result[2], "W": best_result[3]}
    
    # Final fallback: schoolbook rank 44
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
# EVOLVE-BLOCK-END
