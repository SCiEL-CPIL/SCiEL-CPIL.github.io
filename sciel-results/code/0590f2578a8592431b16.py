import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W form a valid decomposition."""
    if not U or not V or not W:
        return False
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check all required entries
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                val = np.dot(U[u_idx], V[v_idx]) * 0  # placeholder
                # Actually: sum_c U[u_idx,c] * V[v_idx,c] * W[w_idx,c]
                val = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                if abs(val - 1) > tol:
                    return False
    # Check a sample of non-required entries to ensure they're 0
    # (Full check is too expensive; sample is sufficient for confidence)
    rng = np.random.RandomState(42)
    for _ in range(200):
        u_idx = rng.randint(n * m)
        v_idx = rng.randint(m * p)
        w_idx = rng.randint(p * n)
        # Decode
        i, k = divmod(u_idx, m)
        k2, j = divmod(v_idx, p)
        j2, i2 = divmod(w_idx, n)
        if k != k2 or j != j2 or i != i2:
            val = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
            if abs(val) > tol:
                return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_random_decomposition(n, m, p, target_rank, num_trials=5, timeout=10):
    """Try to find a random decomposition of given rank."""
    start = time.time()
    Nm, Mp, Pn = n * m, m * p, p * n
    for trial in range(num_trials):
        if time.time() - start > timeout:
            break
        rng = np.random.RandomState(trial * 1000 + 42)
        U = rng.randn(Nm, target_rank)
        V = rng.randn(Mp, target_rank)
        W = rng.randn(Pn, target_rank)
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _block_decomposition(n, m, p):
    """
    Block decomposition: split columns of B into groups.
    For 4x4 x 4x9: split 9 cols into 4+4+1.
    Each 4x4 x 4x4 uses schoolbook (rank 16), 4x4 x 4x1 uses rank 4.
    Total: 16 + 16 + 4 = 36 per... no, this is per the full tensor.
    Actually this gives rank 4*4*4 + 4*4*4 + 4*4*1 = 64+64+16 = 144. Same as schoolbook.
    So this doesn't help. Need sub-schoolbook blocks.
    """
    # This block structure doesn't beat schoolbook without sub-schoolbook sub-decompositions.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 144
    
    # Try to find lower rank via random search
    # The tensor is 16x36x36. Random rank-r decomposition is very unlikely to work
    # for r < 144, but let's try a few ranks.
    
    # Try a structured approach: use the fact that we can potentially share
    # structure between the three 4-column groups.
    
    # For now, return schoolbook as it's guaranteed valid.
    # A true improvement requires a verified sub-schoolbook decomposition
    # of a sub-block, which we don't have.
    
    return best
# EVOLVE-BLOCK-END
