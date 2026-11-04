import time
import numpy as np

# Dimensions: n x m by m x p
n, m, p = 2, 3, 9

def construct():
    """
    Returns a tensor decomposition for multiplying a 2x3 matrix by a 3x9 matrix.
    Attempts to find a rank-minimal exact decomposition.
    Falls back to schoolbook rank 54 if no better exact decomposition is found.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    # Schoolbook decomposition (rank 54)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=float)
    V_sb = np.zeros((m * p, rank_sb), dtype=float)
    W_sb = np.zeros((p * n, rank_sb), dtype=float)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def to_lists(mat):
        return mat.tolist()

    def verify_decomposition(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Check T[i, j, k] = 1
                    val = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
                    # Check T[i, j, k'] = 0 for k' != k
                    for k2 in range(p):
                        if k2 == k: continue
                        val = np.sum(U_r[i, j, :] * V_r[j, k2, :] * W_r[k2, i, :])
                        if abs(val) > 1e-9:
                            return False
                    # Check T[i', j, k] = 0 for i' != i
                    for i2 in range(n):
                        if i2 == i: continue
                        val = np.sum(U_r[i2, j, :] * V_r[j, k, :] * W_r[k, i2, :])
                        if abs(val) > 1e-9:
                            return False
                    # Check T[i, j', k] = 0 for j' != j
                    for j2 in range(m):
                        if j2 == j: continue
                        val = np.sum(U_r[i, j2, :] * V_r[j2, k, :] * W_r[k, i, :])
                        if abs(val) > 1e-9:
                            return False
        return True

    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb

    # Try to find a better decomposition for M_2,3,3 (rank 18 schoolbook)
    # If we find rank R for 2x3x3, total rank is 3*R
    # We try ranks 15, 16, 17 for the block
    for block_rank in [15, 16, 17]:
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            r = block_rank
            # Random initialization
            U_try = np.random.randn(n * m, r) * 0.5
            V_try = np.random.randn(m * p, r) * 0.5
            W_try = np.random.randn(p * n, r) * 0.5
            
            # Round to half-integers
            U_try = np.round(U_try * 2) / 2
            V_try = np.round(V_try * 2) / 2
            W_try = np.round(W_try * 2) / 2
            
            # Verify for 2x3x3 block (k=0..2)
            # We only need to check the first block if we construct the full tensor by repeating
            # But let's just verify the full 2x3x9 tensor constructed from 3 blocks
            U_full = np.zeros((n * m, 3 * r))
            V_full = np.zeros((m * p, 3 * r))
            W_full = np.zeros((p * n, 3 * r))
            
            for b in range(3):
                U_full[:, b*r:(b+1)*r] = U_try
                # V and W need to be shifted for the block
                V_block = np.zeros((m * p, r))
                W_block = np.zeros((p * n, r))
                for j in range(m):
                    for k in range(3):
                        V_block[j * p + (b * 3 + k), :] = V_try[j * 3 + k, :]
                        W_block[(b * 3 + k) * n + 0, :] = W_try[k * n + 0, :]
                        W_block[(b * 3 + k) * n + 1, :] = W_try[k * n + 1, :]
                V_full[:, b*r:(b+1)*r] = V_block
                W_full[:, b*r:(b+1)*r] = W_block
            
            if verify_decomposition(U_full, V_full, W_full):
                best_U, best_V, best_W = U_full, V_full, W_full
                best_rank = 3 * r
                found = True
                break
        
        if found:
            break

    return {
        "U": to_lists(best_U),
        "V": to_lists(best_V),
        "W": to_lists(best_W)
    }
