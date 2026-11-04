"""Matrix multiplication 4x4x4 tensor decomposition - rank 49 via Kronecker of Strassen."""
import numpy as np
import time

def construct():
    """
    Constructs an exact tensor decomposition for 4x4 matrix multiplication.
    Uses the Kronecker product of Strassen's rank-7 decomposition for 2x2,
    giving rank 49. Applies correct bit-interleaving permutation.
    Output follows the trace convention W[k*n+i, r].
    """
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s kill

    # Strassen 2x2 decomposition (rank 7)
    # U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Verify 2x2 Strassen correctness
    for i in range(2):
        for j in range(2):
            for k in range(2):
                expected = 1 if i == k else 0
                actual = sum(U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c] for c in range(7))
                if actual != expected:
                    # Fallback to naive rank 64
                    U = np.zeros((16, 64), dtype=int)
                    V = np.zeros((16, 64), dtype=int)
                    W = np.zeros((16, 64), dtype=int)
                    col = 0
                    for ii in range(4):
                        for jj in range(4):
                            for kk in range(4):
                                U[ii*4+jj, col] = 1
                                V[jj*4+kk, col] = 1
                                W[kk*4+ii, col] = 1
                                col += 1
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Kronecker product to get 4x4 decomposition (rank 49)
    U4_kron = np.kron(U2, U2)  # shape (16, 49)
    V4_kron = np.kron(V2, V2)  # shape (16, 49)
    W4_kron = np.kron(W2, W2)  # shape (16, 49)

    # Apply bit-interleaving permutation to reorder rows
    # For U: S = i*4+j, i=i1*2+i2, j=j1*2+j2 -> K = (i1*2+j1)*4 + (i2*2+j2)
    # For V: S = j*4+k, j=j1*2+j2, k=k1*2+k2 -> K = (j1*2+k1)*4 + (j2*2+k2)
    # For W: S = k*4+i, k=k1*2+k2, i=i1*2+i2 -> K = (k1*2+i1)*4 + (k2*2+i2)
    
    def compute_perm_u(S):
        i = S // 4
        j = S % 4
        i1, i2 = i // 2, i % 2
        j1, j2 = j // 2, j % 2
        return (i1 * 2 + j1) * 4 + (i2 * 2 + j2)

    def compute_perm_v(S):
        j = S // 4
        k = S % 4
        j1, j2 = j // 2, j % 2
        k1, k2 = k // 2, k % 2
        return (j1 * 2 + k1) * 4 + (j2 * 2 + k2)

    def compute_perm_w(S):
        k = S // 4
        i = S % 4
        k1, k2 = k // 2, k % 2
        i1, i2 = i // 2, i % 2
        return (k1 * 2 + i1) * 4 + (k2 * 2 + i2)

    perm_U = [compute_perm_u(S) for S in range(16)]
    perm_V = [compute_perm_v(S) for S in range(16)]
    perm_W = [compute_perm_w(S) for S in range(16)]

    U4 = U4_kron[perm_U, :]
    V4 = V4_kron[perm_V, :]
    W4 = W4_kron[perm_W, :]

    # Verify the 4x4 decomposition
    for i in range(4):
        for j in range(4):
            for k in range(4):
                expected = 1 if i == k else 0
                actual = sum(U4[i*4+j, c] * V4[j*4+k, c] * W4[k*4+i, c] for c in range(49))
                if actual != expected:
                    # Fallback to naive rank 64
                    U = np.zeros((16, 64), dtype=int)
                    V = np.zeros((16, 64), dtype=int)
                    W = np.zeros((16, 64), dtype=int)
                    col = 0
                    for ii in range(4):
                        for jj in range(4):
                            for kk in range(4):
                                U[ii*4+jj, col] = 1
                                V[jj*4+kk, col] = 1
                                W[kk*4+ii, col] = 1
                                col += 1
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Column elimination: remove linearly dependent rank-1 components
    def try_eliminate(U_m, V_m, W_m):
        """Try to eliminate linearly dependent columns."""
        n_cols = U_m.shape[1]
        if n_cols <= 1:
            return U_m, V_m, W_m, n_cols
        stacked = np.vstack([U_m, V_m, W_m]).astype(float)
        kept = list(range(n_cols))
        changed = True
        while changed:
            changed = False
            if len(kept) == 0:
                break
            cur_mat = stacked[:, kept]
            cur_rank = np.linalg.matrix_rank(cur_mat)
            for idx in range(len(kept)):
                if time.time() > DEADLINE - 5:
                    break
                trial_kept = kept[:idx] + kept[idx+1:]
                if len(trial_kept) == 0:
                    break
                trial_mat = stacked[:, trial_kept]
                if np.linalg.matrix_rank(trial_mat) == cur_rank:
                    kept = trial_kept
                    changed = True
                    break
            if time.time() > DEADLINE - 5:
                break
        if len(kept) < n_cols:
            return U_m[:, kept], V_m[:, kept], W_m[:, kept], len(kept)
        return U_m, V_m, W_m, n_cols

    U4, V4, W4, final_rank = try_eliminate(U4, V4, W4)

    # Final verification after elimination
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                expected = 1 if i == k else 0
                actual = sum(U4[i*4+j, c] * V4[j*4+k, c] * W4[k*4+i, c] for c in range(final_rank))
                if actual != expected:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fall back to pre-elimination rank-49
        U4 = U4_kron[perm_U, :]
        V4 = V4_kron[perm_V, :]
        W4 = W4_kron[perm_W, :]
        final_rank = 49

    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
