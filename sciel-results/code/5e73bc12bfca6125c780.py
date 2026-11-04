import numpy as np
import time
import random

DIMENSIONS = [4, 4, 6]


def _build_rank77():
    """Build the rank-77 Strassen-based decomposition."""
    n, m, p = 4, 4, 6
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))

    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    col = 0
    for s in range(7):
        for si in range(7):
            for a in range(4):
                if alpha[s].get(a, 0) == 0:
                    continue
                if alpha[si].get(a, 0) == 0:
                    continue
                x_row, x_col = a // 2, a % 2
                for r in range(4):
                    if alpha[s].get(r, 0) == 0:
                        continue
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                    U[u_idx, col] += alpha[si][a] * alpha[s][r]

            for b in range(4):
                if alpha[s].get(b, 0) == 0:
                    continue
                if alpha[si].get(b, 0) == 0:
                    continue
                y_row, y_inner_col = b // 2, b % 2
                for c in range(4):
                    if beta[s].get(c, 0) == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                    V[v_idx, col] += alpha[si][b] * beta[s][c]

            for a2, b2, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b2 * 3 + zb) * 4 + (a2 * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            col += 1

        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s].get(r, 0) == 0:
                        continue
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                    U[u_idx, col] += alpha[s][r]

                for c in range(4):
                    if beta[s].get(c, 0) == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                    V[v_idx, col] += beta[s][c]

                for a2, b2, gamma in recon[s]:
                    w_idx = (b2 * 3 + 2) * 4 + (a2 * 2 + u)
                    W[w_idx, col] += gamma
                col += 1

    return U, V, W


def _verify(U, V, W, seed=42):
    """Verify decomposition correctness."""
    np.random.seed(seed)
    A = np.random.randn(4, 4)
    B = np.random.randn(4, 6)
    A_flat = A.flatten()
    B_flat = B.flatten()
    C = np.zeros((24,))
    for r in range(U.shape[1]):
        ua = np.dot(U[:, r], A_flat)
        vb = np.dot(V[:, r], B_flat)
        if ua != 0 and vb != 0:
            C += (ua * vb) * W[:, r]
    C = C.reshape(6, 4).T
    return np.allclose(C, A @ B, atol=1e-9)


def _greedy_column_elimination(U, V, W, max_time, rng):
    """
    Structural change: instead of random dense transforms, use greedy column elimination.
    For each column, check if removing it and adjusting others preserves the tensor.
    This is a different optimization family: combinatorial column pruning.
    """
    R = U.shape[1]
    deadline = time.time() + max_time
    best_rank = R
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Precompute test matrices for verification
    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 6)
    A_flat = A_test.flatten()
    B_flat = B_test.flatten()
    
    def _compute_C(U_mat, V_mat, W_mat, cols):
        C = np.zeros(24)
        for c in cols:
            ua = np.dot(U_mat[:, c], A_flat)
            vb = np.dot(V_mat[:, c], B_flat)
            if ua != 0 and vb != 0:
                C += (ua * vb) * W_mat[:, c]
        return C
    
    def _is_valid(U_mat, V_mat, W_mat, cols):
        C = _compute_C(U_mat, V_mat, W_mat, cols)
        C = C.reshape(6, 4).T
        return np.allclose(C, A_test @ B_test, atol=1e-8)
    
    current_cols = list(range(R))
    
    for attempt in range(5000):
        if time.time() > deadline:
            break
        
        # Try to remove a random column
        if len(current_cols) <= 1:
            break
        c_idx = rng.randint(0, len(current_cols) - 1)
        col_to_remove = current_cols[c_idx]
        new_cols = [c for i, c in enumerate(current_cols) if i != c_idx]
        
        if _is_valid(U, V, W, new_cols):
            current_cols = new_cols
            if len(current_cols) < best_rank:
                best_rank = len(current_cols)
                best_U = U[:, current_cols].copy()
                best_V = V[:, current_cols].copy()
                best_W = W[:, current_cols].copy()
    
    return best_U, best_V, best_W, best_rank


def _structured_transform_search(U, V, W, max_time, rng):
    """
    Second structural change: use structured (block-diagonal) transforms
    that respect the Kronecker product structure of the Strassen decomposition.
    """
    R = U.shape[1]
    deadline = time.time() + max_time
    best_rank = R
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for attempt in range(300):
        if time.time() > deadline:
            break
        
        # Create a block-diagonal transform with small random blocks
        T = np.eye(R)
        # Pick 2-4 pairs of columns to mix
        n_mixes = rng.randint(2, 4)
        used = set()
        for _ in range(n_mixes):
            i = rng.randint(0, R - 1)
            j = rng.randint(0, R - 1)
            if i != j and i not in used and j not in used:
                angle = rng.choice([1, -1])
                T[i, j] = angle
                T[j, i] = angle
                used.add(i)
                used.add(j)
        
        U_new = U @ T
        V_new = V @ T
        try:
            T_inv = np.linalg.inv(T)
        except np.linalg.LinAlgError:
            continue
        W_new = W @ T_inv
        
        zero_cols = set()
        for c in range(R):
            if (np.all(np.abs(U_new[:, c]) < 1e-12) or
                np.all(np.abs(V_new[:, c]) < 1e-12) or
                np.all(np.abs(W_new[:, c]) < 1e-12)):
                zero_cols.add(c)
        
        if zero_cols:
            keep = [c for c in range(R) if c not in zero_cols]
            new_rank = len(keep)
            if new_rank < best_rank:
                cand_U = U_new[:, keep]
                cand_V = V_new[:, keep]
                cand_W = W_new[:, keep]
                if _verify(cand_U, cand_V, cand_W, seed=attempt):
                    best_U, best_V, best_W = cand_U, cand_V, cand_W
                    best_rank = new_rank
    
    return best_U, best_V, best_W, best_rank


def _to_list(mat):
    """Convert numpy array to list with integer/half-integer rounding."""
    result = []
    for row in mat:
        new_row = []
        for x in row:
            rx = round(x)
            if abs(x - rx) < 1e-9:
                new_row.append(int(rx))
            else:
                hx = round(x * 2) / 2.0
                if abs(x - hx) < 1e-9:
                    new_row.append(hx)
                else:
                    new_row.append(float(x))
        result.append(new_row)
    return result


def _schoolbook():
    """Fallback schoolbook decomposition, rank 96."""
    R = 96
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    idx = 0
    for i in range(4):
        for k in range(4):
            for j in range(6):
                U[i * 4 + k, idx] = 1
                V[k * 6 + j, idx] = 1
                W[j * 4 + i, idx] = 1
                idx += 1
    return U, V, W


def construct():
    _t0 = time.time()
    deadline = _t0 + 140.0

    # Build baseline rank-77 decomposition
    U, V, W = _build_rank77()

    # Verify baseline
    if not _verify(U, V, W):
        U, V, W = _schoolbook()
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    # Phase 1: Greedy column elimination (different optimization family)
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0:
        rng1 = random.Random(12345)
        U, V, W, rank1 = _greedy_column_elimination(U, V, W, min(remaining - 5.0, 30.0), rng1)
        if not _verify(U, V, W):
            U, V, W = _build_rank77()
    
    # Phase 2: Structured transform search
    remaining = deadline - time.time() - 5.0
    if remaining > 5.0:
        rng2 = random.Random(54321)
        U, V, W, rank2 = _structured_transform_search(U, V, W, remaining, rng2)
        if not _verify(U, V, W):
            U, V, W = _build_rank77()

    # Final verification
    if not _verify(U, V, W):
        U, V, W = _schoolbook()

    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
