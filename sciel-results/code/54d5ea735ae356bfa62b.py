"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0  # Safety margin before 150s limit

    # Strassen patterns for 2x2x2
    # Each pattern: U entries (i,j,val), V entries (j,k,val), W entries (k,i,val)
    strassen_patterns = [
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]
        },
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)]
        },
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(1,0,1), (1,1,1)]
        },
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (0,1,1)]
        },
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (1,0,1)]
        },
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)]
        },
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)]
        }
    ]

    def get_splits(d):
        """Get all possible splits of d into parts of 1 and 2."""
        splits = []
        def _gen(remaining, current):
            if remaining == 0:
                splits.append(tuple(current))
                return
            if remaining >= 1:
                current.append(1)
                _gen(remaining - 1, current)
                current.pop()
            if remaining >= 2:
                current.append(2)
                _gen(remaining - 2, current)
                current.pop()
        _gen(d, [])
        return splits

    def compute_rank(n_splits, m_splits, p_splits):
        """Compute the rank for a given set of splits."""
        rank = 0
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    n_b = n_splits[i_b]
                    m_b = m_splits[j_b]
                    p_b = p_splits[k_b]
                    if n_b == 2 and m_b == 2 and p_b == 2:
                        rank += 7
                    else:
                        rank += n_b * m_b * p_b
        return rank

    def build_decomposition(n_splits, m_splits, p_splits):
        """Build the full decomposition for given splits using entry lists."""
        n_offsets = [0]
        for s in n_splits[:-1]:
            n_offsets.append(n_offsets[-1] + s)
        m_offsets = [0]
        for s in m_splits[:-1]:
            m_offsets.append(m_offsets[-1] + s)
        p_offsets = [0]
        for s in p_splits[:-1]:
            p_offsets.append(p_offsets[-1] + s)

        rank = compute_rank(n_splits, m_splits, p_splits)

        U_entries = []
        V_entries = []
        W_entries = []
        rank_idx = 0

        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    n_b = n_splits[i_b]
                    m_b = m_splits[j_b]
                    p_b = p_splits[k_b]

                    i_off = n_offsets[i_b]
                    j_off = m_offsets[j_b]
                    k_off = p_offsets[k_b]

                    if n_b == 2 and m_b == 2 and p_b == 2:
                        for pattern in strassen_patterns:
                            col = rank_idx
                            for li, lj, val in pattern['U']:
                                gi = i_off + li
                                gj = j_off + lj
                                U_entries.append((gi * m + gj, col, val))
                            for lj, lk, val in pattern['V']:
                                gj = j_off + lj
                                gk = k_off + lk
                                V_entries.append((gj * p + gk, col, val))
                            for lk, li, val in pattern['W']:
                                gk = k_off + lk
                                gi = i_off + li
                                W_entries.append((gk * n + gi, col, val))
                            rank_idx += 1
                    else:
                        for li in range(n_b):
                            for lj in range(m_b):
                                for lk in range(p_b):
                                    col = rank_idx
                                    gi = i_off + li
                                    gj = j_off + lj
                                    gk = k_off + lk
                                    U_entries.append((gi * m + gj, col, 1))
                                    V_entries.append((gj * p + gk, col, 1))
                                    W_entries.append((gk * n + gi, col, 1))
                                    rank_idx += 1

        # Build dense matrices
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        for row, col, val in U_entries:
            U[row][col] = val
        for row, col, val in V_entries:
            V[row][col] = val
        for row, col, val in W_entries:
            W[row][col] = val

        return {"U": U, "V": V, "W": W}, rank

    def compute_tensor_fast(U_np, V_np, W_np, n, m, p):
        """Compute the full multiplication tensor T[i,j,k] using vectorized ops."""
        rank = U_np.shape[1]
        T = np.zeros((n, m, p), dtype=float)
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                u_row = U_np[i*m+j, :]  # shape (rank,)
                for k in range(p):
                    T[i, j, k] = np.dot(u_row, V_np[j*p+k, :] * W_np[k*n+i, :])
        return T

    def try_reduce_rank(U, V, W, rank, time_budget=10.0):
        """
        Polish pass: attempt to reduce rank by finding linearly dependent
        columns in the tensor factorization.
        """
        start_time = time.time()
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # Compute the reference tensor
        T_ref = compute_tensor_fast(U_np, V_np, W_np, n, m, p)

        current_rank = rank
        best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()

        # Try to eliminate columns one by one
        for trial_col in range(current_rank):
            if time.time() - start_time > time_budget:
                break

            u_col = U_np[:, trial_col]
            v_col = V_np[:, trial_col]
            w_col = W_np[:, trial_col]

            # If any factor has all zeros in this column, the column contributes nothing
            if np.all(u_col == 0) or np.all(v_col == 0) or np.all(w_col == 0):
                mask = [c for c in range(current_rank) if c != trial_col]
                best_U = U_np[:, mask]
                best_V = V_np[:, mask]
                best_W = W_np[:, mask]
                current_rank = len(mask)
                continue

            # Try removing and verify
            if current_rank > 1:
                mask = [c for c in range(current_rank) if c != trial_col]
                U_test = U_np[:, mask]
                V_test = V_np[:, mask]
                W_test = W_np[:, mask]

                if time.time() - start_time < time_budget - 0.5:
                    T_test = compute_tensor_fast(U_test, V_test, W_test, n, m, p)
                    if np.allclose(T_test, T_ref, atol=1e-10):
                        best_U = U_test
                        best_V = V_test
                        best_W = W_test
                        current_rank = len(mask)

        if current_rank < rank:
            U_out = best_U.tolist()
            V_out = best_V.tolist()
            W_out = best_W.tolist()
            U_out = [[round(x, 10) if abs(x) > 1e-10 else 0 for x in row] for row in U_out]
            V_out = [[round(x, 10) if abs(x) > 1e-10 else 0 for x in row] for row in V_out]
            W_out = [[round(x, 10) if abs(x) > 1e-10 else 0 for x in row] for row in W_out]
            return {"U": U_out, "V": V_out, "W": W_out}, current_rank

        return None

    def try_linear_combination_reduction(U, V, W, rank, time_budget=20.0):
        """
        Advanced polish: try to express one rank-1 term as a linear combination
        of others by solving a linear system.
        
        For column r, we want to find alpha_s such that:
        U[:,r] ⊗ V[:,r] ⊗ W[:,r] = sum_s alpha_s * (U[:,s] ⊗ V[:,s] ⊗ W[:,s])
        
        This is equivalent to: for all (i,j,k):
        U[i,r]*V[j,r]*W[k,r] = sum_s alpha_s * U[i,s]*V[j,s]*W[k,s]
        
        We can set up a linear system by picking enough (i,j,k) points.
        """
        start_time = time.time()
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        T_ref = compute_tensor_fast(U_np, V_np, W_np, n, m, p)

        current_rank = rank
        best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()

        # For each column, try to see if it can be eliminated
        for trial_col in range(current_rank):
            if time.time() - start_time > time_budget:
                break

            u_col = U_np[:, trial_col]
            v_col = V_np[:, trial_col]
            w_col = W_np[:, trial_col]

            # Skip if already zero
            if np.all(u_col == 0) or np.all(v_col == 0) or np.all(w_col == 0):
                mask = [c for c in range(current_rank) if c != trial_col]
                best_U = U_np[:, mask]
                best_V = V_np[:, mask]
                best_W = W_np[:, mask]
                current_rank = len(mask)
                continue

            # Build linear system: find alpha such that
            # sum_s alpha_s * (U[:,s] ⊗ V[:,s] ⊗ W[:,s]) = U[:,r] ⊗ V[:,r] ⊗ W[:,r]
            # where s ranges over all columns except r

            other_cols = [c for c in range(current_rank) if c != trial_col]
            if len(other_cols) < 1:
                continue

            # Target tensor
            target = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        target[i, j, k] = U_np[i*m+j, trial_col] * V_np[j*p+k, trial_col] * W_np[k*n+i, trial_col]

            # Build matrix A where A[:,s] is the flattened rank-1 tensor for column s
            # A has shape (n*m*p, len(other_cols))
            A = np.zeros((n * m * p, len(other_cols)))
            for idx, s in enumerate(other_cols):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            A[i * m * p + j * p + k, idx] = (
                                U_np[i*m+j, s] * V_np[j*p+k, s] * W_np[k*n+i, s]
                            )

            target_flat = target.flatten()

            # Solve least squares
            try:
                alpha, residuals, rank_A, sv = np.linalg.lstsq(A, target_flat, rcond=1e-10)
            except Exception:
                continue

            # Check if the solution is good
            residual = np.linalg.norm(A @ alpha - target_flat)
            if residual < 1e-8:
                # We can remove column trial_col and add alpha * other_cols
                new_U = best_U[:, [c for c in range(current_rank) if c != trial_col]].copy()
                new_V = best_V[:, [c for c in range(current_rank) if c != trial_col]].copy()
                new_W = best_W[:, [c for c in range(current_rank) if c != trial_col]].copy()

                for idx, s in enumerate(other_cols):
                    new_U[:, idx] += alpha[idx] * U_np[:, s]
                    new_V[:, idx] += alpha[idx] * V_np[:, s]
                    new_W[:, idx] += alpha[idx] * W_np[:, s]

                # Verify
                if time.time() - start_time < time_budget - 1.0:
                    T_test = compute_tensor_fast(new_U, new_V, new_W, n, m, p)
                    if np.allclose(T_test, T_ref, atol=1e-8):
                        best_U = new_U
                        best_V = new_V
                        best_W = new_W
                        current_rank -= 1

        if current_rank < rank:
            U_out = best_U.tolist()
            V_out = best_V.tolist()
            W_out = best_W.tolist()
            U_out = [[round(x, 10) if abs(x) > 1e-10 else 0 for x in row] for row in U_out]
            V_out = [[round(x, 10) if abs(x) > 1e-10 else 0 for x in row] for row in V_out]
            W_out = [[round(x, 10) if abs(x) > 1e-10 else 0 for x in row] for row in W_out]
            return {"U": U_out, "V": V_out, "W": W_out}, current_rank

        return None

    # Generate all possible split patterns
    n_split_options = get_splits(n)
    m_split_options = get_splits(m)
    p_split_options = get_splits(p)

    # Score all candidates
    all_candidates = []
    for ns in n_split_options:
        for ms in m_split_options:
            for ps in p_split_options:
                r = compute_rank(ns, ms, ps)
                if r < n * m * p:
                    all_candidates.append((r, ns, ms, ps))

    # Sort by rank
    all_candidates.sort(key=lambda x: x[0])

    # Try candidates in order
    best_result = None
    best_rank = float('inf')

    max_candidates_to_try = min(len(all_candidates), 15)
    for idx in range(max_candidates_to_try):
        if time.time() > deadline - 30:
            break
        r, ns, ms, ps = all_candidates[idx]
        if r >= best_rank:
            break

        try:
            result, actual_rank = build_decomposition(ns, ms, ps)
            if actual_rank < best_rank:
                best_rank = actual_rank
                best_result = result
        except Exception:
            continue

    # Polish pass 1: simple column removal
    if best_result is not None and time.time() < deadline - 20:
        time_remaining = deadline - time.time() - 15
        polished = try_reduce_rank(
            best_result['U'],
            best_result['V'],
            best_result['W'],
            best_rank,
            time_budget=min(10.0, max(1.0, time_remaining / 2))
        )
        if polished is not None:
            best_result = polished
            best_rank = polished and compute_rank_from_result(best_result) if hasattr(compute_rank_from_result, '__call__') else best_rank
            # Recompute rank from the polished result
            U_test = np.array(best_result['U'])
            best_rank = U_test.shape[1]

    # Polish pass 2: linear combination reduction
    if best_result is not None and time.time() < deadline - 10:
        time_remaining = deadline - time.time() - 5
        polished2 = try_linear_combination_reduction(
            best_result['U'],
            best_result['V'],
            best_result['W'],
            best_rank,
            time_budget=min(20.0, max(1.0, time_remaining))
        )
        if polished2 is not None:
            best_result = polished2
            U_test = np.array(best_result['U'])
            best_rank = U_test.shape[1]

    if best_result is None:
        # Fallback: schoolbook
        best_result, _ = build_decomposition([n], [m], [p])

    return best_result
# EVOLVE-BLOCK-END
