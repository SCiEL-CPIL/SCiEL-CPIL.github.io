"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin before 150s limit

    # Strassen patterns for 2x2x2
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

    def soft_min_score(n_splits, m_splits, p_splits, temperature=0.5):
        """
        Soft-min surrogate: smoothly scores split configurations.
        Lower score = better (more Strassen blocks, fewer schoolbook).
        Uses log-sum-exp style smoothing over the block types.
        """
        raw_scores = []
        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    n_b = n_splits[i_b]
                    m_b = m_splits[j_b]
                    p_b = p_splits[k_b]
                    if n_b == 2 and m_b == 2 and p_b == 2:
                        raw_scores.append(7.0)
                    else:
                        raw_scores.append(float(n_b * m_b * p_b))
        
        # Soft-min: exp(-score/T) weighted sum
        # This gives a smooth surrogate that prefers more Strassen blocks
        if not raw_scores:
            return 0.0
        max_s = max(raw_scores)
        exp_terms = [np.exp(-(s - max_s) / temperature) for s in raw_scores]
        total_exp = sum(exp_terms)
        # Normalized soft-min score (lower is better)
        return -np.log(total_exp / len(raw_scores)) + min(raw_scores)

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

        U_entries = []  # (row_idx, col_idx, val)
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

    def compute_tensor(U_np, V_np, W_np, n, m, p):
        """Compute the full multiplication tensor T[i,j,k]."""
        rank = U_np.shape[1]
        T = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    T[i, j, k] = val
        return T

    def try_reduce_rank(U, V, W, rank, time_budget=5.0):
        """
        Polish pass: attempt to reduce rank by finding linearly dependent
        columns in the tensor factorization.
        
        The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        We check if any column can be removed while preserving the tensor.
        """
        start_time = time.time()
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # Compute the reference tensor
        T_ref = compute_tensor(U_np, V_np, W_np, n, m, p)

        current_rank = rank
        best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()

        # Try to eliminate columns one by one
        for trial_col in range(current_rank):
            if time.time() - start_time > time_budget:
                break

            # Check if column trial_col is linearly dependent on others
            # For a column to be removable, its contribution must be zero
            # OR expressible via other columns (which is complex)
            
            u_col = U_np[:, trial_col]
            v_col = V_np[:, trial_col]
            w_col = W_np[:, trial_col]

            # If any factor has all zeros in this column, the column contributes nothing
            if np.all(u_col == 0) or np.all(v_col == 0) or np.all(w_col == 0):
                # Remove this column
                mask = [c for c in range(current_rank) if c != trial_col]
                best_U = U_np[:, mask]
                best_V = V_np[:, mask]
                best_W = W_np[:, mask]
                current_rank = len(mask)
                continue

            # Check if the column's contribution is zero (shouldn't happen if valid)
            # The contribution is the outer product u_col ⊗ v_col ⊗ w_col (reshaped)
            # This is zero iff any of the three is zero, which we already checked

            # Try: can we find coefficients alpha such that removing col r
            # and adding alpha*c to other columns preserves the tensor?
            # This is equivalent to: the rank-1 tensor u⊗v⊗w must be in the span
            # of other rank-1 tensors. This is hard to check directly.
            
            # Simpler approach: try removing and verify
            if current_rank > 1:
                mask = [c for c in range(current_rank) if c != trial_col]
                U_test = U_np[:, mask]
                V_test = V_np[:, mask]
                W_test = W_np[:, mask]
                
                # Verify by computing tensor (expensive but accurate)
                if time.time() - start_time < time_budget - 0.1:
                    T_test = compute_tensor(U_test, V_test, W_test, n, m, p)
                    if np.allclose(T_test, T_ref, atol=1e-10):
                        best_U = U_test
                        best_V = V_test
                        best_W = W_test
                        current_rank = len(mask)

        if current_rank < rank:
            # Convert back to list of lists
            U_out = best_U.tolist()
            V_out = best_V.tolist()
            W_out = best_W.tolist()
            # Ensure integer/half-integer entries
            U_out = [[round(x, 10) if abs(x) > 1e-10 else 0 for x in row] for row in U_out]
            V_out = [[round(x, 10) if abs(x) > 1e-10 else 0 for x in row] for row in V_out]
            W_out = [[round(x, 10) if abs(x) > 1e-10 else 0 for x in row] for row in W_out]
            return {"U": U_out, "V": V_out, "W": W_out}, current_rank

        return None

    # Generate all possible split patterns
    n_split_options = get_splits(n)
    m_split_options = get_splits(m)
    p_split_options = get_splits(p)

    # Score all candidates - only keep those with rank < 75 (schoolbook)
    all_candidates = []
    for ns in n_split_options:
        for ms in m_split_options:
            for ps in p_split_options:
                r = compute_rank(ns, ms, ps)
                if r < n * m * p:  # Only consider improvements over schoolbook
                    s_score = soft_min_score(ns, ms, ps, temperature=0.3)
                    all_candidates.append((r, s_score, ns, ms, ps))

    # Sort by rank first, then by soft score
    all_candidates.sort(key=lambda x: (x[0], x[1]))

    # Try candidates in order, but limit to top few for time efficiency
    best_result = None
    best_rank = float('inf')
    best_rank_result = None

    max_candidates_to_try = min(len(all_candidates), 20)
    for idx in range(max_candidates_to_try):
        if time.time() > deadline:
            break
        r, s_score, ns, ms, ps = all_candidates[idx]
        if r >= best_rank:
            break

        try:
            result, actual_rank = build_decomposition(ns, ms, ps)
            if actual_rank < best_rank:
                best_rank = actual_rank
                best_rank_result = result
        except Exception:
            continue

    # Polish pass on the best result
    if best_rank_result is not None and time.time() < deadline - 15:
        time_remaining = deadline - time.time() - 10
        polished = try_reduce_rank(
            best_rank_result['U'],
            best_rank_result['V'],
            best_rank_result['W'],
            best_rank,
            time_budget=min(15.0, max(1.0, time_remaining))
        )
        if polished is not None:
            best_result = polished
        else:
            best_result = best_rank_result
    elif best_rank_result is not None:
        best_result = best_rank_result
    else:
        # Fallback: schoolbook
        best_result, _ = build_decomposition([n], [m], [p])

    return best_result
# EVOLVE-BLOCK-END
