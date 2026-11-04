"""Matrix multiplication 3x5x5 rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0

    # Strassen patterns for 2x2x2: each is (U_entries, V_entries, W_entries)
    # U_entries: list of (i, j, val)
    # V_entries: list of (j, k, val)
    # W_entries: list of (k, i, val)
    strassen_222 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    # Strassen-like decomposition for 3x3x3 using 19 multiplications
    # Based on Winograd's 19-multiplication algorithm for 3x3 matrix multiplication
    # We decompose 3x3x3 into a 2x2x2 Strassen + 3x1x1 + 1x3x1 + 1x1x3 + 1x1x1 + 1x1x1 + 1x1x1 + 1x1x1 + 1x1x1
    # Actually, let's use a known 19-mult 3x3x3 decomposition
    # The 3x3x3 tensor has rank 19 (Winograd)
    # For simplicity, we'll use a block approach: split 3=2+1 in each dimension
    # 2x2x2 (7) + 2x2x1 (4) + 2x1x2 (4) + 1x2x2 (4) + 2x1x1 (2) + 1x2x1 (2) + 1x1x2 (2) + 1x1x1 (1) = 26
    # That's worse than 27. Let's try a proper 19-mult.
    
    # Winograd 19-mult for 3x3x3:
    # This is complex. For now, let's focus on what we can do well.
    # Let's use the block decomposition approach with 2+1 splits and Strassen for 2x2x2.
    
    def compute_rank(n_splits, m_splits, p_splits):
        rank = 0
        for n_b in n_splits:
            for m_b in m_splits:
                for p_b in p_splits:
                    if n_b == 2 and m_b == 2 and p_b == 2:
                        rank += 7
                    else:
                        rank += n_b * m_b * p_b
        return rank

    def build_decomposition(n_splits, m_splits, p_splits):
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
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

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
                        for u_entries, v_entries, w_entries in strassen_222:
                            for li, lj, val in u_entries:
                                gi = i_off + li
                                gj = j_off + lj
                                U[gi * m + gj][col] = val
                            for lj, lk, val in v_entries:
                                gj = j_off + lj
                                gk = k_off + lk
                                V[gj * p + gk][col] = val
                            for lk, li, val in w_entries:
                                gk = k_off + lk
                                gi = i_off + li
                                W[gk * n + gi][col] = val
                            col += 1
                    else:
                        for li in range(n_b):
                            for lj in range(m_b):
                                for lk in range(p_b):
                                    gi = i_off + li
                                    gj = j_off + lj
                                    gk = k_off + lk
                                    U[gi * m + gj][col] = 1
                                    V[gj * p + gk][col] = 1
                                    W[gk * n + gi][col] = 1
                                    col += 1

        assert col == rank, f"Column mismatch: {col} vs {rank}"
        return {"U": U, "V": V, "W": W}, rank

    def verify_decomposition(result):
        """Verify using fully vectorized integer arithmetic."""
        U = np.array(result['U'], dtype=np.int64)
        V = np.array(result['V'], dtype=np.int64)
        W = np.array(result['W'], dtype=np.int64)
        rank = U.shape[1]
        
        if rank == 0:
            return False
            
        # Reshape to 3D: U3[i,j,r], V3[j,k,r], W3[k,i,r]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # For each i,j: T[i,j,:] = U3[i,j,:] @ (V3[j,:,:] * W3[:,i,:].T)
        # Actually: T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # = sum_r U3[i,j,r] * (V3[j,k,r] * W3[k,i,r])
        # For fixed i,j: T[i,j,:] = U3[i,j,:] @ (V3[j,:,:] * W3[:,i,:].T)
        # Wait, that's not right either. Let me think...
        # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # For fixed i,j: define M[r,k] = V3[j,k,r] * W3[k,i,r], then T[i,j,k] = sum_r U3[i,j,r] * M[r,k]
        # So T[i,j,:] = U3[i,j,:] @ M where M is rank x p
        
        T = np.zeros((n, m, p), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                # M[r,k] = V3[j,k,r] * W3[k,i,r]
                M = V3[j, :, :] * W3[:, i, :]  # shape (p, rank) -> need (rank, p)
                # Actually V3[j,:,:] has shape (p, rank), W3[:,i,:] has shape (p, rank)
                # M[r,k] = V3[j,k,r] * W3[k,i,r], so M = (V3[j,:,:] * W3[:,i,:]).T
                M = (V3[j, :, :] * W3[:, i, :]).T  # shape (rank, p)
                T[i, j, :] = U3[i, j, :] @ M  # shape (rank,) @ (rank, p) = (p,)
        
        target = np.ones((n, m, p), dtype=np.int64)
        return np.array_equal(T, target)

    def get_splits(d):
        splits = []
        def _gen(remaining, current):
            if remaining == 0:
                splits.append(tuple(current))
                return
            for s in range(1, remaining + 1):
                current.append(s)
                _gen(remaining - s, current)
                current.pop()
        _gen(d, [])
        return splits

    n_split_options = get_splits(n)
    m_split_options = get_splits(m)
    p_split_options = get_splits(p)

    all_candidates = []
    for ns in n_split_options:
        for ms in m_split_options:
            for ps in p_split_options:
                r = compute_rank(ns, ms, ps)
                all_candidates.append((r, ns, ms, ps))

    all_candidates.sort(key=lambda x: x[0])

    best_result = None
    best_rank = float('inf')

    for r, ns, ms, ps in all_candidates:
        if time.time() > deadline:
            break
        if r >= best_rank:
            break

        try:
            result, actual_rank = build_decomposition(ns, ms, ps)
            if verify_decomposition(result):
                if actual_rank < best_rank:
                    best_rank = actual_rank
                    best_result = result
        except Exception:
            continue

    # Fallback
    if best_result is None:
        result, _ = build_decomposition((n,), (m,), (p,))
        best_result = result

    return best_result
# EVOLVE-BLOCK-END
