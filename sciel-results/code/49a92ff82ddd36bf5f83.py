import numpy as np
import time

def construct():
    n, m, p = 6, 6, 6
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s

    # Target tensor T[u, v, w] where u=i*m+k, v=k*p+j, w=j*n+i
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                T[i * m + k, k * p + j, j * n + i] = 1.0

    # Block-Strassen with block size 2: 3x3 blocks of 2x2
    bs = 2
    nb = n // bs  # 3

    # Strassen 2x2 definitions (7 multiplications)
    strassen_defs = [
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]

    recon_coeffs = [
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
        [(2, 1.0), (4, 1.0)],
        [(1, 1.0), (3, 1.0)],
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
    ]

    rank_bs = nb * nb * nb * 7  # 189
    U = np.zeros((n * m, rank_bs), dtype=float)
    V = np.zeros((m * p, rank_bs), dtype=float)
    W = np.zeros((p * n, rank_bs), dtype=float)

    r_idx = 0
    for i_b in range(nb):
        for k_b in range(nb):
            for j_b in range(nb):
                for s in range(7):
                    s_def = strassen_defs[s]
                    for block_idx, coeff_a in s_def['A']:
                        ar = block_idx // 2
                        ac = block_idx % 2
                        i_glob = i_b * bs + ar
                        k_glob = k_b * bs + ac
                        U[i_glob * m + k_glob, r_idx] += coeff_a
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = k_b * bs + br
                        j_glob = j_b * bs + bc
                        V[k_glob * p + j_glob, r_idx] += coeff_b
                    for c_block_idx in range(4):
                        gamma = 0.0
                        for (s_idx, g) in recon_coeffs[c_block_idx]:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0.0:
                            cr = c_block_idx // 2
                            cc = c_block_idx % 2
                            i_glob = i_b * bs + cr
                            j_glob = j_b * bs + cc
                            W[j_glob * n + i_glob, r_idx] += gamma
                    r_idx += 1

    # Verify
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(T_recon, T, atol=1e-9):
        # Fallback: trivial decomposition
        rank_trivial = n * m * p
        U_t = np.zeros((n * m, rank_trivial), dtype=float)
        V_t = np.zeros((m * p, rank_trivial), dtype=float)
        W_t = np.zeros((p * n, rank_trivial), dtype=float)
        idx = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_t[i * m + k, idx] = 1.0
                    V_t[k * p + j, idx] = 1.0
                    W_t[j * n + i, idx] = 1.0
                    idx += 1
        return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}

    # Try to reduce rank by finding and merging dependent terms
    # Two terms r1, r2 can be merged if U[:,r1] is proportional to U[:,r2],
    # V[:,r1] is proportional to V[:,r2], AND W[:,r1] is proportional to W[:,r2].
    # In that case, we can keep only one term (with combined coefficient).
    
    # For efficiency, hash the support patterns and check proportionality
    # This is O(rank^2) which is 189^2 = ~35k checks, each O(36) = manageable
    
    if time.time() < deadline - 30:
        # Find mergeable pairs
        used = [False] * rank_bs
        merged = [True] * rank_bs  # initially all kept
        new_U = U.copy()
        new_V = V.copy()
        new_W = W.copy()
        
        # For each pair, check if they can be merged
        # Term r can be absorbed into term r' if:
        # U[:,r] = c * U[:,r'] for some c != 0
        # V[:,r] = c * V[:,r'] (same c)
        # W[:,r] = c * W[:,r'] (same c)
        # Then we can set new_U[:,r'] += c*new_U[:,r] etc. and zero out term r.
        
        # Simpler: just check for exact duplicates (c=1) first
        # Then check for negatives (c=-1)
        
        for r in range(rank_bs):
            if not merged[r]:
                continue
            for r2 in range(r + 1, rank_bs):
                if not merged[r2]:
                    continue
                # Check if term r is a scalar multiple of term r2
                # Find nonzero entries in U[:,r]
                nz_u = np.nonzero(U[:, r])[0]
                if len(nz_u) == 0:
                    continue
                # Get the scalar from first nonzero entry
                if U[nz_u[0], r2] == 0:
                    continue
                c = U[nz_u[0], r] / U[nz_u[0], r2]
                # Verify proportionality for U
                if not np.allclose(U[:, r], c * U[:, r2], atol=1e-10):
                    continue
                # Verify proportionality for V
                nz_v = np.nonzero(V[:, r])[0]
                if len(nz_v) == 0:
                    continue
                if V[nz_v[0], r2] == 0:
                    continue
                c_v = V[nz_v[0], r] / V[nz_v[0], r2]
                if abs(c_v - c) > 1e-10:
                    continue
                if not np.allclose(V[:, r], c * V[:, r2], atol=1e-10):
                    continue
                # Verify proportionality for W
                nz_w = np.nonzero(W[:, r])[0]
                if len(nz_w) == 0:
                    continue
                if W[nz_w[0], r2] == 0:
                    continue
                c_w = W[nz_w[0], r] / W[nz_w[0], r2]
                if abs(c_w - c) > 1e-10:
                    continue
                if not np.allclose(W[:, r], c * W[:, r2], atol=1e-10):
                    continue
                # Merge: add term r to term r2
                new_U[:, r2] += c * new_U[:, r]
                new_V[:, r2] += c * new_V[:, r]
                new_W[:, r2] += c * new_W[:, r]
                new_U[:, r] = 0
                new_V[:, r] = 0
                new_W[:, r] = 0
                merged[r] = False
                break
        
        # Count remaining terms
        kept = np.where(merged)[0]
        if len(kept) < rank_bs:
            U_red = new_U[:, kept]
            V_red = new_V[:, kept]
            W_red = new_W[:, kept]
            # Verify
            T_red = np.einsum('ir,jr,kr->ijk', U_red, V_red, W_red)
            if np.allclose(T_red, T, atol=1e-9):
                return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
