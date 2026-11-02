import numpy as np
import time

def construct():
    n, m, p = 6, 6, 6
    deadline = time.time() + 105.0

    # Strassen 2x2 block decomposition with 3x3 blocks
    # Rank = 7 * 27 = 189
    rank = 189
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen 2x2 definitions
    strassen_defs = [
        # M1 = (A11 + A22) * (B11 + B22)
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        # M2 = (A21 + A22) * B11
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        # M3 = A11 * (B12 - B22)
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        # M4 = A22 * (B21 - B11)
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        # M5 = (A11 + A12) * B22
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        # M6 = (A21 - A11) * (B11 + B12)
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        # M7 = (A12 - A22) * (B21 + B22)
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]

    recon_coeffs = [
        # C11 = M1 + M4 - M5 + M7
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
        # C12 = M3 + M5
        [(2, 1.0), (4, 1.0)],
        # C21 = M2 + M4
        [(1, 1.0), (3, 1.0)],
        # C22 = M1 - M2 + M3 + M6
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
    ]

    bs = 3
    r_idx = 0
    for s in range(7):
        s_def = strassen_defs[s]
        for u in range(bs):
            for v in range(bs):
                for j in range(bs):
                    for block_idx, coeff_a in s_def['A']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        i_glob = br * bs + u
                        k_glob = bc * bs + v
                        u_row = i_glob * m + k_glob
                        U[u_row, r_idx] += coeff_a
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = br * bs + v
                        j_glob = bc * bs + j
                        v_row = k_glob * p + j_glob
                        V[v_row, r_idx] += coeff_b
                    for c_block_idx in range(4):
                        gamma = 0.0
                        for (s_idx, g) in recon_coeffs[c_block_idx]:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0.0:
                            br = c_block_idx // 2
                            bc = c_block_idx % 2
                            i_glob = br * bs + u
                            j_glob = bc * bs + j
                            w_row = j_glob * n + i_glob
                            W[w_row, r_idx] += gamma
                    r_idx += 1

    # Verify correctness before any post-processing
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                T[i * m + k, k * p + j, j * n + i] = 1.0

    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(T_recon, T, atol=1e-9):
        # Fallback to standard
        U_std = np.zeros((n * m, n * m * p), dtype=float)
        V_std = np.zeros((m * p, n * m * p), dtype=float)
        W_std = np.zeros((p * n, n * m * p), dtype=float)
        idx = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_std[i * m + k, idx] = 1.0
                    V_std[k * p + j, idx] = 1.0
                    W_std[j * n + i, idx] = 1.0
                    idx += 1
        return {"U": U_std.tolist(), "V": V_std.tolist(), "W": W_std.tolist()}

    # Post-processing: Remove linearly dependent columns
    # A column r is redundant if its contribution can be expressed as a linear combination of other columns
    # We check if removing column r still allows exact reconstruction
    
    # Use QR with column pivoting to identify independent columns
    # The rank of the decomposition is the minimum number of columns needed
    
    # Try to find a minimal subset of columns that still spans the tensor
    # This is equivalent to finding the tensor rank of this specific decomposition
    
    # For now, let's check if there are obvious linear dependencies
    # by looking at the column space of U, V, W jointly
    
    # A simpler approach: try removing each column and check if reconstruction still works
    # This is O(rank^2 * n^3) which is feasible for rank=189
    
    current_rank = rank
    kept_cols = list(range(current_rank))
    
    # Greedy removal: try to remove each column if possible
    # Start from the end and work backwards
    for r in range(current_rank - 1, -1, -1):
        if time.time() > deadline - 10:
            break
        if r not in kept_cols:
            continue
        
        # Try removing column r
        new_kept = [c for c in kept_cols if c != r]
        if len(new_kept) == 0:
            continue
            
        U_test = U[:, new_kept]
        V_test = V[:, new_kept]
        W_test = W[:, new_kept]
        
        T_test = np.einsum('ir,jr,kr->ijk', U_test, V_test, W_test)
        if np.allclose(T_test, T, atol=1e-9):
            kept_cols = new_kept
            current_rank = len(kept_cols)
    
    # Final verification
    U_final = U[:, kept_cols]
    V_final = V[:, kept_cols]
    W_final = W[:, kept_cols]
    
    T_final = np.einsum('ir,jr,kr->ijk', U_final, V_final, W_final)
    if not np.allclose(T_final, T, atol=1e-9):
        # Fallback to original
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
