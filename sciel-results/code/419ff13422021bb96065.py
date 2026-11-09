import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = 4, 4, 6

    # Strassen 2x2 x 2x2 coefficients
    alpha = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    beta = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
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

    # Pre-compute non-zero indices for alpha and beta to speed up loops
    alpha_nz = [[(a, alpha[s][a]) for a in range(4) if alpha[s][a] != 0] for s in range(7)]
    beta_nz = [[(b, beta[s][b]) for b in range(4) if beta[s][b] != 0] for s in range(7)]

    # Build the 77-rank decomposition
    R = 77
    U = np.zeros((16, R), dtype=int)
    V = np.zeros((24, R), dtype=int)
    W = np.zeros((24, R), dtype=int)

    col = 0
    for s in range(7):
        if time.time() > deadline:
            break
        
        # Inner Strassen for columns 0,1 of the 4x6 result
        for si in range(7):
            # U construction
            for a, val_a in alpha_nz[si]:
                x_row, x_col = a // 2, a % 2
                for r, val_r in alpha_nz[s]:
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                    U[u_idx, col] += val_a * val_r

            # V construction
            for b, val_b in beta_nz[si]:
                y_row, y_inner_col = b // 2, b % 2
                for c, val_c in beta_nz[s]:
                    ac, bc = c // 2, c % 2
                    v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                    V[v_idx, col] += val_b * val_c

            # W construction
            for (a, b, gamma) in recon[s]:
                for (za, zb, zgamma) in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma

            col += 1

        # Schoolbook for column 2 of the 4x6 result
        for u in range(2):
            for v in range(2):
                # U construction
                for r, val_r in alpha_nz[s]:
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                    U[u_idx, col] += val_r

                # V construction
                for c, val_c in beta_nz[s]:
                    ac, bc = c // 2, c % 2
                    v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                    V[v_idx, col] += val_c

                # W construction
                for (a, b, gamma) in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma

                col += 1

    # Try to reduce rank by removing zero columns and merging
    # Check for zero columns
    nonzero_mask = np.any(U != 0, axis=0) & np.any(V != 0, axis=0) & np.any(W != 0, axis=0)
    
    if not np.all(nonzero_mask):
        keep = np.where(nonzero_mask)[0]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]

    # Also check for duplicate columns that can be merged
    # Build a hash of each column
    def col_hash(mat, col):
        return tuple(mat[:, col].tolist())
    
    R_final = U.shape[1]
    col_keys = {}
    for j in range(R_final):
        key = (col_hash(U, j), col_hash(V, j), col_hash(W, j))
        if key not in col_keys:
            col_keys[key] = []
        col_keys[key].append(j)
    
    # Keep only one copy of each unique column
    new_cols = []
    for key, indices in col_keys.items():
        new_cols.append(indices[0])
    
    if len(new_cols) < R_final:
        U = U[:, new_cols]
        V = V[:, new_cols]
        W = W[:, new_cols]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
