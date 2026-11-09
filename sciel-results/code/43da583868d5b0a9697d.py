import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_sb = nm * mp * pn // p  # n*m*p = 84

    # Build index arrays for the tensor
    # For each (i, j, k) in [0,n) x [0,m) x [0,p):
    #   U index: i*m + j
    #   V index: j*p + k
    #   W index: k*n + i
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))
    indices = np.array(indices)
    u_idx = indices[:, 0].astype(np.int64)
    v_idx = indices[:, 1].astype(np.int64)
    w_idx = indices[:, 2].astype(np.int64)
    N = len(u_idx)  # 84

    def compute_vals(U, V, W):
        """Compute all 84 tensor values."""
        return np.sum(U[u_idx] * V[v_idx] * W[w_idx], axis=1)

    def compute_error(U, V, W):
        vals = compute_vals(U, V, W)
        return np.sum((vals - 1.0) ** 2)

    def snap_to_half_integers(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        return U_s, V_s, W_s

    def is_exact(U, V, W, tol=1e-10):
        vals = compute_vals(U, V, W)
        return np.all(np.abs(vals - 1.0) < tol)

    # Schoolbook construction
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    start_time = time.time()
    deadline = start_time + 115  # Leave 5s safety margin

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    # Try to find lower rank decompositions
    # The known lower bound for 3x4x7 is around 72-78 based on literature
    # We'll search from rank_sb-1 downward
    
    for R in range(rank_sb - 1, max(60, rank_sb - 30), -1):
        if time.time() > deadline:
            break

        found = False
        # Multiple random restarts with different initializations
        for attempt in range(20):
            if time.time() > deadline:
                break

            # Initialize with small random values
            np.random.seed(hash((R, attempt)) % (2**32))
            scale = 1.0 / np.sqrt(R)
            U = np.random.randn(nm, R) * scale
            V = np.random.randn(mp, R) * scale
            W = np.random.randn(pn, R) * scale

            # Gradient descent with adaptive learning rate
            lr = 0.05
            best_err = np.inf
            patience = 0
            max_patience = 50

            for step in range(200):
                if time.time() > deadline:
                    break

                # Compute values and error
                vals = compute_vals(U, V, W)
                errs = vals - 1.0
                err_val = np.sum(errs ** 2)

                if err_val < 1e-10:
                    break

                if err_val < best_err:
                    best_err = err_val
                    patience = 0
                else:
                    patience += 1

                if patience >= max_patience:
                    # Reduce learning rate and reset patience
                    lr *= 0.5
                    patience = 0
                    if lr < 1e-6:
                        break

                # Compute gradients
                dU = np.zeros((nm, R))
                dV = np.zeros((mp, R))
                dW = np.zeros((pn, R))

                # Vectorized gradient computation using np.add.at
                term_U = 2 * errs[:, None] * V[v_idx] * W[w_idx]
                term_V = 2 * errs[:, None] * U[u_idx] * W[w_idx]
                term_W = 2 * errs[:, None] * U[u_idx] * V[v_idx]

                # Scatter-add
                for r in range(R):
                    np.add.at(dU, u_idx, term_U[:, r])
                    np.add.at(dV, v_idx, term_V[:, r])
                    np.add.at(dW, w_idx, term_W[:, r])

                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW

            # After gradient descent, try to snap to half-integers
            U_s, V_s, W_s = snap_to_half_integers(U, V, W)
            if is_exact(U_s, V_s, W_s):
                best_rank = R
                best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
                found = True
                break

            # Also try integer snapping
            U_i, V_i, W_i = np.round(U), np.round(V), np.round(W)
            if is_exact(U_i, V_i, W_i):
                best_rank = R
                best_U, best_V, best_W = U_i.copy(), V_i.copy(), W_i.copy()
                found = True
                break

        if found:
            # Continue searching for even lower rank
            pass

    # If no better solution found, return schoolbook
    if best_rank == rank_sb:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
