import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    # Build the multiplication tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook baseline
    rank_sb = nm * mp // m  # n * m * p = 54
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1

    start_time = time.time()
    time_limit = 135  # Leave 15s margin

    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()

    # Lower bound: at least max(n*m, m*p, p*n) / something
    # For matrix multiplication (n,m,p), rank >= max(n*m, m*p, p*n)
    # But actually the lower bound is max(n, m, p) for the "dimension"
    # A reasonable lower bound is n*m + m*p + p*n - something
    # Let's try down to 30
    lower_bound = 30

    # Try ranks from rank_sb-1 down to lower_bound
    for rank in range(rank_sb - 1, lower_bound - 1, -1):
        if time.time() - start_time > time_limit:
            break

        found = False
        # Try multiple random initializations
        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break

            np.random.seed(42 + restart * 7919 + rank)

            # Random initialization with small values
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1

            # Alternating least squares
            max_iter = 300
            for iteration in range(max_iter):
                if time.time() - start_time > time_limit:
                    break

                # Solve for U given V, W
                # For each a, we need: sum_r U[a,r] * (V[:,r] outer W[:,r]) = T[a, :, :]
                P_flat = np.zeros((rank, mp * pn))
                for r in range(rank):
                    P_flat[r, :] = np.outer(V[:, r], W[:, r]).reshape(-1)

                T_flat_a = T.reshape(nm, mp * pn)

                try:
                    U_new, _, _, _ = np.linalg.lstsq(P_flat.T, T_flat_a.T, rcond=None)
                    U = U_new.T
                except:
                    break

                # Solve for V given U, W
                P_flat_v = np.zeros((rank, nm * pn))
                for r in range(rank):
                    P_flat_v[r, :] = np.outer(U[:, r], W[:, r]).reshape(-1)

                T_flat_b = T.reshape(mp, nm, pn).transpose(1, 0, 2).reshape(mp, nm * pn)

                try:
                    V_new, _, _, _ = np.linalg.lstsq(P_flat_v.T, T_flat_b.T, rcond=None)
                    V = V_new.T
                except:
                    break

                # Solve for W given U, V
                P_flat_w = np.zeros((rank, nm * mp))
                for r in range(rank):
                    P_flat_w[r, :] = np.outer(U[:, r], V[:, r]).reshape(-1)

                T_flat_c = T.reshape(pn, nm, mp).transpose(1, 2, 0).reshape(pn, nm * mp)

                try:
                    W_new, _, _, _ = np.linalg.lstsq(P_flat_w.T, T_flat_c.T, rcond=None)
                    W = W_new.T
                except:
                    break

                # Check error every 20 iterations
                if iteration % 20 == 0:
                    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                    error = np.linalg.norm(T_recon - T)
                    if error < 1e-8:
                        found = True
                        break

            if found:
                best_rank = rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                break

        if found:
            # Try to go lower
            continue

    if best_rank < rank_sb:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
