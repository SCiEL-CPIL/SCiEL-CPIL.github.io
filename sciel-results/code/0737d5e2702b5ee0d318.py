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
    rank_sb = n * m * p  # 54
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
    time_limit = 140  # Leave 10s margin for safety

    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()

    # For [2,3,9], try ranks from 36 down to 30 (known lower bounds suggest ~30-36)
    # Use structured initialization based on tensor structure
    ranks_to_try = list(range(36, 29, -1))  # 36, 35, ..., 30

    for rank in ranks_to_try:
        if time.time() - start_time > time_limit:
            break

        found = False
        # Try multiple structured initializations
        max_restarts = 15
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break

            np.random.seed(12345 + restart * 1000 + rank)

            # Structured initialization: use basis vectors and random combinations
            if restart == 0:
                # Start with a structured pattern
                U = np.zeros((nm, rank))
                V = np.zeros((mp, rank))
                W = np.zeros((pn, rank))
                # Fill with structured patterns
                for r in range(rank):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n*m)) % p
                    if r < n*m*p:
                        U[i*m + j, r] = 1.0
                        V[j*p + k, r] = 1.0
                        W[k*n + i, r] = 1.0
            else:
                # Random initialization with varying scales
                scale = 0.5 + restart * 0.1
                U = np.random.randn(nm, rank) * scale
                V = np.random.randn(mp, rank) * scale
                W = np.random.randn(pn, rank) * scale

            # Alternating least squares with better convergence
            max_iter = 500
            prev_error = float('inf')
            for iteration in range(max_iter):
                if time.time() - start_time > time_limit:
                    break

                # Solve for U given V, W
                # T[a, :, :] = sum_r U[a,r] * V[:,r] outer W[:,r]
                # Reshape for efficient least squares
                VW = np.einsum('br,cr->bcr', V, W)  # shape (mp, pn, rank)
                VW_flat = VW.transpose(2, 0, 1).reshape(rank, mp * pn)  # (rank, mp*pn)
                T_flat_a = T.reshape(nm, mp * pn)  # (nm, mp*pn)

                try:
                    U_new, _, _, _ = np.linalg.lstsq(VW_flat.T, T_flat_a.T, rcond=None)
                    U = U_new.T
                except:
                    break

                # Solve for V given U, W
                UW = np.einsum('ar,cr->acr', U, W)  # shape (nm, pn, rank)
                UW_flat = UW.transpose(2, 0, 1).reshape(rank, nm * pn)  # (rank, nm*pn)
                T_flat_b = T.transpose(1, 0, 2).reshape(mp, nm * pn)  # (mp, nm*pn)

                try:
                    V_new, _, _, _ = np.linalg.lstsq(UW_flat.T, T_flat_b.T, rcond=None)
                    V = V_new.T
                except:
                    break

                # Solve for W given U, V
                UV = np.einsum('ar,br->abr', U, V)  # shape (nm, mp, rank)
                UV_flat = UV.transpose(2, 0, 1).reshape(rank, nm * mp)  # (rank, nm*mp)
                T_flat_c = T.transpose(2, 0, 1).reshape(pn, nm * mp)  # (pn, nm*mp)

                try:
                    W_new, _, _, _ = np.linalg.lstsq(UV_flat.T, T_flat_c.T, rcond=None)
                    W = W_new.T
                except:
                    break

                # Check error every 10 iterations
                if iteration % 10 == 0:
                    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                    error = np.linalg.norm(T_recon - T)
                    if error < 1e-8:
                        found = True
                        break
                    if error > prev_error * 1.1 and iteration > 50:
                        # Diverging, break early
                        break
                    prev_error = error

            if found:
                # Verify the solution is exact
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(T_recon, T, atol=1e-10):
                    best_rank = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    break

        if found:
            continue

    if best_rank < rank_sb:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
