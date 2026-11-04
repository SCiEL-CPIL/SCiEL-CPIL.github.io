import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_schoolbook = n * m * p  # 24

    # Build target tensor T of shape (nm, mp, pn)
    # T[i*m + j, j*p + k, k*n + i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Try ranks from 18 up to 23 (skip 24 since schoolbook already gives that)
    best_result = None

    for rank in range(18, rank_schoolbook):
        if time.time() > deadline:
            break

        # Number of restarts to try for this rank
        # We have limited time, so be conservative
        max_restarts = 30

        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            # Random initialization with small random values
            np.random.seed(42 + rank * 1000 + restart)
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1

            # Alternating least squares
            converged = False
            max_iter = 500
            prev_err = np.inf

            for it in range(max_iter):
                if time.time() > deadline:
                    break

                # Update U: for each row i, solve (V_k * W_k)^T (V_k * W_k) u_i = T[i,:,:].reshape(-1)
                # Actually, the tensor decomposition is T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
                # For fixed V, W: T[i,:,:] = sum_r U[i,r] * outer(V[:,r], W[:,r]).T
                # So for each i: T[i,:,:].flatten() = sum_r U[i,r] * (V[:,r] * W[:,r]).flatten()
                # This is a linear system: M @ u_i = T[i,:,:].flatten()
                # where M[j*pn+k, r] = V[j,r] * W[k,r]

                # Build M for U update
                # M has shape (mp*pn, rank)
                M_U = np.zeros((mp * pn, rank), dtype=np.float64)
                for r in range(rank):
                    M_U[:, r] = (V[:, r] * W[:, r]).flatten()

                # Solve for U
                # T_flat has shape (nm, mp*pn)
                T_flat = T.reshape(nm, mp * pn)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(M_U, T_flat.T, rcond=None)
                    U_new = U_new.T  # shape (nm, rank)
                except:
                    U_new = U

                # Update V
                # T[j,:,:] but with i and k mixed... 
                # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
                # For fixed U, W: T[:,j,:] = sum_r V[j,r] * outer(U[:,r], W[:,r])
                # T[:,j,:].flatten() = sum_r V[j,r] * (U[:,r] * W[:,r]).flatten()
                # M_V has shape (nm*pn, rank)
                M_V = np.zeros((nm * pn, rank), dtype=np.float64)
                for r in range(rank):
                    M_V[:, r] = (U[:, r] * W[:, r]).flatten()

                T_flat_V = T.reshape(mp, nm * pn)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(M_V, T_flat_V.T, rcond=None)
                    V_new = V_new.T  # shape (mp, rank)
                except:
                    V_new = V

                # Update W
                # T[:,:,k] = sum_r W[k,r] * outer(U[:,r], V[:,r])
                # T[:,:,k].flatten() = sum_r W[k,r] * (U[:,r] * V[:,r]).flatten()
                # M_W has shape (nm*mp, rank)
                M_W = np.zeros((nm * mp, rank), dtype=np.float64)
                for r in range(rank):
                    M_W[:, r] = (U[:, r] * V[:, r]).flatten()

                T_flat_W = T.reshape(pn, nm * mp)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M_W, T_flat_W.T, rcond=None)
                    W_new = W_new.T  # shape (pn, rank)
                except:
                    W_new = W

                U, V, W = U_new, V_new, W_new

                # Check convergence
                # Compute error
                err = 0.0
                for r in range(rank):
                    err += np.sum((U[:, r][:, None] * V[:, r][None, :]) * W[:, r])
                # Actually compute full reconstruction
                recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = np.linalg.norm(recon - T)
                
                if abs(prev_err - err) < 1e-10:
                    converged = True
                    break
                prev_err = err

            # Check if we have a valid decomposition
            if time.time() > deadline:
                break

            # Try to snap to half-integers
            def snap_half_int(x):
                return np.round(x * 2) / 2

            U_snap = snap_half_int(U)
            V_snap = snap_half_int(V)
            W_snap = snap_half_int(W)

            # Verify exactness
            recon = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
            if np.allclose(recon, T, atol=1e-9):
                # Check entries are actually half-integers
                if np.allclose(U_snap * 2, np.round(U_snap * 2), atol=1e-9) and \
                   np.allclose(V_snap * 2, np.round(V_snap * 2), atol=1e-9) and \
                   np.allclose(W_snap * 2, np.round(W_snap * 2), atol=1e-9):
                    # Convert to lists
                    U_list = U_snap.tolist()
                    V_list = V_snap.tolist()
                    W_list = W_snap.tolist()
                    
                    # Verify once more with exact arithmetic
                    recon2 = np.zeros((nm, mp, pn), dtype=np.float64)
                    for r in range(rank):
                        for i in range(nm):
                            for j in range(mp):
                                for k in range(pn):
                                    recon2[i, j, k] += U_list[i][r] * V_list[j][r] * W_list[k][r]
                    if np.allclose(recon2, T, atol=1e-9):
                        return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

