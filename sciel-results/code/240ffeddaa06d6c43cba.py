import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140

    n, m, p = DIMENSIONS

    # Construct rank 50 decomposition
    rank_init = 50
    U = np.zeros((n * m, rank_init), dtype=np.float64)
    V = np.zeros((m * p, rank_init), dtype=np.float64)
    W = np.zeros((p * n, rank_init), dtype=np.float64)

    col = 0

    def emit_strassen(i0, j0, k0):
        nonlocal col
        # M1
        U[i0*m + j0, col] = 1.0
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[j0*p + k0, col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        # M2
        U[(i0+1)*m + j0, col] = 1.0
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[j0*p + k0, col] = 1.0
        W[k0*n + (i0+1), col] = 1.0
        W[(k0+1)*n + (i0+1), col] = -1.0
        col += 1
        # M3
        U[i0*m + j0, col] = 1.0
        V[j0*p + (k0+1), col] = 1.0
        V[(j0+1)*p + (k0+1), col] = -1.0
        W[(k0+1)*n + i0, col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        # M4
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[(j0+1)*p + k0, col] = 1.0
        V[j0*p + k0, col] = -1.0
        W[k0*n + i0, col] = 1.0
        W[k0*n + (i0+1), col] = 1.0
        col += 1
        # M5
        U[i0*m + j0, col] = 1.0
        U[i0*m + (j0+1), col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = -1.0
        W[(k0+1)*n + i0, col] = 1.0
        col += 1
        # M6
        U[(i0+1)*m + j0, col] = 1.0
        U[i0*m + j0, col] = -1.0
        V[j0*p + k0, col] = 1.0
        V[j0*p + (k0+1), col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        # M7
        U[i0*m + (j0+1), col] = 1.0
        U[(i0+1)*m + (j0+1), col] = -1.0
        V[(j0+1)*p + k0, col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = 1.0
        col += 1

    # 6 Strassen blocks
    emit_strassen(0, 0, 0)
    emit_strassen(0, 0, 2)
    emit_strassen(0, 2, 0)
    emit_strassen(0, 2, 2)
    emit_strassen(0, 0, 4)
    emit_strassen(0, 2, 4)

    # 8 schoolbook terms for k=6
    for i in range(2):
        for j in range(4):
            U[i*m + j, col] = 1.0
            V[j*p + 6, col] = 1.0
            W[6*n + i, col] = 1.0
            col += 1

    # Verify
    U_3d = U.reshape(n, m, rank_init)
    V_3d = V.reshape(m, p, rank_init)
    W_3d = W.reshape(p, n, rank_init)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    if not np.allclose(T, 1.0, atol=1e-9):
        return None

    current_rank = rank_init
    current_U, current_V, current_W = U, V, W

    # Rank reduction loop
    while True:
        if time.time() > DEADLINE - 5:
            break

        # Compute tensor vectors
        tensors = []
        for r in range(current_rank):
            U_r = current_U[:, r].reshape(n, m)
            V_r = current_V[:, r].reshape(m, p)
            W_r = current_W[:, r].reshape(p, n)
            T_r = np.einsum('ij,jk,ki->ijk', U_r, V_r, W_r)
            tensors.append(T_r.flatten())

        T_mat = np.array(tensors).T  # Shape (56, current_rank)

        # Find null space
        # Use SVD
        try:
            U_svd, s, Vh = np.linalg.svd(T_mat, full_matrices=False)
        except:
            break

        # Check for small singular values
        tol = 1e-9
        rank_mat = np.sum(s > tol)

        if rank_mat == current_rank:
            break

        # Find a null vector
        # The null space is spanned by rows of Vh corresponding to small s
        # We want a vector c such that T_mat @ c = 0
        # Take the last row of Vh (corresponding to smallest s)
        # Actually, if rank < current_rank, the last (current_rank - rank_mat) rows of Vh are null vectors
        # We pick one with a large component to avoid division by zero
        null_vecs = Vh[rank_mat:]  # Shape (num_null, current_rank)

        # Pick the null vector with the largest max component
        best_vec = None
        best_max = 0
        for v in null_vecs:
            mx = np.max(np.abs(v))
            if mx > best_max:
                best_max = mx
                best_vec = v

        if best_vec is None or best_max < tol:
            break

        # Find index k with max component
        k = np.argmax(np.abs(best_vec))
        c_k = best_vec[k]

        if abs(c_k) < tol:
            continue

        # Update coefficients
        # T_k = - sum_{r != k} (c_r / c_k) T_r
        # New sum = sum_{r != k} (1 - c_r/c_k) T_r  <-- Wait, sign check
        # Dependency: sum c_r T_r = 0 => c_k T_k = - sum_{r != k} c_r T_r => T_k = - sum (c_r/c_k) T_r
        # Original Sum S = sum T_r = sum_{r != k} T_r + T_k
        # S = sum_{r != k} T_r - sum_{r != k} (c_r/c_k) T_r = sum_{r != k} (1 - c_r/c_k) T_r

        new_rank = current_rank - 1
        new_U = np.zeros((n*m, new_rank))
        new_V = np.zeros((m*p, new_rank))
        new_W = np.zeros((p*n, new_rank))

        new_col = 0
        for r in range(current_rank):
            if r == k:
                continue
            scale = 1.0 - (best_vec[r] / c_k)
            new_U[:, new_col] = scale * current_U[:, r]
            new_V[:, new_col] = scale * current_V[:, r]
            new_W[:, new_col] = scale * current_W[:, r]
            new_col += 1

        # Verify new decomposition
        new_U_3d = new_U.reshape(n, m, new_rank)
        new_V_3d = new_V.reshape(m, p, new_rank)
        new_W_3d = new_W.reshape(p, n, new_rank)
        T_new = np.einsum('ijr,jkr,kir->ijk', new_U_3d, new_V_3d, new_W_3d)

        if np.allclose(T_new, 1.0, atol=1e-9):
            current_rank = new_rank
            current_U, current_V, current_W = new_U, new_V, new_W
        else:
            break

    # Final polish: try to find more null vectors with higher precision
    # Use QR decomposition for better numerical stability
    for _ in range(100):
        if time.time() > DEADLINE - 2:
            break

        T_mat = np.zeros((n*m*p, current_rank))
        for r in range(current_rank):
            U_r = current_U[:, r].reshape(n, m)
            V_r = current_V[:, r].reshape(m, p)
            W_r = current_W[:, r].reshape(p, n)
            T_r = np.einsum('ij,jk,ki->ijk', U_r, V_r, W_r)
            T_mat[:, r] = T_r.flatten()

        try:
            # Use QR with column pivoting for better null space identification
            Q, R, P = np.linalg.qr(np.hstack([T_mat, np.eye(current_rank)]), mode='complete')
            # The last current_rank columns of Q span the null space
            # But we need to be careful about the structure

            # Alternative: use SVD with tighter tolerance
            U_svd, s, Vh = np.linalg.svd(T_mat, full_matrices=True)
            tol = 1e-12
            rank_mat = np.sum(s > tol)

            if rank_mat >= current_rank:
                break

            null_vecs = Vh[rank_mat:current_rank]
            for nv in null_vecs:
                mx = np.max(np.abs(nv))
                if mx < tol:
                    continue
                k = np.argmax(np.abs(nv))
                c_k = nv[k]
                if abs(c_k) < tol:
                    continue

                new_rank = current_rank - 1
                new_U = np.zeros((n*m, new_rank))
                new_V = np.zeros((m*p, new_rank))
                new_W = np.zeros((p*n, new_rank))

                new_col = 0
                for r in range(current_rank):
                    if r == k:
                        continue
                    scale = 1.0 - (nv[r] / c_k)
                    new_U[:, new_col] = scale * current_U[:, r]
                    new_V[:, new_col] = scale * current_V[:, r]
                    new_W[:, new_col] = scale * current_W[:, r]
                    new_col += 1

                new_U_3d = new_U.reshape(n, m, new_rank)
                new_V_3d = new_V.reshape(m, p, new_rank)
                new_W_3d = new_W.reshape(p, n, new_rank)
                T_new = np.einsum('ijr,jkr,kir->ijk', new_U_3d, new_V_3d, new_W_3d)

                if np.allclose(T_new, 1.0, atol=1e-10):
                    current_rank = new_rank
                    current_U, current_V, current_W = new_U, new_V, new_W
                    break
            else:
                break
        except:
            break

    return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}
# EVOLVE-BLOCK-END
