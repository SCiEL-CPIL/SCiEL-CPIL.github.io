"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s limit

    n, m, p = 2, 2, 13

    # Target tensor T[(i*m+k), (k*p+j), (j*n+i)] = 1
    T_target = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T_target[i * m + k, k * p + j, j * n + i] = 1.0

    def get_strassen_fallback():
        # Verified Strassen decomposition for 2x2 * 2x2
        strassen_U = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1],
        ], dtype=float)
        strassen_V = np.array([
            [1, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, -1, 0, 1, 0],
            [1, 0, -1, 0, 1, 0, 1],
        ], dtype=float)
        strassen_W = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [1, -1, 1, 0, 0, 1, 0],
            [0, 1, -1, 1, 1, 0, 0],
            [1, 1, 0, -1, 0, -1, 0],
        ], dtype=float)

        # Verify Strassen on 2x2x2
        S_target = np.zeros((4, 4, 4), dtype=float)
        S_target[0, 0, 0] = 1
        S_target[0, 1, 2] = 1
        S_target[1, 2, 0] = 1
        S_target[1, 3, 2] = 1
        S_target[2, 0, 1] = 1
        S_target[2, 1, 3] = 1
        S_target[3, 2, 1] = 1
        S_target[3, 3, 3] = 1

        S_recon = np.einsum('at,bt,ct->abc', strassen_U, strassen_V, strassen_W)
        if not np.allclose(S_recon, S_target, atol=1e-10):
            # Fallback to schoolbook
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            column = 0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        U[i * m + k, column] = 1
                        V[k * p + j, column] = 1
                        W[j * n + i, column] = 1
                        column += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        # Build full decomposition: 6 Strassen blocks + 1 schoolbook block
        total_rank = 6 * 7 + 4  # = 46

        U_full = np.zeros((n * m, total_rank))
        V_full = np.zeros((m * p, total_rank))
        W_full = np.zeros((p * n, total_rank))

        col = 0
        for block in range(6):
            j0 = 2 * block
            j1 = 2 * block + 1
            for t in range(7):
                for a in range(4):
                    U_full[a, col] = strassen_U[a, t]
                for k in range(2):
                    for jl in range(2):
                        j_global = j0 + jl
                        v_local = k * 2 + jl
                        v_global = k * p + j_global
                        V_full[v_global, col] = strassen_V[v_local, t]
                for jl in range(2):
                    for i in range(2):
                        j_global = j0 + jl
                        w_local = jl * 2 + i
                        w_global = j_global * n + i
                        W_full[w_global, col] = strassen_W[w_local, t]
                col += 1

        # Schoolbook block for column 12
        j = 12
        for i in range(n):
            for k in range(m):
                U_full[i * m + k, col] = 1
                V_full[k * p + j, col] = 1
                W_full[j * n + i, col] = 1
                col += 1

        # Verify full decomposition
        T_recon = np.einsum('it,jt,kt->ijk', U_full, V_full, W_full)
        if not np.allclose(T_recon, T_target, atol=1e-10):
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            column = 0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        U[i * m + k, column] = 1
                        V[k * p + j, column] = 1
                        W[j * n + i, column] = 1
                        column += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    def snap_and_verify(U, V, W, T_target, atol=1e-9):
        """Snap to half-integers and verify exact decomposition."""
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        T_check = np.einsum('it,jt,kt->ijk', U_int, V_int, W_int)
        if np.allclose(T_check, T_target, atol=atol):
            return U_int, V_int, W_int
        return None

    def smooth_polish(U, V, W, T_target, max_steps=200, deadline=None):
        """Smooth gradient polish with epigraph-style residual minimization,
        followed by a high-precision snap pass. Uses soft residual norm."""
        eta = 0.02
        for step in range(max_steps):
            if deadline is not None and time.time() > deadline:
                break
            T_recon = np.einsum('it,jt,kt->ijk', U, V, W)
            Res = T_recon - T_target
            # Soft (smooth) residual: use squared norm gradient (already smooth)
            grad_U = np.einsum('ijk,jt,kt->it', Res, V, W)
            grad_V = np.einsum('ijk,it,kt->jt', Res, U, W)
            grad_W = np.einsum('ijk,it,jt->kt', Res, U, V)
            U -= eta * grad_U
            V -= eta * grad_V
            W -= eta * grad_W
            err = np.linalg.norm(Res)
            if err < 1e-6:
                res = snap_and_verify(U, V, W, T_target, atol=1e-9)
                if res is not None:
                    return res
        # Final high-precision polish with tiny steps
        for step in range(50):
            if deadline is not None and time.time() > deadline:
                break
            T_recon = np.einsum('it,jt,kt->ijk', U, V, W)
            Res = T_recon - T_target
            grad_U = np.einsum('ijk,jt,kt->it', Res, V, W)
            grad_V = np.einsum('ijk,it,kt->jt', Res, U, W)
            grad_W = np.einsum('ijk,it,jt->kt', Res, U, V)
            U -= 0.002 * grad_U
            V -= 0.002 * grad_V
            W -= 0.002 * grad_W
            err = np.linalg.norm(Res)
            if err < 1e-7:
                res = snap_and_verify(U, V, W, T_target, atol=1e-9)
                if res is not None:
                    return res
        return None

    # Try to find rank 45, 44 via gradient descent + smooth polish + snap
    for R in [45, 44]:
        if time.time() > deadline - 35:
            break
        for restart in range(4):
            if time.time() > deadline - 35:
                break
            random.seed(1000 + restart * 7 + R)
            np.random.seed(1000 + restart * 7 + R)
            # Initialize with small random integers, then scale
            U = np.random.randint(-1, 2, (n * m, R)).astype(float)
            V = np.random.randint(-1, 2, (m * p, R)).astype(float)
            W = np.random.randint(-1, 2, (p * n, R)).astype(float)

            # Gradient descent
            eta = 0.05
            for step in range(150):
                if time.time() > deadline - 35:
                    break
                T_recon = np.einsum('it,jt,kt->ijk', U, V, W)
                Res = T_recon - T_target
                grad_U = np.einsum('ijk,jt,kt->it', Res, V, W)
                grad_V = np.einsum('ijk,it,kt->jt', Res, U, W)
                grad_W = np.einsum('ijk,it,jt->kt', Res, U, V)
                U -= eta * grad_U
                V -= eta * grad_V
                W -= eta * grad_W
                err = np.linalg.norm(Res)
                if err < 1e-4:
                    break

            # Smooth polish pass
            res = smooth_polish(U, V, W, T_target, max_steps=100, deadline=deadline - 35)
            if res is not None:
                U_int, V_int, W_int = res
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}

    # Cyclic-structure exploitation: try to share rank terms across the 13 columns
    # using the cyclic shift of C_13. The idea: if the 13-column block has a
    # circulant-like structure, we might reduce rank. This is speculative; we
    # only attempt it if we have time, and only return if verified.
    if time.time() < deadline - 40:
        # Attempt: use a single "cyclic" decomposition where the 13 columns
        # are treated as a cyclic convolution. This is hard to verify quickly,
        # so we skip unless we have ample time. For now, rely on the fallback.
        pass

    # Return the verified rank-46 Strassen decomposition
    return get_strassen_fallback()
# EVOLVE-BLOCK-END
