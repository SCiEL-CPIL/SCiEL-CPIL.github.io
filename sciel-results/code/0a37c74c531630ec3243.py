"""
Rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.

Strategy:
1. Build a rank-46 decomposition using 4 Strassen 2x2 blocks + schoolbook for the
   remaining cross-terms (the 5th row/column of the 5x5 matrix).
2. Optionally attempt to reduce rank further via random initialization + gradient
   descent, snapping to half-integers, and validating.
3. Return the best valid decomposition found (lowest rank).

The tensor T has entries T[u, v, w] = 1 iff the middle indices match, where:
  u = i*m + j  (i in [0,n), j in [0,m))
  v = j*p + k  (j in [0,m), k in [0,p))
  w = k*n + i  (k in [0,p), i in [0,n))
  n=2, m=5, p=5
"""

import time
import numpy as np

DIMENSIONS = [2, 5, 5]


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10

    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def snap_to_half(arr):
        return np.round(arr * 2) / 2

    def snap_to_quarter(arr):
        return np.round(arr * 4) / 4

    def try_snap_validate(U, V, W):
        # Try half-integers first (most likely for Strassen-based)
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        # Try integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
        # Try quarters
        U_q = snap_to_quarter(U)
        V_q = snap_to_quarter(V)
        W_q = snap_to_quarter(W)
        if validate(U_q, V_q, W_q):
            return True, U_q, V_q, W_q
        return False, None, None, None

    def build_rank46():
        """
        Build a rank-46 decomposition:
        - 4 Strassen 2x2 blocks covering the 4x4 sub-matrix (indices 0-3 in j and k)
        - Schoolbook for the remaining terms involving index 4 in j or k
        """
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))

        def add_strassen(j_off, k_off, col_start):
            """Add a Strassen 2x2*2x2 block at offset (j_off, k_off) starting at col_start.
            
            For multiplying:
              A_sub = [[A[i0,j0], A[i0,j0+1]],
                       [A[i1,j0], A[i1,j0+1]]]
              B_sub = [[B[j0,k0], B[j0,k0+1]],
                       [B[j0+1,k0], B[j0+1,k0+1]]]
              C_sub = A_sub * B_sub
            
            Where i0=0, i1=1 (since n=2), j0=j_off, j1=j_off+1, k0=k_off, k1=k_off+1
            """
            c = col_start
            i0, i1 = 0, 1
            j0, j1 = j_off, j_off + 1
            k0, k1 = k_off, k_off + 1
            
            # P1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
            U[i0 * m + j0][c] += 1
            U[i1 * m + j1][c] += 1
            V[j0 * p + k0][c] += 1
            V[j1 * p + k1][c] += 1
            W[k0 * n + i0][c] += 1
            W[k0 * n + i1][c] += 1
            W[k1 * n + i0][c] += 1
            W[k1 * n + i1][c] += 1
            c += 1

            # P2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
            U[i1 * m + j0][c] += 1
            U[i1 * m + j1][c] += 1
            V[j0 * p + k0][c] += 1
            W[k0 * n + i1][c] += 1
            c += 1

            # P3 = A[i0,j0] * (B[j0,k1] - B[j1,k1])
            U[i0 * m + j0][c] += 1
            V[j0 * p + k1][c] += 1
            V[j1 * p + k1][c] -= 1
            W[k1 * n + i0][c] += 1
            c += 1

            # P4 = A[i1,j1] * (B[j1,k0] - B[j0,k0])
            U[i1 * m + j1][c] += 1
            V[j1 * p + k0][c] += 1
            V[j0 * p + k0][c] -= 1
            W[k0 * n + i1][c] += 1
            c += 1

            # P5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
            U[i0 * m + j0][c] += 1
            U[i0 * m + j1][c] += 1
            V[j1 * p + k1][c] += 1
            W[k1 * n + i0][c] += 1
            c += 1

            # P6 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k1])
            U[i1 * m + j0][c] += 1
            U[i0 * m + j0][c] -= 1
            V[j0 * p + k0][c] += 1
            V[j0 * p + k1][c] += 1
            W[k0 * n + i1][c] += 1
            W[k1 * n + i1][c] += 1
            c += 1

            # P7 = (A[i0,j1] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
            U[i0 * m + j1][c] += 1
            U[i1 * m + j1][c] -= 1
            V[j1 * p + k0][c] += 1
            V[j1 * p + k1][c] += 1
            W[k0 * n + i0][c] += 1
            W[k1 * n + i0][c] += 1
            c += 1

        # Add 4 Strassen blocks for the 4x4 sub-matrix
        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)

        col = 28

        # Schoolbook for terms involving j=4 or k=4
        # Terms where j=4, k in [0,4): A[i,4] * B[4,k]
        for i in range(n):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10:
                    return U, V, W

        # Terms where j in [0,4), k=4: A[i,j] * B[j,4]
        for i in range(n):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
                if time.time() > deadline - 10:
                    return U, V, W

        # Term where j=4, k=4: A[i,4] * B[4,4]
        for i in range(n):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            if time.time() > deadline - 10:
                return U, V, W

        return U, V, W

    def newton_polish(U, V, W, max_iter=100, tol=1e-14):
        """High-precision Newton-style polish for near-exact solutions."""
        U = U.copy()
        V = V.copy()
        W = W.copy()

        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)
            if loss < tol:
                break

            # Compute gradients
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)

            # Adaptive step size
            grad_norm = np.sqrt(np.sum(gradU**2) + np.sum(gradV**2) + np.sum(gradW**2))
            if grad_norm < 1e-15:
                break
            lr = 0.005 * min(1.0, 1e-6 / grad_norm)

            U -= lr * gradU
            V -= lr * gradV
            W -= lr * gradW

        return U, V, W

    def fast_optimize(U_init, V_init, W_init, target_rank, max_iter=300, lr=0.05,
                      beta1=0.9, beta2=0.999, eps=1e-8, lr_decay=0.9995):
        """Adam optimizer with adaptive restarts."""
        R = target_rank
        U = U_init[:, :R].copy()
        V = V_init[:, :R].copy()
        W = W_init[:, :R].copy()

        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)

        prev_loss = float('inf')
        no_improve = 0
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        current_lr = lr
        restart_count = 0
        t = 0

        for it in range(max_iter):
            if time.time() > deadline - 15:
                break

            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)

            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            if loss < 1e-10:
                break

            # Stagnation detection with restart
            if loss > prev_loss * 1.005 and no_improve > 15:
                restart_count += 1
                if restart_count > 12:
                    break
                # Smaller perturbation for stability
                U += np.random.randn(*U.shape) * 0.01
                V += np.random.randn(*V.shape) * 0.01
                W += np.random.randn(*W.shape) * 0.01
                mU = np.zeros_like(U)
                vU = np.zeros_like(U)
                mV = np.zeros_like(V)
                vV = np.zeros_like(V)
                mW = np.zeros_like(W)
                vW = np.zeros_like(W)
                no_improve = 0
                prev_loss = loss
                t = 0
                continue
            else:
                no_improve += 1
                prev_loss = loss

            # Adam update
            t += 1
            # Cosine-like decay
            current_lr = lr * (1 + np.cos(np.pi * it / max_iter)) / 2

            # Gradients
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)

            # Adam updates
            mU = beta1 * mU + (1 - beta1) * gradU
            vU = beta2 * vU + (1 - beta2) * gradU ** 2
            mhatU = mU / (1 - beta1 ** t)
            vhatU = vU / (1 - beta2 ** t)
            U -= current_lr * mhatU / (np.sqrt(vhatU) + eps)

            mV = beta1 * mV + (1 - beta1) * gradV
            vV = beta2 * vV + (1 - beta2) * gradV ** 2
            mhatV = mV / (1 - beta1 ** t)
            vhatV = vV / (1 - beta2 ** t)
            V -= current_lr * mhatV / (np.sqrt(vhatV) + eps)

            mW = beta1 * mW + (1 - beta1) * gradW
            vW = beta2 * vW + (1 - beta2) * gradW ** 2
            mhatW = mW / (1 - beta1 ** t)
            vhatW = vW / (1 - beta2 ** t)
            W -= current_lr * mhatW / (np.sqrt(vhatW) + eps)

        return best_U, best_V, best_W

    def build_schoolbook():
        """Build the schoolbook rank-50 decomposition as fallback."""
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))

        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

        return U, V, W

    # Step 1: Build rank-46 Strassen-based decomposition
    U46, V46, W46 = build_rank46()

    # Validate rank-46
    ok46, U46s, V46s, W46s = try_snap_validate(U46, V46, W46)
    if not ok46:
        # Try Newton polish then snap
        U46p, V46p, W46p = newton_polish(U46, V46, W46, max_iter=50)
        ok46, U46s, V46s, W46s = try_snap_validate(U46p, V46p, W46p)

    if ok46:
        best_U, best_V, best_W = U46s, V46s, W46s
        best_rank = 46
    else:
        # Fall back to schoolbook
        best_U, best_V, best_W = build_schoolbook()
        best_rank = 50

    # Step 2: Try to find lower rank decompositions with multiple strategies
    if time.time() < deadline - 30:
        for target_rank in range(45, 38, -1):
            if time.time() > deadline - 20:
                break

            found = False
            # Multiple restarts with different initializations
            for restart in range(3):
                if time.time() > deadline - 15:
                    break

                if best_rank >= target_rank:
                    # Truncate from best known
                    U_init = best_U[:, :target_rank].copy()
                    V_init = best_V[:, :target_rank].copy()
                    W_init = best_W[:, :target_rank].copy()
                    # Add small random perturbation
                    U_init += np.random.randn(U_dim, target_rank) * 0.01
                    V_init += np.random.randn(V_dim, target_rank) * 0.01
                    W_init += np.random.randn(W_dim, target_rank) * 0.01
                else:
                    # Random initialization with structure
                    U_init = np.random.randn(U_dim, target_rank) * 0.3
                    V_init = np.random.randn(V_dim, target_rank) * 0.3
                    W_init = np.random.randn(W_dim, target_rank) * 0.3

                # Run optimization with more iterations
                U_opt, V_opt, W_opt = fast_optimize(
                    U_init, V_init, W_init,
                    target_rank,
                    max_iter=250,
                    lr=0.04
                )

                # Try snapping
                ok, U_s, V_s, W_s = try_snap_validate(U_opt, V_opt, W_opt)
                if ok:
                    best_U, best_V, best_W = U_s, V_s, W_s
                    best_rank = target_rank
                    found = True
                    break

            if found:
                break

    # Step 3: Final validation and return
    # Ensure we have a valid decomposition
    if not validate(best_U, best_V, best_W, atol=1e-6):
        # Last resort: schoolbook
        best_U, best_V, best_W = build_schoolbook()
        best_rank = 50

    # Convert to lists as required
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
