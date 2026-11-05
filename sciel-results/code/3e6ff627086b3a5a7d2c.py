import numpy as np
import time
import scipy.optimize

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin from 150s

    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n

    # Build the target tensor T
    T = np.zeros((size_u, size_v, size_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def check_decomposition(U, V, W, tol=1e-8):
        """Check if the decomposition exactly reconstructs T with half-integer entries."""
        R = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(R, T, atol=tol):
            return False
        # Try rounding to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        R_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(R_r, T, atol=1e-9):
            return True
        # Also try integer rounding
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        R_i = np.einsum('ar,br,cr->abc', U_i, V_i, W_i)
        if np.allclose(R_i, T, atol=1e-9):
            return True
        return False

    def polish_decomposition(U, V, W, max_iter=50):
        """High-precision polish pass after ALS convergence."""
        for it in range(max_iter):
            if time.time() > deadline:
                break
            # Update U
            M_vw = np.einsum('br,cr->bcr', V, W).reshape(size_v * size_w, -1)
            T_1 = T.reshape(size_u, size_v * size_w)
            U_new, _, _, _ = np.linalg.lstsq(M_vw.T, T_1.T, rcond=None)
            U = U_new.T

            # Update V
            M_ua = np.einsum('ar,cr->acr', U, W).reshape(size_u * size_w, -1)
            T_2 = T.transpose(1, 0, 2).reshape(size_v, size_u * size_w)
            V_new, _, _, _ = np.linalg.lstsq(M_ua.T, T_2.T, rcond=None)
            V = V_new.T

            # Update W
            M_uv = np.einsum('ar,br->abr', U, V).reshape(size_u * size_v, -1)
            T_3 = T.transpose(0, 2, 1).reshape(size_w, size_u * size_v)
            W_new, _, _, _ = np.linalg.lstsq(M_uv.T, T_3.T, rcond=None)
            W = W_new.T
        return U, V, W

    def find_decomposition(rank, max_attempts=50, max_iter=100):
        """Find a rank-`rank` decomposition using ALS with multiple initializations."""
        if time.time() > deadline:
            return None

        best_residual = float('inf')
        best_UVW = None

        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break

            # Different initialization strategies
            if attempt == 0:
                # Use Winograd-inspired structured initialization for rank 23
                if rank == 23:
                    U, V, W = winograd_init_23()
                else:
                    U = np.random.randn(size_u, rank) * 0.5
                    V = np.random.randn(size_v, rank) * 0.5
                    W = np.random.randn(size_w, rank) * 0.5
            elif attempt == 1:
                # Sparse random initialization
                U = np.random.randn(size_u, rank)
                V = np.random.randn(size_v, rank)
                W = np.random.randn(size_w, rank)
                # Zero out some entries to encourage sparsity
                mask_u = np.random.rand(size_u, rank) > 0.5
                mask_v = np.random.rand(size_v, rank) > 0.5
                mask_w = np.random.rand(size_w, rank) > 0.5
                U[mask_u] = 0
                V[mask_v] = 0
                W[mask_w] = 0
            elif attempt == 2:
                # Block-structured initialization
                U = np.zeros((size_u, rank))
                V = np.zeros((size_v, rank))
                W = np.zeros((size_w, rank))
                for r in range(rank):
                    # Assign random blocks
                    idx_u = np.random.randint(0, size_u, size=3)
                    idx_v = np.random.randint(0, size_v, size=3)
                    idx_w = np.random.randint(0, size_w, size=3)
                    U[idx_u, r] = np.random.randn(3)
                    V[idx_v, r] = np.random.randn(3)
                    W[idx_w, r] = np.random.randn(3)
            else:
                # Random with varying scales
                scale = 0.1 + 0.9 * (attempt % 5) / 5.0
                U = np.random.randn(size_u, rank) * scale
                V = np.random.randn(size_v, rank) * scale
                W = np.random.randn(size_w, rank) * scale

            # ALS iterations
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break

                # Update U
                M_vw = np.einsum('br,cr->bcr', V, W).reshape(size_v * size_w, rank)
                T_1 = T.reshape(size_u, size_v * size_w)
                U, _, _, _ = np.linalg.lstsq(M_vw.T, T_1.T, rcond=None)
                U = U.T

                # Update V
                M_ua = np.einsum('ar,cr->acr', U, W).reshape(size_u * size_w, rank)
                T_2 = T.transpose(1, 0, 2).reshape(size_v, size_u * size_w)
                V, _, _, _ = np.linalg.lstsq(M_ua.T, T_2.T, rcond=None)
                V = V.T

                # Update W
                M_uv = np.einsum('ar,br->abr', U, V).reshape(size_u * size_v, rank)
                T_3 = T.transpose(0, 2, 1).reshape(size_w, size_u * size_v)
                W, _, _, _ = np.linalg.lstsq(M_uv.T, T_3.T, rcond=None)
                W = W.T

                # Check residual
                R = np.einsum('ar,br,cr->abc', U, V, W)
                residual = np.linalg.norm(R - T) / np.linalg.norm(T)

                if residual < best_residual:
                    best_residual = residual
                    best_UVW = (U.copy(), V.copy(), W.copy())

                # Early exit if converged
                if residual < 1e-10:
                    break

            # Polish pass
            if best_UVW is not None:
                U_p, V_p, W_p = polish_decomposition(best_UVW[0], best_UVW[1], best_UVW[2], max_iter=20)
                if check_decomposition(U_p, V_p, W_p):
                    U_p = np.round(U_p * 2) / 2
                    V_p = np.round(V_p * 2) / 2
                    W_p = np.round(W_p * 2) / 2
                    # Final verification
                    R_final = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
                    if np.allclose(R_final, T, atol=1e-9):
                        return {
                            "U": U_p.tolist(),
                            "V": V_p.tolist(),
                            "W": W_p.tolist()
                        }

        return None

    def winograd_init_23():
        """Initialize using Winograd's algorithm structure for 3x3 matrix multiplication."""
        rank = 23
        U = np.zeros((size_u, rank))
        V = np.zeros((size_v, rank))
        W = np.zeros((size_w, rank))

        # Use a pattern based on the Winograd transform structure
        for r in range(rank):
            a_sel = [i for i in range(9) if (i * 7 + r) % 3 == 0]
            b_sel = [i for i in range(9) if (i * 5 + r) % 3 == 1]
            c_sel = [i for i in range(9) if (i * 11 + r) % 3 == 2]

            for a in a_sel:
                U[a, r] = 1.0 if (a + r) % 2 == 0 else -1.0
            for b in b_sel:
                V[b, r] = 1.0 if (b + r) % 2 == 0 else -1.0
            for c in c_sel:
                W[c, r] = 1.0 if (c + r) % 2 == 0 else -1.0

        # Add some random perturbation to break symmetry
        U += np.random.randn(size_u, rank) * 0.1
        V += np.random.randn(size_v, rank) * 0.1
        W += np.random.randn(size_w, rank) * 0.1

        return U, V, W

    best_result = None

    # Try ranks from 23 upward (23 is the known minimal rank for 3x3)
    for rank in [23, 24, 25, 26]:
        if time.time() > deadline - 15:
            break
        res = find_decomposition(rank, max_attempts=30, max_iter=80)
        if res is not None:
            best_result = res
            break

    if best_result is not None:
        return best_result

    # Fallback to schoolbook (rank 27)
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}
