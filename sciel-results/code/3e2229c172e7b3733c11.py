import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0

    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        return T

    T = build_tensor()

    def verify(U, V, W, tol=1e-8):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T_recon - T)) < tol

    def build_strassen_2x2x2():
        n_s, m_s, p_s = 2, 2, 2
        rank = 7
        U = np.zeros((n_s * m_s, rank))
        V = np.zeros((m_s * p_s, rank))
        W = np.zeros((p_s * n_s, rank))
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        # M2 = (A10+A11) B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        # M3 = A00 (B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        # M4 = A11 (B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        # M5 = (A00+A01) B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        return U, V, W

    def build_2x2x1():
        n_s, m_s, p_s = 2, 2, 1
        rank = 2
        U = np.zeros((n_s * m_s, rank))
        V = np.zeros((m_s * p_s, rank))
        W = np.zeros((p_s * n_s, rank))
        U[0, 0] = 1; U[2, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1
        U[1, 1] = 1; U[3, 1] = 1
        V[1, 1] = 1
        W[0, 1] = 1; W[1, 1] = 1
        return U, V, W

    def build_2x1x7():
        n_s, m_s, p_s = 2, 1, 7
        rank = 1
        U = np.zeros((n_s * m_s, rank))
        V = np.zeros((m_s * p_s, rank))
        W = np.zeros((p_s * n_s, rank))
        U[:, 0] = 1.0
        V[:, 0] = 1.0
        W[:, 0] = 1.0
        return U, V, W

    # Build rank-24 hybrid: (2,2,7) via 3x Strassen + 1x (2,2,1), plus (2,1,7)
    U_s, V_s, W_s = build_strassen_2x2x2()
    U_221, V_221, W_221 = build_2x2x1()
    U_217, V_217, W_217 = build_2x1x7()

    n_sub, m_sub, p_sub = 2, 2, 7
    p_parts = [2, 2, 2, 1]
    decomp_parts = [(U_s, V_s, W_s), (U_s, V_s, W_s), (U_s, V_s, W_s), (U_221, V_221, W_221)]

    rank_227 = sum(d[0].shape[1] for d in decomp_parts)
    U_227 = np.zeros((n_sub * m_sub, rank_227))
    V_227 = np.zeros((m_sub * p_sub, rank_227))
    W_227 = np.zeros((p_sub * n_sub, rank_227))

    col_off = 0
    p_off = 0
    for pi, (U_p, V_p, W_p) in enumerate(decomp_parts):
        r_p = U_p.shape[1]
        p_len = p_parts[pi]
        U_227[:, col_off:col_off + r_p] = U_p
        for j in range(m_sub):
            for k in range(p_len):
                V_227[j * p_sub + (p_off + k), col_off:col_off + r_p] = V_p[j * p_len + k, :]
        for k in range(p_len):
            for i in range(n_sub):
                W_227[(p_off + k) * n_sub + i, col_off:col_off + r_p] = W_p[k * n_sub + i, :]
        col_off += r_p
        p_off += p_len

    rank_total = rank_227 + U_217.shape[1]
    U_final = np.zeros((2 * 3, rank_total))
    V_final = np.zeros((3 * 7, rank_total))
    W_final = np.zeros((7 * 2, rank_total))

    r1 = rank_227
    for i in range(2):
        for j in range(2):
            U_final[i * 3 + j, :r1] = U_227[i * 2 + j, :]
    for j in range(2):
        for k in range(7):
            V_final[j * 7 + k, :r1] = V_227[j * 7 + k, :]
    for k in range(7):
        for i in range(2):
            W_final[k * 2 + i, :r1] = W_227[k * 2 + i, :]

    r2 = U_217.shape[1]
    for i in range(2):
        for j in range(1):
            U_final[i * 3 + (2 + j), r1:r1 + r2] = U_217[i * 1 + j, :]
    for j in range(1):
        for k in range(7):
            V_final[(2 + j) * 7 + k, r1:r1 + r2] = V_217[j * 7 + k, :]
    for k in range(7):
        for i in range(2):
            W_final[k * 2 + i, r1:r1 + r2] = W_217[k * 2 + i, :]

    best_rank = rank_total
    best_U, best_V, best_W = U_final.copy(), V_final.copy(), W_final.copy()

    if not verify(best_U, best_V, best_W):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, c] = 1.0
                    V_sb[j * p + k, c] = 1.0
                    W_sb[k * n + i, c] = 1.0
                    c += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Direct combinatorial search: try to reduce rank from 24 to 23
    # Use simulated annealing on the residual, with entries constrained to a small set
    candidate_values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]

    def residual(U, V, W):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_recon - T) ** 2)

    def local_search(U, V, W, current_rank, target_rank, max_iter=5000):
        """Try to reduce rank by removing one column and adjusting remaining columns."""
        if current_rank <= target_rank:
            return U, V, W, current_rank

        # Try removing each column and solving for the rest
        for remove_col in range(current_rank):
            if time.time() > deadline - 5:
                break

            # Keep all columns except remove_col
            keep = [c for c in range(current_rank) if c != remove_col]
            U_keep = U[:, keep]
            V_keep = V[:, keep]
            W_keep = W[:, keep]

            # Now we have rank-1 decomposition. Try to adjust entries to fit T.
            # Use a simple gradient-free approach: randomly perturb entries
            best_res = residual(U_keep, V_keep, W_keep)
            best_Uk, best_Vk, best_Wk = U_keep.copy(), V_keep.copy(), W_keep.copy()

            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break

                # Pick a random matrix and entry to perturb
                mat_choice = np.random.randint(3)
                if mat_choice == 0:
                    row = np.random.randint(U_keep.shape[0])
                    col = np.random.randint(U_keep.shape[1])
                    old_val = U_keep[row, col]
                    U_keep[row, col] = np.random.choice(candidate_values)
                    new_res = residual(U_keep, V_keep, W_keep)
                    if new_res < best_res:
                        best_res = new_res
                        best_Uk, best_Vk, best_Wk = U_keep.copy(), V_keep.copy(), W_keep.copy()
                    else:
                        U_keep[row, col] = old_val

                elif mat_choice == 1:
                    row = np.random.randint(V_keep.shape[0])
                    col = np.random.randint(V_keep.shape[1])
                    old_val = V_keep[row, col]
                    V_keep[row, col] = np.random.choice(candidate_values)
                    new_res = residual(U_keep, V_keep, W_keep)
                    if new_res < best_res:
                        best_res = new_res
                        best_Uk, best_Vk, best_Wk = U_keep.copy(), V_keep.copy(), W_keep.copy()
                    else:
                        V_keep[row, col] = old_val

                else:
                    row = np.random.randint(W_keep.shape[0])
                    col = np.random.randint(W_keep.shape[1])
                    old_val = W_keep[row, col]
                    W_keep[row, col] = np.random.choice(candidate_values)
                    new_res = residual(U_keep, V_keep, W_keep)
                    if new_res < best_res:
                        best_res = new_res
                        best_Uk, best_Vk, best_Wk = U_keep.copy(), V_keep.copy(), W_keep.copy()
                    else:
                        W_keep[row, col] = old_val

            if best_res < 1e-8:
                return best_Uk, best_Vk, best_Wk, target_rank

        return U, V, W, current_rank

    # Try to reduce from 24 to 23
    U_try, V_try, W_try, new_rank = local_search(best_U, best_V, best_W, best_rank, best_rank - 1, max_iter=2000)
    if new_rank < best_rank and verify(U_try, V_try, W_try):
        best_rank = new_rank
        best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()

    # If still at 24, try a different approach: random restart with rank 23
    if best_rank == 24 and time.time() < deadline - 20:
        for trial in range(50):
            if time.time() > deadline - 10:
                break

            target_rank = 23
            U_rand = np.random.choice(candidate_values, size=(n * m, target_rank))
            V_rand = np.random.choice(candidate_values, size=(m * p, target_rank))
            W_rand = np.random.choice(candidate_values, size=(p * n, target_rank))

            # Local search to minimize residual
            best_res = residual(U_rand, V_rand, W_rand)
            best_Ur, best_Vr, best_Wr = U_rand.copy(), V_rand.copy(), W_rand.copy()

            for it in range(3000):
                if time.time() > deadline - 10:
                    break

                mat_choice = np.random.randint(3)
                if mat_choice == 0:
                    row = np.random.randint(U_rand.shape[0])
                    col = np.random.randint(U_rand.shape[1])
                    old_val = U_rand[row, col]
                    U_rand[row, col] = np.random.choice(candidate_values)
                    new_res = residual(U_rand, V_rand, W_rand)
                    if new_res < best_res:
                        best_res = new_res
                        best_Ur, best_Vr, best_Wr = U_rand.copy(), V_rand.copy(), W_rand.copy()
                    else:
                        U_rand[row, col] = old_val
                elif mat_choice == 1:
                    row = np.random.randint(V_rand.shape[0])
                    col = np.random.randint(V_rand.shape[1])
                    old_val = V_rand[row, col]
                    V_rand[row, col] = np.random.choice(candidate_values)
                    new_res = residual(U_rand, V_rand, W_rand)
                    if new_res < best_res:
                        best_res = new_res
                        best_Ur, best_Vr, best_Wr = U_rand.copy(), V_rand.copy(), W_rand.copy()
                    else:
                        V_rand[row, col] = old_val
                else:
                    row = np.random.randint(W_rand.shape[0])
                    col = np.random.randint(W_rand.shape[1])
                    old_val = W_rand[row, col]
                    W_rand[row, col] = np.random.choice(candidate_values)
                    new_res = residual(U_rand, V_rand, W_rand)
                    if new_res < best_res:
                        best_res = new_res
                        best_Ur, best_Vr, best_Wr = U_rand.copy(), V_rand.copy(), W_rand.copy()
                    else:
                        W_rand[row, col] = old_val

            if best_res < 1e-8 and verify(best_Ur, best_Vr, best_Wr):
                best_rank = target_rank
                best_U, best_V, best_W = best_Ur.copy(), best_Vr.copy(), best_Wr.copy()
                break

    # Final verification
    if not verify(best_U, best_V, best_W):
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, c] = 1.0
                    V_sb[j * p + k, c] = 1.0
                    W_sb[k * n + i, c] = 1.0
                    c += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
