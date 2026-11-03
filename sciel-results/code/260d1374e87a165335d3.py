import time
import random
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    deadline = time.time() + 135  # Safety margin before 150s kill

    # Strassen rank-7 components for 2x2*2x2 (local flat indices 0..3)
    strassen = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
    ]

    def emit_block(a_cs, b_rs, b_cs, c_cs, off, U, V, W, rank):
        for s in range(7):
            col = off + s
            if col >= rank:
                continue
            uc, vc, wc = strassen[s]
            for li, c in uc.items():
                r, cc = li // 2, li % 2
                U[r * m + a_cs + cc][col] += c
            for li, c in vc.items():
                r, cc = li // 2, li % 2
                V[(b_rs + r) * p + b_cs + cc][col] += c
            for li, c in wc.items():
                r, cc = li // 2, li % 2
                W[(c_cs + cc) * n + r][col] += c

    def build_strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        emit_block(0, 0, 0, 0, 0, U, V, W, rank)
        emit_block(0, 0, 2, 2, 7, U, V, W, rank)
        emit_block(2, 2, 0, 0, 14, U, V, W, rank)
        emit_block(2, 2, 2, 2, 21, U, V, W, rank)
        return U, V, W

    def verify_decomposition(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    def verify_np(U, V, W, n, m, p):
        Ua = np.asarray(U, dtype=np.float64)
        Va = np.asarray(V, dtype=np.float64)
        Wa = np.asarray(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def is_rank1_tensor(T, nm, mp, pn):
        T000 = None
        i0 = j0 = k0 = None
        for i in range(nm):
            if T000 is not None:
                break
            for j in range(mp):
                if T000 is not None:
                    break
                for k in range(pn):
                    if T[i, j, k] != 0:
                        T000 = T[i, j, k]
                        i0, j0, k0 = i, j, k
                        break
        if T000 is None:
            return True, (None, None, None)
        for i in range(nm):
            for j in range(mp):
                for k in range(pn):
                    if T[i, j, k] * T000 != T[i, j0, k] * T[i0, j, k0]:
                        return False, (None, None, None)
                    if T[i, j, k] * T000 != T[i, j0, k0] * T[i0, j, k]:
                        return False, (None, None, None)
        return True, (i0, j0, k0)

    def try_merge_two_terms(U, V, W, n, m, p, deadline):
        rank = len(U[0])
        nm, mp, pn = n * m, m * p, p * n
        U_cols = [np.array([U[i][c] for i in range(nm)], dtype=np.float64) for c in range(rank)]
        V_cols = [np.array([V[i][c] for i in range(mp)], dtype=np.float64) for c in range(rank)]
        W_cols = [np.array([W[i][c] for i in range(pn)], dtype=np.float64) for c in range(rank)]
        
        # Try all pairs
        for a in range(rank):
            if time.time() > deadline - 10:
                break
            ua, va, wa = U_cols[a], V_cols[a], W_cols[a]
            for b in range(a + 1, rank):
                ub, vb, wb = U_cols[b], V_cols[b], W_cols[b]
                T = (np.outer(np.outer(ua, va), wa) + np.outer(np.outer(ub, vb), wb)).reshape(nm, mp, pn)
                r1, (i0, j0, k0) = is_rank1_tensor(T, nm, mp, pn)
                if not r1 or i0 is None:
                    continue
                u_prime = T[:, j0, k0].copy()
                v_prime = T[i0, :, k0].copy()
                w_prime = T[i0, j0, :].copy()
                new_rank = rank - 1
                U_new = [[0] * new_rank for _ in range(nm)]
                V_new = [[0] * new_rank for _ in range(mp)]
                W_new = [[0] * new_rank for _ in range(pn)]
                col_map = [r for r in range(rank) if r != a and r != b]
                for new_c, old_c in enumerate(col_map):
                    for i in range(nm):
                        U_new[i][new_c] = U[i][old_c]
                    for i in range(mp):
                        V_new[i][new_c] = V[i][old_c]
                    for i in range(pn):
                        W_new[i][new_c] = W[i][old_c]
                lc = new_rank - 1
                for i in range(nm):
                    U_new[i][lc] = u_prime[i]
                for i in range(mp):
                    V_new[i][lc] = v_prime[i]
                for i in range(pn):
                    W_new[i][lc] = w_prime[i]
                if verify_np(U_new, V_new, W_new, n, m, p):
                    return U_new, V_new, W_new
        return None

    def try_merge_k_terms(U, V, W, n, m, p, k, deadline):
        """Try to merge k terms into one rank-1 term."""
        rank = len(U[0])
        nm, mp, pn = n * m, m * p, p * n
        U_cols = [np.array([U[i][c] for i in range(nm)], dtype=np.float64) for c in range(rank)]
        V_cols = [np.array([V[i][c] for i in range(mp)], dtype=np.float64) for c in range(rank)]
        W_cols = [np.array([W[i][c] for i in range(pn)], dtype=np.float64) for c in range(rank)]
        
        for combo in combinations(range(rank), k):
            if time.time() > deadline - 8:
                break
            T = np.zeros((nm, mp, pn), dtype=np.float64)
            for idx in combo:
                T += np.outer(np.outer(U_cols[idx], V_cols[idx]), W_cols[idx])
            r1, (i0, j0, k0) = is_rank1_tensor(T, nm, mp, pn)
            if not r1 or i0 is None:
                continue
            u_prime = T[:, j0, k0].copy()
            v_prime = T[i0, :, k0].copy()
            w_prime = T[i0, j0, :].copy()
            new_rank = rank - (k - 1)
            U_new = [[0] * new_rank for _ in range(nm)]
            V_new = [[0] * new_rank for _ in range(mp)]
            W_new = [[0] * new_rank for _ in range(pn)]
            col_map = [r for r in range(rank) if r not in combo]
            for new_c, old_c in enumerate(col_map):
                for i in range(nm):
                    U_new[i][new_c] = U[i][old_c]
                for i in range(mp):
                    V_new[i][new_c] = V[i][old_c]
                for i in range(pn):
                    W_new[i][new_c] = W[i][old_c]
            lc = new_rank - 1
            for i in range(nm):
                U_new[i][lc] = u_prime[i]
            for i in range(mp):
                V_new[i][lc] = v_prime[i]
            for i in range(pn):
                W_new[i][lc] = w_prime[i]
            if verify_np(U_new, V_new, W_new, n, m, p):
                return U_new, V_new, W_new
        return None

    def solve_linear_reduction(U, V, W, n, m, p, keep_indices, deadline):
        """
        Try to express the contribution of killed columns as a linear combination
        of the kept columns. This is a more general approach than border rank lifting.
        """
        rank = len(U[0])
        nm, mp, pn = n * m, m * p, p * n
        R = len(keep_indices)
        if R == 0:
            return None
        
        Ua = np.asarray(U, dtype=np.float64)
        Va = np.asarray(V, dtype=np.float64)
        Wa = np.asarray(W, dtype=np.float64)
        
        # For each (i,j,k), we need:
        # sum_{t in keep} alpha_t * U[i,j,keep_t] * V[j,k,keep_t] * W[k,i,keep_t] 
        #   + sum_{t in kill} U[i,j,kill_t] * V[j,k,kill_t] * W[k,i,kill_t] = 1
        #
        # This is nonlinear in alpha_t. Instead, we try a different approach:
        # Assume the killed columns can be absorbed by scaling the kept columns.
        # This is the border rank approach, but we make it more robust.
        
        # Construct M where M[(i,j,k), t] = U[i*m+j, keep[t]] * V[j*p+k, keep[t]] * W[k*n+i, keep[t]]
        N_eqs = n * m * p
        M = np.zeros((N_eqs, R), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for t in range(R):
                        col = keep_indices[t]
                        M[idx, t] = Ua[i * m + j, col] * Va[j * p + k, col] * Wa[k * n + i, col]
                    idx += 1
        
        y = np.ones(N_eqs, dtype=np.float64)
        
        try:
            beta, residuals, rank_M, sv = np.linalg.lstsq(M, y, rcond=None)
        except np.linalg.LinAlgError:
            return None
            
        if residuals is None or residuals[0] > 1e-8:
            return None
            
        # Compute alpha = cbrt(beta)
        alpha = np.zeros(R)
        for t in range(R):
            if beta[t] < 0:
                alpha[t] = -np.cbrt(-beta[t])
            else:
                alpha[t] = np.cbrt(beta[t])
                
        # Round alpha to nearest half-integer
        alpha_rounded = np.array([int(round(a * 2)) / 2.0 for a in alpha])
        
        # Check if rounded values still work
        U_new = [[0] * R for _ in range(nm)]
        V_new = [[0] * R for _ in range(mp)]
        W_new = [[0] * R for _ in range(pn)]
        
        for t in range(R):
            a_val = alpha_rounded[t]
            col = keep_indices[t]
            for i in range(nm):
                U_new[i][t] = a_val * Ua[i, col]
            for i in range(mp):
                V_new[i][t] = a_val * Va[i, col]
            for i in range(pn):
                W_new[i][t] = a_val * Wa[i, col]
                
        if verify_np(U_new, V_new, W_new, n, m, p):
            return U_new, V_new, W_new
            
        return None

    def try_lift(U, V, W, n, m, p, deadline, nkill, max_trials=50):
        rank = len(U[0])
        rng = random.Random(12345 + nkill * 1000)
        for _ in range(max_trials):
            if time.time() > deadline - 10:
                break
            kill = rng.sample(range(rank), nkill)
            keep = [c for c in range(rank) if c not in set(kill)]
            res = solve_linear_reduction(U, V, W, n, m, p, keep, deadline)
            if res is not None:
                return res
        return None

    U28, V28, W28 = build_strassen_28()
    if not verify_decomposition(U28, V28, W28, n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        return {"U": U, "V": V, "W": W}

    best_U, best_V, best_W = U28, V28, W28
    best_rank = 28

    # Phase 1: Try border rank lifting to reduce from 28
    if time.time() < deadline - 25:
        for nkill in [5, 6, 4, 7, 3]:
            if time.time() > deadline - 15:
                break
            lifted = try_lift(best_U, best_V, best_W, n, m, p, deadline, nkill, max_trials=80)
            if lifted is not None:
                best_U, best_V, best_W = lifted
                best_rank = len(best_U[0])
                break

    # Phase 2: Pairwise merge reduction
    merge_count = 0
    while time.time() < deadline - 15:
        merged = try_merge_two_terms(best_U, best_V, best_W, n, m, p, deadline)
        if merged is None:
            break
        best_U, best_V, best_W = merged
        best_rank = len(best_U[0])
        merge_count += 1
        if merge_count >= 10:
            break

    # Phase 3: Try 3-term merges
    if time.time() < deadline - 12:
        merged3 = try_merge_k_terms(best_U, best_V, best_W, n, m, p, 3, deadline)
        if merged3 is not None:
            best_U, best_V, best_W = merged3
            best_rank = len(best_U[0])

    # Phase 4: Another round of lifting from reduced base
    if time.time() < deadline - 10 and best_rank > 22:
        for nkill in [2, 3, 4]:
            if time.time() > deadline - 8:
                break
            lifted = try_lift(best_U, best_V, best_W, n, m, p, deadline, nkill, max_trials=50)
            if lifted is not None:
                best_U, best_V, best_W = lifted
                best_rank = len(best_U[0])
                break

    # Phase 5: Final merge attempts
    while time.time() < deadline - 5:
        merged = try_merge_two_terms(best_U, best_V, best_W, n, m, p, deadline)
        if merged is None:
            break
        best_U, best_V, best_W = merged
        best_rank = len(best_U[0])

    return {"U": best_U, "V": best_V, "W": best_W}
