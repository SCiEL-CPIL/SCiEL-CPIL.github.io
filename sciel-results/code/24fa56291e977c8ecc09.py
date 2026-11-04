import numpy as np
import time
import sys
import random
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    N_U = n * m
    N_V = m * p
    N_W = p * n

    start_time = time.time()
    deadline = 130.0  # 20s margin from 150s limit

    # Helper to create schoolbook solution
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(N_U)]
        V = [[0] * rank for _ in range(N_V)]
        W = [[0] * rank for _ in range(N_W)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    best_result = get_schoolbook()
    best_rank = n * m * p

    # Precompute target tensor
    T_target = np.zeros((N_U, N_V, N_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    T_flat = T_target.flatten()

    def build_block_2x3x3(R, U11, V11, W11):
        """Build full 2x3x7 solution from a 2x3x3 rank-R block."""
        R_total = R + R + 6
        U_full = np.zeros((N_U, R_total))
        V_full = np.zeros((N_V, R_total))
        W_full = np.zeros((N_W, R_total))

        # Block 1: Cols 0-2 of B, Cols 0-2 of C
        U_full[:, 0:R] = U11
        for j in range(3):
            for k in range(3):
                local_idx = j * 3 + k
                global_idx = j * 7 + k
                V_full[global_idx, 0:R] = V11[local_idx, :]
        W_full[:, 0:R] = W11

        # Block 2: Cols 3-5 of B, Cols 3-5 of C
        U_full[:, R:2*R] = U11
        for j in range(3):
            for k in range(3):
                local_idx = j * 3 + k
                global_idx = j * 7 + (k + 3)
                V_full[global_idx, R:2*R] = V11[local_idx, :]
        # W for C2: k in 3,4,5 -> rows (k*2+i) = 6..11
        W_full[6:12, R:2*R] = W11

        # Block 3: Col 6 of B, Col 6 of C
        U6, V6, W6 = get_2x3x1_rank6()
        U_full[:, 2*R:2*R+6] = U6
        for j in range(3):
            global_idx = j * 7 + 6
            V_full[global_idx, 2*R:2*R+6] = V6[j, :]
        W_full[12:14, 2*R:2*R+6] = W6

        # Verify
        T_calc = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
        if np.allclose(T_calc, T_target, atol=1e-8):
            return U_full, V_full, W_full
        return None

    def get_2x3x1_rank6():
        nn, mm, pp = 2, 3, 1
        NU, NV, NW = nn * mm, mm * pp, pp * nn
        R = 6
        U = np.zeros((NU, R))
        V = np.zeros((NV, R))
        W = np.zeros((NW, R))

        col = 0
        for j in range(mm):
            U[0 * mm + j, col] = 1
            V[j * pp + 0, col] = 1
            W[0 * nn + 0, col] = 1
            col += 1
            U[1 * mm + j, col] = 1
            V[j * pp + 0, col] = 1
            W[1 * nn + 0, col] = 1
            col += 1
        return U, V, W

    def find_2x3x3_rank(R, max_restarts=200):
        """Try to find rank-R decomposition for 2x3x3."""
        nn, mm, pp = 2, 3, 3
        NU, NV, NW = nn * mm, mm * pp, pp * nn

        T_sub = np.zeros((NU, NV, NW), dtype=np.float64)
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    T_sub[i * mm + j, j * pp + k, k * nn + i] = 1.0
        T_sub_flat = T_sub.flatten()

        len_U = NU * R
        len_V = NV * R
        len_W = NW * R
        total_vars = len_U + len_V + len_W

        def residual(params):
            U = params[:len_U].reshape(NU, R)
            V = params[len_U:len_U+len_V].reshape(NV, R)
            W = params[len_U+len_V:].reshape(NW, R)
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            return (T_calc.flatten() - T_sub_flat)

        best_sol = None
        best_cost = np.inf

        # Try structured initializations first
        # Strategy 1: Random
        for restart in range(max_restarts):
            if time.time() > deadline - 15:
                break
            if restart % 50 == 0:
                random.seed(None)
            np.random.seed(restart * 7919 + 13)

            x0 = np.random.uniform(-1, 1, total_vars)
            try:
                sol = least_squares(residual, x0, method='lm', max_nfev=2000)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_sol = sol.x
            except Exception:
                continue

        # Strategy 2: Try with some structure - initialize U with identity-like patterns
        if best_cost > 1e-6:
            for restart in range(min(100, max_restarts)):
                if time.time() > deadline - 15:
                    break
                U_init = np.zeros((NU, R))
                V_init = np.zeros((NV, R))
                W_init = np.zeros((NW, R))

                # Random assignment of non-zero entries
                for r in range(R):
                    # Pick random positions
                    u_idx = np.random.choice(NU, size=1, replace=False)
                    v_idx = np.random.choice(NV, size=1, replace=False)
                    w_idx = np.random.choice(NW, size=1, replace=False)
                    U_init[u_idx[0], r] = np.random.choice([-1, 1])
                    V_init[v_idx[0], r] = np.random.choice([-1, 1])
                    W_init[w_idx[0], r] = np.random.choice([-1, 1])

                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                try:
                    sol = least_squares(residual, x0, method='lm', max_nfev=2000)
                    if sol.cost < best_cost:
                        best_cost = sol.cost
                        best_sol = sol.x
                except Exception:
                    continue

        if best_sol is None or best_cost > 1e-8:
            return None

        # Round to half-integers
        x_rounded = np.round(best_sol * 2) / 2.0
        res_rounded = residual(x_rounded)
        if np.linalg.norm(res_rounded) < 1e-8:
            U = x_rounded[:len_U].reshape(NU, R)
            V = x_rounded[len_U:len_U+len_V].reshape(NV, R)
            W = x_rounded[len_U+len_V:].reshape(NW, R)
            return U, V, W

        # Try rounding to integers
        x_int = np.round(best_sol)
        res_int = residual(x_int)
        if np.linalg.norm(res_int) < 1e-8:
            U = x_int[:len_U].reshape(NU, R)
            V = x_int[len_U:len_U+len_V].reshape(NV, R)
            W = x_int[len_U+len_V:].reshape(NW, R)
            return U, V, W

        # If not exactly on half-integers but close, try a final polish
        # Use the found solution as starting point with tighter tolerance
        try:
            sol2 = least_squares(residual, best_sol, method='lm', max_nfev=5000, xtol=1e-14)
            if sol2.cost < 1e-10:
                x2 = sol2.x
                x_rounded2 = np.round(x2 * 2) / 2.0
                res_r2 = residual(x_rounded2)
                if np.linalg.norm(res_r2) < 1e-8:
                    U = x_rounded2[:len_U].reshape(NU, R)
                    V = x_rounded2[len_U:len_U+len_V].reshape(NV, R)
                    W = x_rounded2[len_U+len_V:].reshape(NW, R)
                    return U, V, W
        except Exception:
            pass

        return None

    # Try to find 2x3x3 decompositions for various ranks
    # Start with rank 11 (theoretical minimum is ceil(42/4)=11? No, that's for 4 blocks)
    # For 2x3x3, schoolbook is 18. Let's try 12, 13, 14, 15.
    for R in [12, 13, 14, 15, 11, 16]:
        if time.time() > deadline - 10:
            break
        print(f"Trying 2x3x3 rank {R}...", file=sys.stderr)
        res = find_2x3x3_rank(R)
        if res is not None:
            U11, V11, W11 = res
            full = build_block_2x3x3(R, U11, V11, W11)
            if full is not None:
                U_full, V_full, W_full = full
                rank_total = R + R + 6
                if rank_total < best_rank:
                    best_rank = rank_total
                    best_result = {
                        "U": U_full.tolist(),
                        "V": V_full.tolist(),
                        "W": W_full.tolist()
                    }
                    print(f"Found rank {rank_total} solution!", file=sys.stderr)
                break  # Found a solution, stop trying higher ranks for this block structure

    # Also try direct full-rank optimization for the entire 2x3x7 tensor
    # This is more expensive but might find a better non-block solution
    def try_full_direct(rank, max_restarts=20):
        """Try to find a rank-`rank` decomposition for the full 2x3x7 tensor."""
        len_U = N_U * rank
        len_V = N_V * rank
        len_W = N_W * rank
        total_vars = len_U + len_V + len_W

        def residual(params):
            U = params[:len_U].reshape(N_U, rank)
            V = params[len_U:len_U+len_V].reshape(N_V, rank)
            W = params[len_U+len_V:].reshape(N_W, rank)
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            return (T_calc.flatten() - T_flat)

        best_sol = None
        best_cost = np.inf

        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            np.random.seed(restart * 104729 + 42)
            x0 = np.random.uniform(-0.5, 0.5, total_vars)
            try:
                sol = least_squares(residual, x0, method='lm', max_nfev=1000)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_sol = sol.x
            except Exception:
                continue

        if best_sol is None or best_cost > 1e-8:
            return None

        x_rounded = np.round(best_sol * 2) / 2.0
        res_rounded = residual(x_rounded)
        if np.linalg.norm(res_rounded) < 1e-8:
            U = x_rounded[:len_U].reshape(N_U, rank)
            V = x_rounded[len_U:len_U+len_V].reshape(N_V, rank)
            W = x_rounded[len_U+len_V:].reshape(N_W, rank)
            return U, V, W

        return None

    # Try a few direct ranks that might be better than block decomposition
    # Block gives 2*R+6. If R=12, that's 30. Direct might find 28 or 29.
    for direct_rank in [28, 29, 30, 31, 32]:
        if time.time() > deadline - 3:
            break
        if direct_rank >= best_rank:
            continue
        print(f"Trying direct rank {direct_rank}...", file=sys.stderr)
        res = try_full_direct(direct_rank)
        if res is not None:
            U, V, W = res
            if direct_rank < best_rank:
                best_rank = direct_rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                print(f"Found direct rank {direct_rank} solution!", file=sys.stderr)
                break

    return best_result
# EVOLVE-BLOCK-END
