import numpy as np
import time
from itertools import combinations

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    Uses Strassen rank-21 as baseline, then attempts rank-20 via structured search.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12

    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Build rank-21 Strassen baseline
    rank21 = _build_strassen_21(nm, mp, pn)
    U21, V21, W21 = rank21
    U21 = np.array(U21, dtype=float)
    V21 = np.array(V21, dtype=float)
    W21 = np.array(W21, dtype=float)

    # Verify rank-21
    T_check = np.einsum('ar,br,cr->abc', U21, V21, W21)
    if not np.allclose(T_check, T, atol=1e-8):
        # Fallback to schoolbook rank-24
        U, V, W = _build_schoolbook_24(nm, mp, pn)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find rank-20
    result = _try_rank_20(T, nm, mp, pn, U21, V21, W21, deadline)
    if result is not None:
        U, V, W = result
        # Final verification
        T_check = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_check, T, atol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank-21
    return {"U": U21.tolist(), "V": V21.tolist(), "W": W21.tolist()}

def _build_strassen_21(nm, mp, pn):
    """Build the rank-21 Strassen-based decomposition."""
    rank = 21
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]

    def map_a(local_idx):
        return local_idx

    def map_b(local_idx, g):
        k0 = 2 * g
        k1 = 2 * g + 1
        if local_idx == 0:
            return 0 * 6 + k0
        elif local_idx == 1:
            return 0 * 6 + k1
        elif local_idx == 2:
            return 1 * 6 + k0
        else:
            return 1 * 6 + k1

    def map_c(local_idx, g):
        k0 = 2 * g
        k1 = 2 * g + 1
        if local_idx == 0:
            return k0 * 2 + 0
        elif local_idx == 1:
            return k1 * 2 + 0
        elif local_idx == 2:
            return k0 * 2 + 1
        else:
            return k1 * 2 + 1

    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            for local_idx in range(4):
                u_val = step['u'][local_idx]
                if u_val != 0:
                    U[map_a(local_idx), r] = u_val

                v_val = step['v'][local_idx]
                if v_val != 0:
                    V[map_b(local_idx, g), r] = v_val

                w_val = step['w'][local_idx]
                if w_val != 0:
                    W[map_c(local_idx, g), r] = w_val

    return U, V, W

def _build_schoolbook_24(nm, mp, pn):
    """Build the schoolbook rank-24 decomposition."""
    rank = 24
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    r = 0
    for i in range(2):
        for j in range(2):
            for k in range(6):
                a = i * 2 + j
                b = j * 6 + k
                c = k * 2 + i
                U[a, r] = 1.0
                V[b, r] = 1.0
                W[c, r] = 1.0
                r += 1
    return U, V, W

def _try_rank_20(T, nm, mp, pn, U21, V21, W21, deadline):
    """
    Try to find a rank-20 decomposition by dropping one term from rank-21
    and optimizing the remaining 20 terms.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None

    best_loss = np.inf
    best_params = None

    # Strategy 1: Drop each of the 21 terms and optimize the remaining 20
    for drop_idx in range(21):
        if time.time() > deadline - 20:
            break

        # Build initial rank-20 by dropping term drop_idx
        U_init = np.zeros((nm, 20))
        V_init = np.zeros((mp, 20))
        W_init = np.zeros((pn, 20))

        idx = 0
        for r in range(21):
            if r == drop_idx:
                continue
            U_init[:, idx] = U21[:, r]
            V_init[:, idx] = V21[:, r]
            W_init[:, idx] = W21[:, r]
            idx += 1

        params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

        def objective_and_grad(params, T, nm, mp, pn, rank):
            U = params[:nm*rank].reshape(nm, rank)
            V = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            W = params[nm*rank+mp*rank:].reshape(pn, rank)

            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            E = T_approx - T
            loss = 0.5 * np.sum(E**2)

            grad_U = -np.einsum('abc,br,cr->ar', E, V, W)
            grad_V = -np.einsum('abc,ar,cr->br', E, U, W)
            grad_W = -np.einsum('abc,ar,br->cr', E, U, V)

            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return loss, grad

        try:
            res = minimize(
                objective_and_grad,
                params0,
                args=(T, nm, mp, pn, 20),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50}
            )

            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x.copy()

                # Check if we found a valid solution
                if res.fun < 1e-10:
                    U = res.x[:nm*20].reshape(nm, 20)
                    V = res.x[nm*20:nm*20+mp*20].reshape(mp, 20)
                    W = res.x[nm*20+mp*20:].reshape(pn, 20)
                    T_check = np.einsum('ar,br,cr->abc', U, V, W)
                    if np.allclose(T_check, T, atol=1e-8):
                        return U, V, W
        except Exception:
            continue

    # Strategy 2: Try merging pairs of terms
    if time.time() < deadline - 30:
        for i, j in combinations(range(21), 2):
            if time.time() > deadline - 15:
                break

            # Build rank-20 by merging terms i and j
            U_init = np.zeros((nm, 20))
            V_init = np.zeros((mp, 20))
            W_init = np.zeros((pn, 20))

            merged_idx = 0
            for r in range(21):
                if r == i or r == j:
                    continue
                U_init[:, merged_idx] = U21[:, r]
                V_init[:, merged_idx] = V21[:, r]
                W_init[:, merged_idx] = W21[:, r]
                merged_idx += 1

            # Initialize merged term as weighted average
            w_i, w_j = 0.5, 0.5
            U_init[:, 19] = w_i * U21[:, i] + w_j * U21[:, j]
            V_init[:, 19] = w_i * V21[:, i] + w_j * V21[:, j]
            W_init[:, 19] = w_i * W21[:, i] + w_j * W21[:, j]

            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

            try:
                res = minimize(
                    objective_and_grad,
                    params0,
                    args=(T, nm, mp, pn, 20),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50}
                )

                if res.fun < best_loss:
                    best_loss = res.fun
                    best_params = res.x.copy()

                    if res.fun < 1e-10:
                        U = res.x[:nm*20].reshape(nm, 20)
                        V = res.x[nm*20:nm*20+mp*20].reshape(mp, 20)
                        W = res.x[nm*20+mp*20:].reshape(pn, 20)
                        T_check = np.einsum('ar,br,cr->abc', U, V, W)
                        if np.allclose(T_check, T, atol=1e-8):
                            return U, V, W
            except Exception:
                continue

    # If we found a good but not perfect solution, try a final polish
    if best_params is not None and best_loss < 1e-6:
        def objective_and_grad(params, T, nm, mp, pn, rank):
            U = params[:nm*rank].reshape(nm, rank)
            V = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            W = params[nm*rank+mp*rank:].reshape(pn, rank)

            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            E = T_approx - T
            loss = 0.5 * np.sum(E**2)

            grad_U = -np.einsum('abc,br,cr->ar', E, V, W)
            grad_V = -np.einsum('abc,ar,cr->br', E, U, W)
            grad_W = -np.einsum('abc,ar,br->cr', E, U, V)

            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return loss, grad

        try:
            res = minimize(
                objective_and_grad,
                best_params,
                args=(T, nm, mp, pn, 20),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 100}
            )

            if res.fun < 1e-10:
                U = res.x[:nm*20].reshape(nm, 20)
                V = res.x[nm*20:nm*20+mp*20].reshape(mp, 20)
                W = res.x[nm*20+mp*20:].reshape(pn, 20)
                T_check = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(T_check, T, atol=1e-8):
                    return U, V, W
        except Exception:
            pass

    return None
