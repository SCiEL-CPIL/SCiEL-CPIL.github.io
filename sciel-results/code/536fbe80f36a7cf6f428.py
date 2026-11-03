import time
import numpy as np
import scipy.optimize

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    def snap_to_half(x):
        return round(x * 2) / 2.0

    def snap_matrix(M):
        return np.array([[snap_to_half(x) for x in row] for row in M])

    def verify_decomp(U, V, W, rank, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        T_current = np.einsum('ar,br,cr->abc', U, V, W)
        T_target = np.zeros((9, 18, 18))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i * m + j, j * p + k, k * n + i] = 1.0
        if np.max(np.abs(T_current - T_target)) > tol:
            return False
        return True

    def find_decomposition(rank, max_time=30):
        start_time = time.time()
        rng = np.random.RandomState(rank * 1000 + 7)

        T_target = np.zeros((9, 18, 18))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i * m + j, j * p + k, k * n + i] = 1.0

        dim_u = n * m  # 9
        dim_v = m * p  # 18
        dim_w = p * n  # 18
        total_dim = dim_u + dim_v + dim_w

        def unpack(x):
            U = x[:dim_u * rank].reshape(dim_u, rank)
            V = x[dim_u * rank:(dim_u + dim_v) * rank].reshape(dim_v, rank)
            W = x[(dim_u + dim_v) * rank:].reshape(dim_w, rank)
            return U, V, W

        def objective(x):
            U, V, W = unpack(x)
            T_current = np.einsum('ur,vr,wr->uvw', U, V, W)
            Residual = T_current - T_target
            return np.sum(Residual ** 2)

        def gradient(x):
            U, V, W = unpack(x)
            T_current = np.einsum('ur,vr,wr->uvw', U, V, W)
            Residual = T_current - T_target
            grad_U = np.einsum('uvw,vr,wr->ur', Residual, V, W)
            grad_V = np.einsum('uvw,ur,wr->vr', Residual, U, W)
            grad_W = np.einsum('uvw,ur,vr->wr', Residual, U, V)
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

        best_res = None
        best_obj = float('inf')

        # Initialization strategies
        def random_init(scale=0.3):
            return rng.randn(total_dim * rank) * scale

        def sparse_init():
            x = rng.randn(total_dim * rank) * 0.5
            mask = rng.rand(total_dim * rank) > 0.6
            x[mask] = 0
            return x

        def block_init():
            # Try to initialize with structure that might help
            x = np.zeros(total_dim * rank)
            for r in range(rank):
                # Random but structured
                idx_u = rng.randint(0, dim_u)
                idx_v = rng.randint(0, dim_v)
                idx_w = rng.randint(0, dim_w)
                x[idx_u * rank + r] = 1.0
                x[(dim_u * rank) + idx_v * rank + r] = 1.0
                x[(dim_u + dim_v) * rank + idx_w * rank + r] = 1.0
            return x

        init_strategies = [
            lambda: random_init(0.1),
            lambda: random_init(0.3),
            lambda: random_init(0.5),
            sparse_init,
            block_init
        ]

        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() - start_time > max_time:
                break

            strategy_idx = attempt % len(init_strategies)
            x0 = init_strategies[strategy_idx]()

            if attempt > 0:
                x0 = x0 + rng.randn(total_dim * rank) * 0.01

            try:
                res = scipy.optimize.minimize(
                    objective, x0, method='L-BFGS-B', jac=gradient,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-9, 'maxls': 40}
                )
            except Exception:
                continue

            obj_val = objective(res.x)
            if obj_val < best_obj:
                best_obj = obj_val
                best_res = res.x.copy()

                # Check if snapping works
                U, V, W = unpack(best_res)
                U_snapped = snap_matrix(U)
                V_snapped = snap_matrix(V)
                W_snapped = snap_matrix(W)

                if verify_decomp(U_snapped, V_snapped, W_snapped, rank):
                    return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()

        # Final check
        if best_res is not None:
            U, V, W = unpack(best_res)
            U_snapped = snap_matrix(U)
            V_snapped = snap_matrix(V)
            W_snapped = snap_matrix(W)
            if verify_decomp(U_snapped, V_snapped, W_snapped, rank):
                return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()

        return None

    # Try ranks from 46 upwards (2*23 for two 3x3x3 blocks)
    # The trivial decomposition has rank 54
    # We try to find better decompositions
    for rank in [46, 48, 50, 52, 54]:
        if time.time() > deadline - 10:
            break
        max_time_for_rank = min(25, max(5, int(deadline - time.time() - 10)))
        decomp = find_decomposition(rank, max_time=max_time_for_rank)
        if decomp is not None:
            U, V, W = decomp
            return {"U": U, "V": V, "W": W}

    # Fallback: trivial decomposition with rank 54
    rank = 54
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(18)]
    W = [[0] * rank for _ in range(18)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
