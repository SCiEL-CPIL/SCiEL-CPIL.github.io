import numpy as np
import scipy
import time

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s limit

    n, m, p = 3, 3, 6
    dim_U = n * m   # 9
    dim_V = m * p   # 18
    dim_W = p * n   # 18

    # Build target tensor T_target[a, b, c]
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def compute_residual(U, V, W):
        """Compute R[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]"""
        return np.einsum('ar,br,cr->abc', U, V, W)

    def snap_and_verify(U, V, W):
        """Snap to nearest 0.5 and verify exactness."""
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        R = compute_residual(U_s, V_s, W_s)
        if np.allclose(R, T_target, atol=1e-9):
            return True, U_s, V_s, W_s
        # Also try integer rounding
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        R_i = compute_residual(U_i, V_i, W_i)
        if np.allclose(R_i, T_target, atol=1e-9):
            return True, U_i, V_i, W_i
        return False, None, None, None

    def get_schoolbook(rank=None):
        """Return schoolbook decomposition."""
        if rank is None:
            rank = n * m * p
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def make_structured_init(rank, seed):
        """Create structured initialization based on schoolbook with random perturbations.
        
        Strategy: Start from a rank-54 schoolbook-like structure, then randomly
        select 'rank' columns (with possible overlaps) and add small noise.
        This places the optimizer in a basin closer to valid solutions.
        """
        rng = np.random.RandomState(seed)
        
        # Build the full set of (i,j,k) triples
        triples = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    triples.append((i, j, k))
        
        # Randomly sample 'rank' triples (with replacement to allow structure)
        # Bias toward covering all indices
        selected = []
        for _ in range(rank):
            idx = rng.randint(0, len(triples))
            selected.append(triples[idx])
        
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        
        for col, (i, j, k) in enumerate(selected):
            U[i * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
        
        # Add small random perturbation to all entries to break symmetry
        U += rng.randn(dim_U, rank) * 0.3
        V += rng.randn(dim_V, rank) * 0.3
        W += rng.randn(dim_W, rank) * 0.3
        
        return U, V, W

    def make_random_init(rank, seed, scale=0.5):
        """Pure random initialization."""
        rng = np.random.RandomState(seed)
        U = rng.randn(dim_U, rank) * scale
        V = rng.randn(dim_V, rank) * scale
        W = rng.randn(dim_W, rank) * scale
        return U, V, W

    def optimize(U_init, V_init, W_init, rank, max_iter=3000):
        """Run L-BFGS-B optimization on the concatenated parameters."""
        params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def obj(params):
            U_p = params[:dim_U * rank].reshape(dim_U, rank)
            V_p = params[dim_U * rank: dim_U * rank + dim_V * rank].reshape(dim_V, rank)
            W_p = params[dim_U * rank + dim_V * rank:].reshape(dim_W, rank)
            R = compute_residual(U_p, V_p, W_p)
            return np.sum((R - T_target) ** 2)
        
        try:
            res = scipy.optimize.minimize(
                obj, params, method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            U_opt = res.x[:dim_U * rank].reshape(dim_U, rank)
            V_opt = res.x[dim_U * rank: dim_U * rank + dim_V * rank].reshape(dim_V, rank)
            W_opt = res.x[dim_U * rank + dim_V * rank:].reshape(dim_W, rank)
            return U_opt, V_opt, W_opt, res.fun
        except Exception:
            return U_init, V_init, W_init, float('inf')

    def polish(U, V, W, rank, max_iter=5000):
        """High-precision polish pass with tighter tolerances."""
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def obj(params):
            U_p = params[:dim_U * rank].reshape(dim_U, rank)
            V_p = params[dim_U * rank: dim_U * rank + dim_V * rank].reshape(dim_V, rank)
            W_p = params[dim_U * rank + dim_V * rank:].reshape(dim_W, rank)
            R = compute_residual(U_p, V_p, W_p)
            return np.sum((R - T_target) ** 2)
        
        try:
            res = scipy.optimize.minimize(
                obj, params, method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            U_opt = res.x[:dim_U * rank].reshape(dim_U, rank)
            V_opt = res.x[dim_U * rank: dim_U * rank + dim_V * rank].reshape(dim_V, rank)
            W_opt = res.x[dim_U * rank + dim_V * rank:].reshape(dim_W, rank)
            return U_opt, V_opt, W_opt
        except Exception:
            return U, V, W

    # Try ranks from 46 up to 53
    for rank in range(46, 54):
        if time.time() > deadline - 40:
            break

        # Number of attempts: more for lower ranks
        num_attempts = 15 if rank <= 47 else 10

        for attempt in range(num_attempts):
            if time.time() > deadline - 15:
                break

            seed = 1000 + rank * 100 + attempt * 7

            # Alternate between structured and random init
            if attempt % 2 == 0:
                U_init, V_init, W_init = make_structured_init(rank, seed)
            else:
                U_init, V_init, W_init = make_random_init(rank, seed)

            # First optimization pass
            U_opt, V_opt, W_opt, final_loss = optimize(U_init, V_init, W_init, rank, max_iter=2000)

            # If loss is small enough, try snapping
            if final_loss < 1e-6:
                # Polish pass for higher precision
                U_pol, V_pol, W_pol = polish(U_opt, V_opt, W_opt, rank, max_iter=3000)
                success, U_s, V_s, W_s = snap_and_verify(U_pol, V_pol, W_pol)
                if success:
                    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

            # Also try snapping the unpolished result
            if final_loss < 1e-4:
                success, U_s, V_s, W_s = snap_and_verify(U_opt, V_opt, W_opt)
                if success:
                    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    # Fallback: schoolbook rank 54
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
