"""Matrix multiplication 3,3,7 tensor decomposition."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 135  # safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, trials=20, tol=1e-6):
    """Check that the decomposition computes C[i,k] = sum_j A[i,j] B[j,k]."""
    rank = U.shape[1]
    rng = np.random.RandomState(0)
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        C = np.zeros((n, p))
        for t in range(rank):
            a = U[:, t].reshape(n, m)
            b = V[:, t].reshape(m, p)
            w = W[:, t].reshape(p, n)
            C += np.einsum('ij,jk,ki->ik', a, b, w)
        if np.max(np.abs(C - C_expected)) > tol:
            return False
    return True

def _solve_group(n, m, p, target_rank, seed, time_budget):
    """
    Solve for a rank-`target_rank` decomposition of the n x m by m x p
    multiplication tensor via numerical optimization.
    Returns (U, V, W) numpy arrays, or None on failure.
    """
    deadline = time.time() + time_budget
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    R = target_rank

    # Index sets for the tensor T[i,j,k] = 1 (all of them)
    idx_i = np.arange(n)
    idx_j = np.arange(m)
    idx_k = np.arange(p)

    # Number of scalar constraints = n*m*p
    n_constraints = n * m * p

    # Random initialization with small noise around a sparse pattern
    def random_init(s):
        rng = np.random.RandomState(s)
        # Initialize each term as a sparse "basis" pattern with random signs
        U = np.zeros((dim_u, R))
        V = np.zeros((dim_v, R))
        W = np.zeros((dim_w, R))
        for t in range(R):
            # pick a random (i,j) for U, (j',k') for V, (k'',i'') for W
            ii = rng.randint(n); jj = rng.randint(m)
            j2 = rng.randint(m); k2 = rng.randint(p)
            k3 = rng.randint(p); i3 = rng.randint(n)
            U[ii * m + jj, t] = rng.choice([1.0, -1.0])
            V[j2 * p + k2, t] = rng.choice([1.0, -1.0])
            W[k3 * n + i3, t] = rng.choice([1.0, -1.0])
        # add small noise to make it non-degenerate
        U += 0.01 * rng.randn(dim_u, R)
        V += 0.01 * rng.randn(dim_v, R)
        W += 0.01 * rng.randn(dim_w, R)
        return U, V, W

    def residuals(U, V, W):
        # T[i,j,k] = sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] - 1
        # Build the tensor from the decomposition
        T = np.einsum('it,jt,kt->ijk',
                      U.reshape(n, m, R),
                      V.reshape(m, p, R),
                      W.reshape(p, n, R))
        return (T - 1.0).flatten()

    def loss(U, V, W):
        r = residuals(U, V, W)
        return float(np.sum(r * r))

    best = None
    best_loss = float('inf')

    n_seeds = 8
    for s in range(n_seeds):
        if time.time() > deadline:
            break
        U, V, W = random_init(seed + s)
        cur_loss = loss(U, V, W)

        # Gradient-free optimization: L-BFGS via scipy
        try:
            from scipy.optimize import minimize
        except Exception:
            # fallback: simple random restarts only
            if cur_loss < best_loss:
                best_loss = cur_loss
                best = (U.copy(), V.copy(), W.copy())
            continue

        def objective(x):
            Ux = x[:dim_u * R].reshape(dim_u, R)
            Vx = x[dim_u * R:dim_u * R + dim_v * R].reshape(dim_v, R)
            Wx = x[dim_u * R + dim_v * R:].reshape(dim_w, R)
            return loss(Ux, Vx, Wx)

        def jac(x):
            # Numerical gradient (central differences) -- expensive but robust
            # Use a cheap finite difference with step
            h = 1e-4
            g = np.zeros_like(x)
            f0 = objective(x)
            for idx in range(len(x)):
                xp = x.copy(); xp[idx] += h
                xm = x.copy(); xm[idx] -= h
                g[idx] = (objective(xp) - objective(xm)) / (2 * h)
            return g

        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 200, 'maxfun': 400,
                                    'ftol': 1e-12, 'gtol': 1e-6})
            xopt = res.x
            Uo = xopt[:dim_u * R].reshape(dim_u, R)
            Vo = xopt[dim_u * R:dim_u * R + dim_v * R].reshape(dim_v, R)
            Wo = xopt[dim_u * R + dim_v * R:].reshape(dim_w, R)
            lo = loss(Uo, Vo, Wo)
            if lo < best_loss:
                best_loss = lo
                best = (Uo.copy(), Vo.copy(), Wo.copy())
        except Exception:
            if cur_loss < best_loss:
                best_loss = cur_loss
                best = (U.copy(), V.copy(), W.copy())

    if best is None:
        return None
    U, V, W = best
    # Round to nearest half-integer if close
    def round_half(X):
        X2 = np.round(X * 2.0) / 2.0
        return X2
    Ur, Vr, Wr = round_half(U), round_half(V), round_half(W)
    if loss(Ur, Vr, Wr) < 1e-6:
        return Ur, Vr, Wr
    return U, V, W

def _build_full_decomposition():
    """
    Build the full 3x3 * 3x7 decomposition by solving a single
    rank-R optimization for the whole tensor, then falling back to
    a block construction if that fails.
    """
    n, m, p = 3, 3, 7

    # First, try to solve the whole tensor at rank 24 (improvement over 63).
    # We have 63 scalar constraints; rank 24 gives 24*27=648 variables.
    # This is very underdetermined, so a solution should exist if the
    # tensor rank is <= 24.
    if _time_left() > 60:
        sol = _solve_group(n, m, p, target_rank=24, seed=1,
                           time_budget=min(55, _time_left() - 5))
        if sol is not None:
            U, V, W = sol
            if _verify(U, V, W, n, m, p, trials=30, tol=1e-5):
                return U, V, W

    # Fallback: block construction. Split p=7 into 3+3+1.
    # Use schoolbook for each block but try a lower-rank 3x3x3 first.
    U_blocks = []
    V_blocks = []
    W_blocks = []

    # Block 1: columns 0,1,2
    if _time_left() > 40:
        sol = _solve_group(n, m, 3, target_rank=23, seed=2,
                           time_budget=min(35, _time_left() - 5))
        if sol is not None and _verify(sol[0], sol[1], sol[2], n, m, 3,
                                       trials=20, tol=1e-5):
            U1, V1, W1 = sol
            U_blocks.append(U1)
            V_blocks.append(V1)
            W_blocks.append(W1)
        else:
            U1, V1, W1 = _schoolbook(n, m, 3)
            U_blocks.append(U1); V_blocks.append(V1); W_blocks.append(W1)
    else:
        U1, V1, W1 = _schoolbook(n, m, 3)
        U_blocks.append(U1); V_blocks.append(V1); W_blocks.append(W1)

    # Block 2: columns 3,4,5
    if _time_left() > 20:
        sol = _solve_group(n, m, 3, target_rank=23, seed=3,
                           time_budget=min(15, _time_left() - 3))
        if sol is not None and _verify(sol[0], sol[1], sol[2], n, m, 3,
                                       trials=20, tol=1e-5):
            U2, V2, W2 = sol
        else:
            U2, V2, W2 = _schoolbook(n, m, 3)
        U_blocks.append(U2); V_blocks.append(V2); W_blocks.append(W2)
    else:
        U2, V2, W2 = _schoolbook(n, m, 3)
        U_blocks.append(U2); V_blocks.append(V2); W_blocks.append(W2)

    # Block 3: column 6 (rank 9 schoolbook)
    U3, V3, W3 = _schoolbook(n, m, 1)
    U_blocks.append(U3); V_blocks.append(V3); W_blocks.append(W3)

    total_rank = sum(b.shape[1] for b in U_blocks)
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    col = 0
    # Map each block's W rows to the correct global k offset
    k_offsets = [0, 3, 6]
    for blk, koff in zip(range(3), k_offsets):
        Ub, Vb, Wb = U_blocks[blk], V_blocks[blk], W_blocks[blk]
        r = Ub.shape[1]
        U[:, col:col + r] = Ub
        # V: block uses columns koff..koff+2 of B
        Vb_full = np.zeros((m * p, r))
        Vb_full[koff * 1:(koff + 3) * 1, :] = Vb  # Vb has shape (m*3, r)
        V[:, col:col + r] = Vb_full
        # W: block uses rows koff..koff+2 of W (which index k)
        Wb_full = np.zeros((p * n, r))
        Wb_full[koff * n:(koff + 3) * n, :] = Wb
        W[:, col:col + r] = Wb_full
        col += r

    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix
    multiplication. Returns dict with U, V, W as lists.
    """
    n, m, p = DIMENSIONS

    # Emergency fallback: schoolbook
    def fallback():
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if _time_left() < 15:
        return fallback()

    try:
        U, V, W = _build_full_decomposition()
        rank = U.shape[1]
        # Verify
        if not _verify(U, V, W, n, m, p, trials=30, tol=1e-5):
            return fallback()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        return fallback()
# EVOLVE-BLOCK-END
