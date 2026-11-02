import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    nr, mr, pr = n * m, m * p, p * n  # 4, 16, 16

    # ------------------------------------------------------------------
    # The multiplication tensor: T[a,b,c] = 1 with a=i*m+j, b=j*p+k, c=k*n+i
    # (the official trace convention for W).  T has 32 ones, one per (i,j,k).
    # ------------------------------------------------------------------
    T = np.zeros((nr, mr, pr), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W, tol=1e-8):
        Tc = np.einsum('at,bt,ct->abc', U, V, W)
        err = float(np.max(np.abs(Tc - T)))
        return err < tol, err

    # ------------------------------------------------------------------
    # Deterministic rank-28 fallback: Strassen (rank-7) applied to each of the
    # 4 column-pairs (2x2 x 2x2 blocks).  This is exactly correct and always
    # returned if the search finds nothing better, so the score never regresses
    # below the parent's 1.1429.
    # ------------------------------------------------------------------
    def get_strassen_28():
        rank = 28
        U = np.zeros((nr, rank), dtype=float)
        V = np.zeros((mr, rank), dtype=float)
        W = np.zeros((pr, rank), dtype=float)
        strassen_products = [
            {'A': {(0, 0): 1, (1, 1): 1}, 'B': {(0, 0): 1, (1, 1): 1}, 'C': {(0, 0): 1, (1, 1): 1}},
            {'A': {(1, 0): 1, (1, 1): 1}, 'B': {(0, 0): 1}, 'C': {(1, 0): 1, (1, 1): -1}},
            {'A': {(0, 0): 1}, 'B': {(0, 1): 1, (1, 1): -1}, 'C': {(0, 1): 1, (1, 1): 1}},
            {'A': {(1, 1): 1}, 'B': {(1, 0): 1, (0, 0): -1}, 'C': {(0, 0): 1, (1, 0): 1}},
            {'A': {(0, 0): 1, (0, 1): 1}, 'B': {(1, 1): 1}, 'C': {(0, 0): -1, (0, 1): 1}},
            {'A': {(1, 0): 1, (0, 0): -1}, 'B': {(0, 0): 1, (0, 1): 1}, 'C': {(1, 1): 1}},
            {'A': {(0, 1): 1, (1, 1): -1}, 'B': {(1, 0): 1, (1, 1): 1}, 'C': {(0, 0): 1}},
        ]
        for q in range(p // 2):
            k1 = 2 * q
            for r, prod in enumerate(strassen_products):
                t = q * 7 + r
                for (i, j), coeff in prod['A'].items():
                    U[i * m + j, t] = coeff
                for (j, k_sub), coeff in prod['B'].items():
                    V[j * p + (k1 + k_sub), t] = coeff
                for (i, k_sub), coeff in prod['C'].items():
                    W[(k1 + k_sub) * n + i, t] = coeff
        return U, V, W

    # ------------------------------------------------------------------
    # Structured random rank-r initialization.  For each column-pair block q we
    # draw an independent rank-(r//4) random 2x2x2 decomposition.  This respects
    # the tensor's per-block sparsity, so the start is a *genuine* rank-r
    # candidate (not a broken truncation), which is what lets gradient polish
    # actually converge to an exact solution.
    # ------------------------------------------------------------------
    def random_init(rank, seed):
        rng = np.random.default_rng(seed)
        U = np.zeros((nr, rank), dtype=float)
        V = np.zeros((mr, rank), dtype=float)
        W = np.zeros((pr, rank), dtype=float)
        s = max(1, rank // 4)  # per-block rank
        for q in range(p // 2):
            k1 = 2 * q
            for r in range(s):
                t = q * s + r
                U[:, t] = rng.normal(size=nr)
                v = np.zeros(mr); w = np.zeros(pr)
                for j in range(m):
                    for k in range(2):
                        v[j * p + (k1 + k)] = rng.normal()
                for k in range(2):
                    for i in range(n):
                        w[(k1 + k) * n + i] = rng.normal()
                V[:, t] = v
                W[:, t] = w
        return U, V, W

    # ------------------------------------------------------------------
    # Warm-start from a known higher-rank solution: truncate the rank-28
    # Strassen solution to rank r by keeping the most "important" columns
    # (those with largest norm in U), then polish.  This gives the optimizer
    # a structured starting point closer to a valid solution than pure random.
    # ------------------------------------------------------------------
    def warm_start_from_28(U28, V28, W28, rank):
        if rank >= 28:
            return U28[:, :rank].copy(), V28[:, :rank].copy(), W28[:, :rank].copy()
        # Rank columns by the L2 norm of their U-column (proxy for importance)
        u_norms = np.linalg.norm(U28, axis=0)
        order = np.argsort(u_norms)[::-1][:rank]
        return U28[:, order].copy(), V28[:, order].copy(), W28[:, order].copy()

    # ------------------------------------------------------------------
    # Smooth polish: plain gradient flow on the *smooth* tensor-matching loss
    # with an Armijo backtracking line search (handles the non-convex /
    # non-smooth-in-norm landscape better than a fixed learning rate).
    # ------------------------------------------------------------------
    def polish(U0, V0, W0, deadline, tol=1e-9, max_iters=400, lr0=0.02):
        U, V, W = U0.copy(), V0.copy(), W0.copy()
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_loss = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            err = np.einsum('at,bt,ct->abc', U, V, W) - T
            loss = float(np.sum(err * err))
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_loss < tol:
                break
            gU = np.einsum('abc,bt,ct->at', err, V, W)
            gV = np.einsum('abc,at,ct->bt', err, U, W)
            gW = np.einsum('abc,at,bt->ct', err, U, V)
            gnorm = float(np.sqrt(np.sum(gU * gU) + np.sum(gV * gV) + np.sum(gW * gW)))
            if gnorm < 1e-14:
                break
            lr = lr0 * (0.98 ** it)
            step = 0.5
            improved = False
            for _ in range(12):
                Un = U - lr * step * gU
                Vn = V - lr * step * gV
                Wn = W - lr * step * gW
                en = np.einsum('at,bt,ct->abc', Un, Vn, Wn) - T
                ln = float(np.sum(en * en))
                if ln < loss - 1e-12 * max(1.0, loss):
                    U, V, W, loss = Un, Vn, Wn, ln
                    improved = True
                    break
                step *= 0.5
            if not improved:
                break
        return best_U, best_V, best_W, best_loss

    # ------------------------------------------------------------------
    # L-BFGS-style polish using scipy.optimize.minimize with a smooth loss.
    # This often escapes local minima that plain gradient flow gets stuck in,
    # especially when combined with multiple restarts.
    # ------------------------------------------------------------------
    def polish_lbfgs(U0, V0, W0, deadline, tol=1e-10, maxiter=200):
        from scipy.optimize import minimize
        # Flatten parameters
        U_flat = U0.flatten()
        V_flat = V0.flatten()
        W_flat = W0.flatten()
        x0 = np.concatenate([U_flat, V_flat, W_flat])
        nU = U0.size
        nV = V0.size
        nW = W0.size

        def loss_and_grad(x):
            U = x[:nU].reshape(nr, -1)
            V = x[nU:nU+nV].reshape(mr, -1)
            W = x[nU+nV:].reshape(pr, -1)
            err = np.einsum('at,bt,ct->abc', U, V, W) - T
            loss = float(np.sum(err * err))
            gU = np.einsum('abc,bt,ct->at', err, V, W)
            gV = np.einsum('abc,at,ct->bt', err, U, W)
            gW = np.einsum('abc,at,bt->ct', err, U, V)
            grad = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
            return loss, grad

        # Use L-BFGS-B with gradient
        try:
            res = minimize(
                loss_and_grad, x0, jac=True, method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': tol, 'gtol': 1e-8}
            )
            x_opt = res.x
            U_opt = x_opt[:nU].reshape(nr, -1)
            V_opt = x_opt[nU:nU+nV].reshape(mr, -1)
            W_opt = x_opt[nU+nV:].reshape(pr, -1)
            err = np.einsum('at,bt,ct->abc', U_opt, V_opt, W_opt) - T
            loss = float(np.sum(err * err))
            return U_opt, V_opt, W_opt, loss
        except Exception:
            return U0, V0, W0, np.inf

    # ------------------------------------------------------------------
    # Final high-precision pass: drive the loss to a *strict* tolerance so the
    # half-integer rounding below lands on an exact integer/half-integer solution.
    # ------------------------------------------------------------------
    def high_precision_polish(U0, V0, W0, deadline, tol=1e-12, max_iters=300, lr0=0.005):
        return polish(U0, V0, W0, deadline, tol=tol, max_iters=max_iters, lr0=lr0)

    def round_halfint(U, V, W):
        U2 = np.round(U * 2) / 2
        V2 = np.round(V * 2) / 2
        W2 = np.round(W * 2) / 2
        if verify(U2, V2, W2, tol=1e-7)[0]:
            return U2, V2, W2
        Ui = np.round(U); Vi = np.round(V); Wi = np.round(W)
        if verify(Ui, Vi, Wi, tol=1e-7)[0]:
            return Ui, Vi, Wi
        return None

    # ------------------------------------------------------------------
    # Main search.
    # ------------------------------------------------------------------
    start = time.time()
    deadline = start + 140.0  # hard safety margin well under the 150 s kill

    # Always-valid rank-28 baseline (never regresses).
    U28, V28, W28 = get_strassen_28()
    ok28, _ = verify(U28, V28, W28)
    if not ok28:  # extremely defensive: fall back to schoolbook rank 32
        U = np.zeros((nr, 32)); V = np.zeros((mr, 32)); W = np.zeros((pr, 32))
        for idx in range(32):
            i = idx // 16; j = (idx // 8) % 2; k = idx % 8
            U[i * m + j, idx] = 1.0
            V[j * p + k, idx] = 1.0
            W[k * n + i, idx] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 28
    best_U, best_V, best_W = U28, V28, W28

    # Try ranks 27 down to 16.  For each rank, several structured random
    # restarts AND warm-starts from the rank-28 solution; the first that
    # polishes to a valid exact solution is kept and we move to the next
    # (lower) rank.
    for target in range(27, 15, -1):
        if time.time() > deadline - 8.0:
            break
        s = max(1, target // 4)
        found = False
        for restart in range(6):
            if time.time() > deadline - 5.0:
                break
            # Alternate between random init and warm-start from rank-28
            if restart % 2 == 0:
                U0, V0, W0 = random_init(target, seed=100000 + target * 13 + restart * 7)
            else:
                U0, V0, W0 = warm_start_from_28(U28, V28, W28, target)
                # Add small random perturbation to escape exact truncation
                rng = np.random.default_rng(200000 + target * 17 + restart * 11)
                U0 += rng.normal(0, 0.01, size=U0.shape)
                V0 += rng.normal(0, 0.01, size=V0.shape)
                W0 += rng.normal(0, 0.01, size=W0.shape)
            
            # Try plain gradient polish first
            U1, V1, W1, loss1 = polish(U0, V0, W0, deadline, tol=1e-9, max_iters=200, lr0=0.02)
            
            if loss1 < 1e-9:
                U2, V2, W2, _ = high_precision_polish(U1, V1, W1, deadline, tol=1e-12)
                r = round_halfint(U2, V2, W2)
                if r is None:
                    r = round_halfint(U1, V1, W1)
                if r is not None:
                    best_rank = target
                    best_U, best_V, best_W = r
                    found = True
                    break
            
            # If plain polish didn't work, try L-BFGS polish
            if loss1 > 1e-9 and time.time() < deadline - 10.0:
                U_lb, V_lb, W_lb, loss_lb = polish_lbfgs(U1, V1, W1, deadline, tol=1e-10, maxiter=100)
                if loss_lb < 1e-9:
                    U2, V2, W2, _ = high_precision_polish(U_lb, V_lb, W_lb, deadline, tol=1e-12)
                    r = round_halfint(U2, V2, W2)
                    if r is None:
                        r = round_halfint(U_lb, V_lb, W_lb)
                    if r is not None:
                        best_rank = target
                        best_U, best_V, best_W = r
                        found = True
                        break
        
        if found:
            # keep going lower while time allows (search is per-rank, cheap)
            continue
        # rank `target` not cracked; lower ranks are harder, so stop.
        break

    U, V, W = best_U, best_V, best_W
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
