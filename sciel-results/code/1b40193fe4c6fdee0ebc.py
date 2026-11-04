"""Different strategy: numerical tensor-rank search + exact rational recovery.

We parameterize U (n*m x R), V (m*p x R), W (p*n x R) as dense matrices and
minimize || sum_r u_r (x) v_r (x) w_r - T ||^2 with Adam. After convergence we
round to the nearest half-integer and check the tensor equation EXACTLY.
We sweep R from a lower bound up, returning the smallest R that verifies.
"""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start = time.time()
    deadline = start + 140  # safety margin before 150s kill

    nm = n * m      # 8
    mp = m * p      # 20
    pn = p * n      # 10

    # Build the target tensor T flattened: T_flat[i*m + j, j*p + k, k*n + i] = 1
    # We represent T as a 3D array T3[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    T3 = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T3[a, b, c] = 1.0

    T_flat = T3.reshape(nm, -1)  # (nm, mp*pn)
    T_norm = np.linalg.norm(T3)

    def verify_exact(U, V, W, R):
        """Check sum_r U[a,r] V[b,r] W[c,r] == 1 for all (a,b,c) with T3[a,b,c]==1,
        and == 0 elsewhere. U,V,W assumed half-integer."""
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        # Compute the full 3-tensor
        T = np.einsum('ar,br,cr->abc', U, V, W)
        # Check entries where target is 1
        mask1 = T3 > 0.5
        if not np.allclose(T[mask1], 1.0, atol=1e-9, rtol=0):
            return False
        mask0 = T3 < 0.5
        if not np.allclose(T[mask0], 0.0, atol=1e-9, rtol=0):
            return False
        return True

    def round_half(x):
        """Round to nearest half-integer."""
        return np.round(2.0 * x) / 2.0

    def adam_optimize(U, V, W, R, steps=3000, lr=0.01, seed=0):
        """Adam on (U,V,W) minimizing ||einsum - T3||^2."""
        rng = np.random.default_rng(seed)
        U = U.copy()
        V = V.copy()
        W = W.copy()

        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0

        best_loss = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        for step in range(steps):
            t += 1
            # Forward: T = einsum('ar,br,cr->abc', U, V, W)
            T = np.einsum('ar,br,cr->abc', U, V, W)
            E = T - T3
            loss = np.sum(E * E)
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            # Gradients
            # dL/dU[a,r] = 2 * sum_{b,c} E[a,b,c] * V[b,r] * W[c,r]
            # = 2 * (E * (V^T W^T)) contracted
            # Efficiently: for each r, grad_U[:,r] = 2 * einsum('abc,br,cr->a', E, V[:,r], W[:,r])
            # Use batched: grad_U = 2 * np.einsum('abc,br,cr->ar', E, V, W)
            grad_U = 2.0 * np.einsum('abc,br,cr->ar', E, V, W)
            grad_V = 2.0 * np.einsum('abc,br,cr->br', E, U, W)
            grad_W = 2.0 * np.einsum('abc,br,cr->cr', E, U, V)

            # Adam update
            def adam_update(param, g, m_p, v_p):
                m_p[:] = b1 * m_p + (1 - b1) * g
                v_p[:] = b2 * v_p + (1 - b2) * (g * g)
                m_hat = m_p / (1 - b1 ** t)
                v_hat = v_p / (1 - b2 ** t)
                param[:] -= lr * m_hat / (np.sqrt(v_hat) + eps)

            adam_update(U, grad_U, m_U, v_U)
            adam_update(V, grad_V, m_V, v_V)
            adam_update(W, grad_W, m_W, v_W)

            if step % 200 == 0 and time.time() > deadline - 20:
                break

        return best_U, best_V, best_W, best_loss

    def try_rank(R, seed=0):
        """Try to find an exact half-integer decomposition of rank R."""
        rng = np.random.default_rng(seed)
        # Initialize with small random values
        U = rng.normal(0, 0.5, (nm, R))
        V = rng.normal(0, 0.5, (mp, R))
        W = rng.normal(0, 0.5, (pn, R))

        U, V, W, loss = adam_optimize(U, V, W, R, steps=2000, lr=0.005, seed=seed)
        # Round to half-integers
        Ur = round_half(U)
        Vr = round_half(V)
        Wr = round_half(W)
        if verify_exact(Ur, Vr, Wr, R):
            return Ur, Vr, Wr

        # Try a second seed
        U = rng.normal(0, 0.5, (nm, R))
        V = rng.normal(0, 0.5, (mp, R))
        W = rng.normal(0, 0.5, (pn, R))
        U, V, W, loss = adam_optimize(U, V, W, R, steps=2000, lr=0.005, seed=seed + 1000)
        Ur = round_half(U)
        Vr = round_half(V)
        Wr = round_half(W)
        if verify_exact(Ur, Vr, Wr, R):
            return Ur, Vr, Wr

        return None

    # Lower bound: at least max(nm, mp, pn) is not right; use a simple bound.
    # rank >= max(nm, mp, pn) is false. A better bound: rank >= max(n, m, p) * something.
    # For now, start from a reasonable lower bound.
    # The tensor has support size n*m*p = 40. Rank must be at least ceil(40 / max(nm,mp,pn))... not helpful.
    # Use rank >= max(n*m, m*p, p*n) / something. Actually, just start from a few values.
    # A safe lower bound: rank >= max(n, p) * m / something. Let's just try from 20 upward.
    # Actually, the minimum rank of M_{2,4,5} is at least 2*4 = 8 (trivial) but likely much higher.
    # Let's try ranks from 20 to 39.

    # First, let's have the schoolbook as fallback
    def construct_schoolbook():
        U = np.zeros((nm, nm * mp * pn))
        V = np.zeros((mp, nm * mp * pn))
        W = np.zeros((pn, nm * mp * pn))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try ranks from 20 to 39 (below schoolbook 40)
    for R in range(20, 40):
        if time.time() > deadline - 15:
            break
        result = try_rank(R, seed=R)
        if result is not None:
            Ur, Vr, Wr = result
            # Final verification
            if verify_exact(Ur, Vr, Wr, R):
                return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}

    # Fallback: schoolbook
    return construct_schoolbook()
