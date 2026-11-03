import numpy as np
import math
import time

N = 150

def construct():
    # Use a good initial configuration: Fibonacci spiral, then a few random restarts
    # to avoid local minima.

    def fib_spiral():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        # Normalize
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        return P

    def random_points():
        X = np.random.randn(N, 3)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    # Pairwise distance matrix helpers
    def pairwise_d2(P):
        # |pi - pj|^2 = 2 - 2 pi.pj
        dots = P @ P.T
        np.fill_diagonal(dots, 0.0)
        d2 = 2.0 - 2.0 * dots
        # Numerical safety
        d2 = np.maximum(d2, 1e-15)
        return d2

    def energy_from_d2(d2):
        iu = np.triu_indices(N, k=1)
        d2_iu = d2[iu]
        # E = sum_{i<j} log(1 / |pi - pj|) = - sum log(sqrt(d2))
        return -float(np.sum(np.log(np.sqrt(d2_iu))))

    def energy(P):
        d2 = pairwise_d2(P)
        return energy_from_d2(d2)

    def rgrad(P):
        dots = P @ P.T
        np.fill_diagonal(dots, 0.0)
        d2 = 2.0 - 2.0 * dots
        d2 = np.maximum(d2, 1e-15)

        # diff[k, j] = P_j - P_k
        diff = P[None, :, :] - P[:, None, :]
        terms = diff / d2[:, :, None]
        G = np.sum(terms, axis=1)

        # Project to tangent space
        dots_G_P = np.sum(G * P, axis=1)
        G_proj = G - dots_G_P[:, None] * P
        return G_proj

    def exp_map(P, V):
        v_norm = np.linalg.norm(V, axis=1)
        v_norm_safe = np.where(v_norm > 1e-12, v_norm, 1e-12)
        cos_t = np.cos(v_norm_safe)
        sin_t = np.sin(v_norm_safe) / v_norm_safe
        Q = cos_t[:, None] * P + sin_t[:, None] * V
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        return Q

    def line_search(P, G, E_cur, max_iter=40):
        # Backtracking line search on the manifold with exponential map.
        # Start with a step that is roughly the inverse of the gradient scale.
        gnorm = np.linalg.norm(G)
        if gnorm < 1e-14:
            return P, E_cur, 0.0
        # Initial step size heuristic
        alpha = 1.0 / (gnorm + 1e-12)
        # Cap to avoid huge rotations
        alpha = min(alpha, 1.0)
        c1 = 1e-4
        for _ in range(max_iter):
            V = -alpha * G
            P_new = exp_map(P, V)
            E_new = energy(P_new)
            if E_new < E_cur - c1 * alpha * gnorm * gnorm:
                return P_new, E_new, alpha
            alpha *= 0.5
            if alpha < 1e-12:
                break
        return P, E_cur, alpha

    def riemannian_optimize(P0, max_iters=300, tol=1e-12):
        P = P0.copy()
        E = energy(P)
        best_P = P.copy()
        best_E = E
        for it in range(max_iters):
            G = rgrad(P)
            gnorm = np.linalg.norm(G)
            if gnorm < tol:
                break
            P_new, E_new, alpha = line_search(P, G, E)
            if E_new < best_E:
                best_E = E_new
                best_P = P_new.copy()
            if E_new >= E - 1e-15 * (1.0 + abs(E)):
                # No meaningful progress
                break
            P = P_new
            E = E_new
        return best_P, best_E

    # Try a few restarts: Fibonacci and a couple random starts
    candidates = []
    candidates.append(fib_spiral())
    # Random restarts with fixed seed for reproducibility
    rng = np.random.default_rng(12345)
    for _ in range(3):
        X = rng.standard_normal((N, 3))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        candidates.append(X)

    best_P = None
    best_E = np.inf

    # Time budget: leave time for final polish
    t0 = time.time()
    time_limit = 100.0  # seconds

    for P0 in candidates:
        if time.time() - t0 > time_limit:
            break
        P_opt, E_opt = riemannian_optimize(P0, max_iters=250)
        if E_opt < best_E:
            best_E = E_opt
            best_P = P_opt

    # Final local polish using SciPy trust-constr with sphere constraints
    try:
        from scipy.optimize import minimize

        x0 = best_P.reshape(-1)

        def obj(x):
            P = x.reshape(N, 3)
            # Ensure normalization during optimization
            P /= np.linalg.norm(P, axis=1, keepdims=True)
            return energy(P)

        def jac(x):
            P = x.reshape(N, 3)
            P /= np.linalg.norm(P, axis=1, keepdims=True)
            G = rgrad(P)
            return G.reshape(-1)

        # Constraint: ||x_i||^2 - 1 = 0 for each point
        def cons(x):
            P = x.reshape(N, 3)
            return np.sum(P * P, axis=1) - 1.0

        res = minimize(
            obj,
            x0,
            jac=jac,
            method='trust-constr',
            constraints=[{'type': 'eq', 'fun': cons}],
            options={'maxiter': 200, 'gtol': 1e-12, 'xtol': 1e-12, 'verbose': 0}
        )

        if res.success or (res.fun is not None and res.fun < best_E):
            P_res = res.x.reshape(N, 3)
            P_res /= np.linalg.norm(P_res, axis=1, keepdims=True)
            E_res = energy(P_res)
            if np.all(np.isfinite(P_res)) and E_res < best_E:
                best_E = E_res
                best_P = P_res

    except Exception:
        # If SciPy fails, keep the Riemannian result
        pass

    # Final normalization and validation
    best_P /= np.linalg.norm(best_P, axis=1, keepdims=True)
    return best_P
