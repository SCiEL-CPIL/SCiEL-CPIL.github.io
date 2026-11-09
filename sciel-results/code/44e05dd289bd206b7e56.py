import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist

_DEADLINE_SECONDS = 630.0
_SAFETY_MARGIN = 60.0


def _raw_score(points):
    """
    Calculate the score: (d_min / d_max)^2.
    Returns 0 if invalid.
    """
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0

    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    dmin = np.min(dists)
    dmax = np.max(dists)
    if not (np.isfinite(dmin) and np.isfinite(dmax)):
        return 0.0
    if dmax <= 1e-15 or dmin <= 1e-15:
        return 0.0
    r = dmin / dmax
    return float(r * r)


def _objective_de(x):
    """
    Objective for differential evolution.
    We want to maximize (d_min/d_max)^2, so minimize its negative.
    x is a flat array of shape (42,).
    """
    X = x.reshape(14, 3)
    dists = pdist(X)
    dmin = np.min(dists)
    dmax = np.max(dists)
    if dmax <= 1e-15 or dmin <= 1e-15:
        return 1.0
    return -(dmin / dmax) ** 2


def _objective_grad(x):
    """
    Objective for local minimization (L-BFGS-B).
    We want to maximize (d_min/d_max)^2.
    Minimize f = -log(d_min/d_max) = log(d_max) - log(d_min)
    This is smoother than the squared ratio.
    """
    X = x.reshape(14, 3)
    D = X[:, None, :] - X[None, :, :]
    D2 = np.sum(D * D, axis=2)
    D2 = np.maximum(D2, 1e-30)
    dists = np.sqrt(D2)
    G = D / dists[:, :, None]

    iu = np.triu_indices(14, k=1)
    pd_d = dists[iu]
    pd_G = G[iu]

    dmin = np.min(pd_d)
    dmax = np.max(pd_d)

    if dmin <= 1e-15 or dmax <= 1e-15:
        return 1.0, np.zeros(42)

    # Gradient of log(dmax) - log(dmin)
    # d(log dmax)/dX[k] = (X[k] - X[argmax])/dmax for the max pair
    # d(log dmin)/dX[k] = (X[k] - X[argmin])/dmin for the min pair
    # We want to minimize f = log(dmax) - log(dmin)
    # grad_f = grad_log_dmax - grad_log_dmin

    max_idx = np.argmax(pd_d)
    min_idx = np.argmin(pd_d)

    k_max = iu[0][max_idx]
    l_max = iu[1][max_idx]
    k_min = iu[0][min_idx]
    l_min = iu[1][min_idx]

    g = np.zeros((14, 3))

    # Gradient of log(dmax)
    g[k_max] += pd_G[max_idx] / dmax
    g[l_max] -= pd_G[max_idx] / dmax

    # Gradient of log(dmin)
    g[k_min] -= pd_G[min_idx] / dmin
    g[l_min] += pd_G[min_idx] / dmin

    value = np.log(dmax) - np.log(dmin)
    return value, g.reshape(-1)


def _objective_soft(x, beta=200.0):
    """
    Soft objective using log-sum-exp for smoothness.
    Minimizes log(dmax_soft) - log(dmin_soft).
    """
    X = x.reshape(14, 3)
    D = X[:, None, :] - X[None, :, :]
    D2 = np.sum(D * D, axis=2)
    D2 = np.maximum(D2, 1e-30)
    dists = np.sqrt(D2)

    iu = np.triu_indices(14, k=1)
    pd_d = dists[iu]

    # Soft max: log(sum(exp(beta * d))) / beta
    # Soft min: -log(sum(exp(-beta * d))) / beta
    # To avoid overflow, shift by max/min
    max_d = np.max(pd_d)
    min_d = np.min(pd_d)
    
    # log-sum-exp trick
    # log(sum(exp(beta*(d - max_d)))) + beta*max_d
    e_max = np.exp(beta * (pd_d - max_d))
    soft_max = (np.log(np.sum(e_max)) + beta * max_d) / beta
    
    e_min = np.exp(-beta * (pd_d - min_d))
    soft_min = -(np.log(np.sum(e_min)) - beta * min_d) / beta
    
    # Gradient
    # d(soft_max)/dX = (sum(exp(beta*(d-max_d)) * grad_d) / sum(exp(beta*(d-max_d)))) / 1  ... wait
    # Let S = sum(exp(beta*(d_i - max_d))). soft_max = (log(S) + beta*max_d)/beta = log(S)/beta + max_d.
    # d(soft_max)/dX = (1/beta) * (1/S) * sum(exp(beta*(d_i - max_d)) * beta * grad_d_i)
    # = (1/S) * sum(exp(beta*(d_i - max_d)) * grad_d_i)
    # Note: grad_d_i is vector in R^42.
    
    # We need gradients of each distance d_i wrt X.
    # d(dist_ij)/dX = (X_i - X_j)/dist_ij for i, and -(X_i - X_j)/dist_ij for j.
    
    # This is expensive to compute for all pairs in the sum if we do it naively.
    # But N=14 is small.
    
    G = D / dists[:, :, None]
    # G[i,j] = (X_i - X_j) / dist_ij
    
    # We need to sum over pairs (i,j) with i<j.
    # Let w_i = exp(beta*(d_i - max_d)) for max, and w'_i = exp(-beta*(d_i - min_d)) for min.
    
    w_max = e_max
    w_min = e_min
    
    sum_w_max = np.sum(w_max)
    sum_w_min = np.sum(w_min)
    
    grad_soft_max = np.zeros((14, 3))
    grad_soft_min = np.zeros((14, 3))
    
    # Iterate over pairs
    for idx in range(len(pd_d)):
        i, j = iu[0][idx], iu[1][idx]
        g_ij = G[i, j] # (X_i - X_j) / d_ij
        
        # Contribution to grad_soft_max
        coeff_max = w_max[idx] / sum_w_max
        grad_soft_max[i] += coeff_max * g_ij
        grad_soft_max[j] -= coeff_max * g_ij
        
        # Contribution to grad_soft_min
        coeff_min = w_min[idx] / sum_w_min
        grad_soft_min[i] -= coeff_min * g_ij # because d(-log(sum(exp(-beta*d))))/dX = ...
        # Let's re-derive grad_soft_min carefully.
        # soft_min = -1/beta * log(sum(exp(-beta*(d_i - min_d))))
        # Let T = sum(exp(-beta*(d_i - min_d))).
        # d(soft_min)/dX = -1/beta * (1/T) * sum(exp(-beta*(d_i - min_d)) * (-beta) * grad_d_i)
        # = (1/T) * sum(exp(-beta*(d_i - min_d)) * grad_d_i)
        # So it's the same form as max, but with different weights.
        # Wait, the sign in the exponent is negative.
        # d/dX exp(-beta*d_i) = -beta * exp(-beta*d_i) * grad_d_i.
        # So sum term is sum(w'_i * (-beta) * grad_d_i).
        # Then multiply by -1/beta.
        # Result: (1/T) * sum(w'_i * grad_d_i).
        # So yes, same form.
        
        grad_soft_min[i] += coeff_min * g_ij
        grad_soft_min[j] -= coeff_min * g_ij

    # Objective: log(dmax) - log(dmin) -> soft_max - soft_min
    value = soft_max - soft_min
    
    # Gradient: grad_soft_max - grad_soft_min
    grad = grad_soft_max - grad_soft_min
    
    return value, grad.reshape(-1)


def _polish(x0, deadline, use_soft=True):
    """
    Polish a starting point using L-BFGS-B.
    First with soft objective, then with hard objective.
    """
    if time.time() > deadline:
        return x0

    x_current = x0.copy()
    
    # Phase 1: Soft polish
    if use_soft and time.time() < deadline - 5:
        def obj_soft(x):
            if time.time() > deadline:
                raise RuntimeError("deadline")
            return _objective_soft(x)[0]
        
        def jac_soft(x):
            if time.time() > deadline:
                raise RuntimeError("deadline")
            return _objective_soft(x)[1]

        try:
            res = minimize(
                obj_soft,
                x_current,
                jac=jac_soft,
                method="L-BFGS-B",
                options={
                    "maxiter": 1000,
                    "maxfun": 50000,
                    "ftol": 1e-12,
                    "gtol": 1e-10,
                },
            )
            x_current = res.x
        except Exception:
            pass

    # Phase 2: Hard polish
    if time.time() < deadline - 5:
        def obj_hard(x):
            if time.time() > deadline:
                raise RuntimeError("deadline")
            return _objective_grad(x)[0]
        
        def jac_hard(x):
            if time.time() > deadline:
                raise RuntimeError("deadline")
            return _objective_grad(x)[1]

        try:
            res = minimize(
                obj_hard,
                x_current,
                jac=jac_hard,
                method="L-BFGS-B",
                options={
                    "maxiter": 2000,
                    "maxfun": 100000,
                    "ftol": 1e-15,
                    "gtol": 1e-14,
                    "maxls": 200,
                },
            )
            x_current = res.x
        except Exception:
            pass

    return x_current


def min_max_dist_dim3_14():
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS - _SAFETY_MARGIN

    # Bounds for each coordinate
    # We expect points to be in a reasonable range, say [-2, 2]
    bounds = [(-2.0, 2.0)] * 42

    best_pts = None
    best_score = 0.0

    # Phase 1: Global search with Differential Evolution
    # Use a small population to stay within time limits
    rng = np.random.default_rng(42)
    x_init = rng.uniform(-1.0, 1.0, 42)

    de_kwargs = {
        "bounds": bounds,
        "x0": x_init,
        "maxiter": 100,
        "popsize": 6,
        "tol": 1e-8,
        "mutation": (0.5, 1.0),
        "recombination": 0.7,
        "seed": 42,
        "polish": False,
        "init": "latinhypercube",
    }

    try:
        res = differential_evolution(_objective_de, **de_kwargs)
        x_de = res.x
        s_de = _raw_score(x_de.reshape(14, 3))
        if s_de > best_score:
            best_score = s_de
            best_pts = x_de.copy()
    except Exception:
        pass

    # Phase 2: Polish the best result from DE
    if best_pts is not None and time.time() < deadline:
        x_polished = _polish(best_pts, deadline)
        s_polished = _raw_score(x_polished.reshape(14, 3))
        if s_polished > best_score:
            best_score = s_polished
            best_pts = x_polished.copy()

    # Phase 3: Local restarts from random points if time permits
    rng = np.random.default_rng(123)
    restarts = 0
    while time.time() < deadline - 60:
        x_rand = rng.uniform(-1.0, 1.0, 42)
        x_polished = _polish(x_rand, deadline)
        s_polished = _raw_score(x_polished.reshape(14, 3))
        if s_polished > best_score:
            best_score = s_polished
            best_pts = x_polished.copy()
        restarts += 1
        # Limit restarts to avoid infinite loop if polish is fast
        if restarts > 20:
            break

    # If we still have time, do a final high-precision polish on the best point
    if best_pts is not None and time.time() < deadline - 10:
        x_final = _polish(best_pts, deadline, use_soft=False)
        s_final = _raw_score(x_final.reshape(14, 3))
        if s_final > best_score:
            best_score = s_final
            best_pts = x_final.copy()

    if best_pts is None:
        # Fallback
        best_pts = np.zeros(42)

    best_pts = np.asarray(best_pts, dtype=np.float64).reshape(14, 3)
    if not np.all(np.isfinite(best_pts)):
        best_pts = np.zeros((14, 3))

    return best_pts
