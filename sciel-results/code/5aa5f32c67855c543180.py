"""Optimized 16-point placement minimizing (max_pairwise / min_pairwise)^2."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 16
D = 2
DEADLINE = 150.0  # seconds


def _pairwise_dists(pts):
    """Return array of all pairwise distances (upper triangle, flattened)."""
    d = cdist(pts, pts)
    iu = np.triu_indices(N, k=1)
    return d[iu]


def _objective(params, softness=100.0):
    """
    Smooth surrogate for (max/min)^2 using log-sum-exp.
    We minimize log(max^2) - log(min^2) = 2*(logmax - logmin).
    Using soft versions:
      logmax  ~ soft_max(log(d^2))  = logsumexp(log(d^2)) - log(n_pairs)  [approx]
      logmin  ~ soft_min(log(d^2))  = -logsumexp(-log(d^2)) + log(n_pairs)
    Actually simpler: minimize (softmax(d^2)/softmax... )
    Let's use: f = soft_max(d^2) / soft_min(d^2)  where
      soft_max(x) = exp(beta * max) ... no, use logsumexp directly.
    
    Better: minimize  log(soft_max(d^2)) - log(soft_min(d^2))
      soft_max(d^2) = (1/beta) * logsumexp(beta * d^2)
      soft_min(d^2) = -(1/beta) * logsumexp(-beta * d^2)
    Then f = log(soft_max) - log(soft_min) = [loglogsumexp(beta*d^2) - log(beta)] 
           - [log(-logsumexp(-beta*d^2)) + log(beta)]  -- this gets messy with signs.
    
    Cleaner: minimize g = logsumexp(beta*d^2) + logsumexp(-beta*d^2)
    This is a smooth upper bound related to the ratio. Minimizing this
    pushes all d^2 to be equal.
    """
    pts = params.reshape(N, D)
    d = _pairwise_dists(pts)
    # Avoid log(0) issues
    d2 = np.maximum(d * d, 1e-12)
    logd2 = np.log(d2)
    beta = softness
    # soft log-max of d^2
    logmax = np.log(np.sum(np.exp(beta * logd2))) - np.log(len(logd2)) - np.log(beta)
    # soft log-min of d^2
    logmin = -np.log(np.sum(np.exp(-beta * logd2))) + np.log(len(logd2)) + np.log(beta)
    # Objective: 2*(logmax - logmin) approximates log((max/min)^2)
    return 2.0 * (logmax - logmin)


def _true_score(pts):
    """Compute the actual (max/min)^2 score."""
    d = _pairwise_dists(pts)
    if d.min() < 1e-15:
        return 1e12
    return (d.max() / d.min()) ** 2


def _hexagonal_init(scale=1.0):
    """4x4 hexagonal lattice init."""
    pts = []
    for i in range(4):
        for j in range(4):
            x = j * scale + (0.5 if i % 2 else 0.0) * scale * 0.5
            y = i * scale * np.sqrt(3) / 2
            pts.append([x, y])
    return np.array(pts, dtype=float)


def _square_init(scale=1.0):
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append([i * scale, j * scale])
    return np.array(pts, dtype=float)


def _random_init(scale=1.0):
    return np.random.randn(N, D) * scale * 0.5


def construct():
    t0 = time.time()
    best_pts = None
    best_score = 1e18

    def _update(pts):
        nonlocal best_pts, best_score
        s = _true_score(pts)
        if s < best_score:
            best_score = s
            best_pts = pts.copy()

    # Multiple restarts from different initializations
    inits = [
        _hexagonal_init(1.0),
        _square_init(1.0),
        _random_init(1.0),
        _hexagonal_init(0.8),
        _random_init(0.5),
    ]

    # Anneal softness from coarse to fine
    beta_schedule = [5.0, 20.0, 50.0, 100.0, 200.0, 500.0]

    for init_idx, x0 in enumerate(inits):
        if time.time() - t0 > DEADLINE - 10.0:
            break
        _update(x0)
        x_best = x0.copy()
        s_best = best_score

        for beta in beta_schedule:
            if time.time() - t0 > DEADLINE - 10.0:
                break
            try:
                res = minimize(
                    _objective,
                    x_best.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12},
                )
                x_new = res.x.reshape(N, D)
                _update(x_new)
                if _true_score(x_new) < s_best:
                    x_best = x_new
                    s_best = _true_score(x_new)
            except Exception:
                pass

    # Final polish with very high softness on the best point
    if best_pts is not None and time.time() - t0 < DEADLINE - 5.0:
        try:
            res = minimize(
                _objective,
                best_pts.flatten(),
                args=(1000.0,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14},
            )
            _update(res.x.reshape(N, D))
        except Exception:
            pass

    # Safety: ensure distinct points
    if best_pts is None:
        best_pts = _square_init(1.0)

    # Enforce distinctness with tiny jitter if any duplicates
    d = _pairwise_dists(best_pts)
    if d.min() < 1e-10:
        best_pts = best_pts + np.random.randn(N, D) * 1e-6

    return best_pts


# Quick self-test
if __name__ == "__main__":
    pts = construct()
    print("Score:", _true_score(pts))
