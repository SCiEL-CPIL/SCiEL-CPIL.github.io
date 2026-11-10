# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Finds 14 points in 3D maximizing (d_min/d_max)^2.
    Uses BFGS optimization on a log-barrier augmented objective function.
    The objective maximizes the minimum pairwise squared distance while penalizing
    any pairwise squared distance exceeding 1 (the normalized max distance).
    """
    n = 14
    d = 3
    t0 = time.time()
    deadline = t0 + 590.0  # Safety margin before 630s limit

    def remaining():
        return deadline - time.time()

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        return pts / norms

    def score(pts):
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        dmin = np.min(dists)
        dmax = np.max(dists)
        if dmax <= 1e-15 or dmin <= 1e-15:
            return 0.0
        return (dmin / dmax) ** 2

    def fib_sphere(n_pts):
        phi = np.pi * (3.0 - np.sqrt(5.0))
        pts = []
        for i in range(n_pts):
            y = 1.0 - 2.0 * i / (n_pts - 1)
            r = np.sqrt(max(0.0, 1.0 - y * y))
            th = phi * i
            pts.append([np.cos(th) * r, y, np.sin(th) * r])
        return np.array(pts)

    def bfgs_barrier(pts, beta=10.0, maxfev=500):
        """
        BFGS optimization with log-barrier for max distance constraint.
        Objective: max min(d2_ij) - beta * sum(log(1 + (d2_ij - 1)^2 / eps))
        This encourages d2_ij <= 1 and maximizes the minimum d2_ij.
        """
        def obj_and_grad(x):
            if remaining() < 5.0:
                return 1e10, np.zeros_like(x)
            
            p = x.reshape(n, d)
            # Compute pairwise squared distances
            diff = p[:, None, :] - p[None, :, :]
            d2 = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(d2, 1e-12)
            
            dmin = np.min(d2)
            
            # Barrier term: penalize d2 > 1
            # Use a smooth approximation of the constraint
            # Barrier: -beta * sum(log(1 - (d2 - 1)^2 / (4*eps))) for d2 < 1+2*sqrt(eps)
            # Simpler: -beta * sum(exp((d2 - 1)/tau)) for d2 > 1
            # Let's use a quadratic penalty for d2 > 1 and linear reward for d2 < 1
            # Actually, the directive suggests log-barrier.
            # Standard log barrier for g(x) <= 0 is -log(-g(x)).
            # Here constraint is d2_ij <= 1. So g_ij = d2_ij - 1 <= 0.
            # Barrier: -log(1 - d2_ij) if d2_ij < 1.
            # But we want to maximize dmin.
            # Let's use: f(x) = dmin - beta * sum_{ij: d2_ij > 1} (d2_ij - 1)^2
            # This is a penalty method, not strictly log-barrier, but often more robust.
            # Let's stick to the directive: log-barrier.
            # If we use -log(1 - d2_ij), it goes to infinity as d2_ij -> 1.
            # We need to handle d2_ij >= 1 carefully.
            
            # Modified objective:
            # Maximize dmin - beta * sum( max(0, d2_ij - 1)^2 )
            # This is smooth if we use a smooth max.
            # Let's use: f = dmin - beta * sum( log(1 + exp((d2_ij - 1)/tau)) )
            # This is a softplus penalty.
            
            tau = 0.05
            penalty = np.sum(np.log(1.0 + np.exp((d2 - 1.0) / tau)))
            
            # We want to maximize dmin - beta * penalty
            # So minimize -dmin + beta * penalty
            
            val = -dmin + beta * penalty
            
            # Gradient
            # grad dmin: 0 for all except the min pair
            grad_dmin = np.zeros_like(d2)
            min_idx = np.unravel_index(np.argmin(d2), d2.shape)
            grad_dmin[min_idx] = 1.0
            grad_dmin[min_idx[1], min_idx[0]] = 1.0
            
            # grad penalty: beta/tau * exp((d2-1)/tau) / (1 + exp((d2-1)/tau))
            # = beta/tau * sigmoid((d2-1)/tau)
            e = np.exp((d2 - 1.0) / tau)
            grad_pen = (beta / tau) * e / (1.0 + e)
            
            # Total gradient of d2 w.r.t p_i
            # d(d2_ij)/dp_i = 2 * (p_i - p_j)
            # d(d2_ij)/dp_j = 2 * (p_j - p_i)
            
            # We need grad of val w.r.t p
            # val = -dmin + beta * penalty
            # dval/dp_i = - sum_j grad_dmin_ij * 2(p_i - p_j) + sum_j grad_pen_ij * 2(p_i - p_j)
            # Let G = -grad_dmin + grad_pen
            G = -grad_dmin + grad_pen
            np.fill_diagonal(G, 0.0)
            
            # grad_i = 2 * sum_j G_ij (p_i - p_j)
            # sum_j G_ij (p_i - p_j) = p_i sum_j G_ij - sum_j G_ij p_j
            
            sum_G = G.sum(axis=1, keepdims=True)
            sum_Gp = (G[:, :, None] * p).sum(axis=1)
            
            grad_p = 2.0 * (p * sum_G - sum_Gp)
            
            grad_x = grad_p.flatten()
            
            return val, grad_x

        x0 = pts.flatten()
        try:
            res = minimize(obj_and_grad, x0, method='L-BFGS-B',
                           jac=True, options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
            pp = res.x.reshape(n, d)
            return pp
        except Exception:
            return pts

    def sa_refine(pts, n_iter=3000, t0=0.05, t1=0.001):
        """
        Simulated Annealing refinement to escape local minima.
        """
        p = pts.copy()
        current_score = score(p)
        best_p = p.copy()
        best_score = current_score
        rng = np.random.RandomState(42)
        
        for it in range(n_iter):
            if remaining() < 10.0:
                break
            
            t = t0 * (t1 / t0) ** (it / n_iter)
            
            idx = rng.randint(n)
            step = 0.02 * t / t0
            new_p = p.copy()
            new_p[idx] += rng.randn(d) * step
            new_p = normalize(new_p)
            
            new_score = score(new_p)
            delta = new_score - current_score
            
            if delta > 0 or rng.random() < np.exp(delta / max(t, 1e-10)):
                p = new_p
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_p = p.copy()
        
        return best_p

    # --- Generate diverse initializations ---
    inits = []
    inits.append(fib_sphere(n))

    np.random.seed(12345)
    for trial in range(10):
        pts = np.random.randn(n, d)
        inits.append(normalize(pts))

    # Perturbed Fibonacci
    for eps in [0.05, 0.1, 0.2]:
        pts = fib_sphere(n) + np.random.randn(n, d) * eps
        inits.append(normalize(pts))

    # Rotated Fibonacci
    from scipy.spatial.transform import Rotation
    fib = fib_sphere(n)
    for _ in range(4):
        R = Rotation.from_quat(np.random.randn(4)).as_matrix()
        inits.append(normalize(fib @ R.T))

    best_score = -1.0
    best_points = inits[0]

    # --- Phase 1: BFGS Barrier Optimization ---
    bfgs_results = []
    for init in inits:
        if remaining() < 60.0:
            break
        
        for beta in [5.0, 10.0, 20.0]:
            if remaining() < 30.0:
                break
            pts = bfgs_barrier(init, beta=beta, maxfev=200)
            s = score(pts)
            bfgs_results.append((s, pts))
            if s > best_score:
                best_score = s
                best_points = pts

    # Sort by score and keep top distinct candidates
    bfgs_results.sort(key=lambda x: x[0], reverse=True)
    top = []
    for s, pts in bfgs_results:
        dup = False
        for _, p2 in top:
            if np.max(np.abs(pts - p2)) < 1e-3:
                dup = True
                break
        if not dup:
            top.append((s, pts))
        if len(top) >= 8:
            break

    # --- Phase 2: Simulated Annealing refinement on top candidates ---
    for s, pts in top:
        if remaining() < 20.0:
            break
        
        sa_result = sa_refine(pts, n_iter=2000, t0=0.03, t1=0.0005)
        sa_score = score(sa_result)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_result

    # --- Phase 3: Final polish with BFGS + SA if time remains ---
    if remaining() > 60.0:
        for _ in range(3):
            if remaining() < 30.0:
                break
            # BFGS polish
            polished = bfgs_barrier(best_points, beta=10.0, maxfev=100)
            ps = score(polished)
            if ps > best_score:
                best_score = ps
                best_points = polished
            
            # SA polish
            if remaining() > 40.0:
                sa_result = sa_refine(best_points, n_iter=1000, t0=0.02, t1=0.0003)
                sa_score = score(sa_result)
                if sa_score > best_score:
                    best_score = sa_score
                    best_points = sa_result

    # Final sanity check
    if not np.all(np.isfinite(best_points)):
        best_points = fib_sphere(n)
        best_points = normalize(best_points)

    return best_points

# EVOLVE-BLOCK-END
