"""Improved packing of 50 equal circles in a unit square using multi-start optimization."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

N = 50
DEADLINE = 120.0  # seconds, safety margin before 150s kill


def _score(c):
    """Compute the radius for a given set of centers."""
    c = np.asarray(c, dtype=np.float64)
    n = c.shape[0]
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Wall distances
    wall_dist = np.minimum(0.5 - np.abs(c), axis=1)
    # Minimum pairwise distance / 2
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    min_wall = np.min(wall_dist)
    return min(min_pair, min_wall)


def _negative_score(c):
    """Negative of score for minimization."""
    return -_score(c)


def _soft_min_radius(c, beta=50.0):
    """
    Smooth surrogate for the max-min radius objective.
    Uses log-sum-exp to create a differentiable approximation.
    """
    c = np.asarray(c, dtype=np.float64)
    n = c.shape[0]
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    # Wall distances
    wall_dist = np.minimum(0.5 - np.abs(c), axis=1)
    
    # Soft-min over pairwise distances/2
    pair_vals = dist / 2.0
    pair_vals = pair_vals[np.isfinite(pair_vals)]
    # Use log-sum-exp for soft-min: -log(sum(exp(-x/beta)))
    shifted = -pair_vals / beta
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_pair = -beta * log_sum
    
    # Soft-min over wall distances
    shifted_w = -wall_dist / beta
    max_shifted_w = np.max(shifted_w)
    log_sum_w = max_shifted_w + np.log(np.sum(np.exp(shifted_w - max_shifted_w)))
    soft_wall = -beta * log_sum_w
    
    # Combined soft-min
    all_vals = np.concatenate([soft_pair.reshape(1), soft_wall.reshape(1)])
    # Actually, we want min(soft_pair, soft_wall), use another soft-min
    shifted_all = -all_vals / beta
    max_shifted_all = np.max(shifted_all)
    log_sum_all = max_shifted_all + np.log(np.sum(np.exp(shifted_all - max_shifted_all)))
    return -beta * log_sum_all


def _negative_soft_min(c, beta=50.0):
    """Negative of soft-min radius for minimization."""
    return -_soft_min_radius(c, beta)


def _slsqp_polish(c0, maxiter=500, ftol=1e-15):
    """
    SLSQP polish: maximize min(min_pair/2, min_wall) subject to constraints.
    Uses auxiliary variable t with epigraph formulation.
    """
    n = c0.shape[0]
    x0 = np.concatenate([c0.flatten(), [0.0]])  # centers + t
    
    def objective(x):
        c = x[:2*n].reshape(n, 2)
        t = x[2*n]
        # Maximize t, so minimize -t
        return -t
    
    def constraints(x):
        c = x[:2*n].reshape(n, 2)
        t = x[2*n]
        cons = []
        # Pairwise distance constraints: |c_i - c_j| >= 2t for all i<j
        for i in range(n):
            for j in range(i+1, n):
                diff = c[i] - c[j]
                dist_sq = np.sum(diff**2)
                # dist_sq >= 4t^2 => dist_sq - 4t^2 >= 0
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: x[2*i] - x[2*j+0] - (x[2*j] - x[2*i+1])**2 - 4*(x[2*n])**2})
        # Wall constraints: 0.5 - |c_ik| >= t for all i,k
        for i in range(n):
            for k in range(2):
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, k=k: 0.5 - np.abs(x[2*i+k]) - x[2*n]})
        return cons
    
    # This is too slow with individual constraints. Use a different approach.
    # Instead, directly optimize the nonsmooth objective with SLSQP on the original formulation.
    
    def obj_direct(x):
        c = x.reshape(n, 2)
        return -_score(c)
    
    def grad_direct(x):
        c = x.reshape(n, 2)
        # Numerical gradient
        eps = 1e-8
        grad = np.zeros_like(x)
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            xm = x.copy()
            xm[i] -= eps
            grad[i] = (obj_direct(xp) - obj_direct(xm)) / (2*eps)
        return grad
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (2*n)
    
    try:
        res = minimize(obj_direct, x0.flatten(), method='SLSQP', 
                       bounds=bounds, options={'maxiter': maxiter, 'ftol': ftol})
        if res.success or res.fun < obj_direct(x0.flatten()):
            return res.x.reshape(n, 2)
    except Exception:
        pass
    
    return c0


def _multi_start_optimize():
    """Multi-start optimization with various initial configurations."""
    best_c = None
    best_score = -np.inf
    start_time = time.time()
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Grid-based initializations with different grid sizes
    for m in range(7, 12):
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        if len(grid) >= N:
            inits.append(grid[:N])
    
    # 2. Random initializations
    np.random.seed(42)
    for _ in range(10):
        c = np.random.uniform(-0.5, 0.5, (N, 2))
        inits.append(c)
    
    # 3. Hexagonal lattice initializations
    for spacing in [0.1, 0.12, 0.14, 0.16]:
        hex_pts = []
        row = 0
        while len(hex_pts) < N:
            for col in range(int(1.0 / spacing) + 2):
                x = -0.5 + col * spacing + (row % 2) * spacing / 2
                y = -0.5 + row * spacing * np.sqrt(3) / 2
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    hex_pts.append([x, y])
            row += 1
        if len(hex_pts) >= N:
            inits.append(np.array(hex_pts[:N]))
    
    # 4. Ring/annular patterns
    for n_rings in [3, 4, 5]:
        pts = []
        for ring in range(1, n_rings + 1):
            n_in_ring = 6 * ring if ring > 1 else 1
            if ring == 1:
                pts.append([0.0, 0.0])
            else:
                for k in range(n_in_ring):
                    angle = 2 * np.pi * k / n_in_ring
                    r = ring * 0.08
                    pts.append([r * np.cos(angle), r * np.sin(angle)])
        if len(pts) >= N:
            inits.append(np.array(pts[:N]))
    
    # Evaluate each initial configuration
    for init_c in inits:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        c = init_c.copy()
        s = _score(c)
        if s > best_score:
            best_score = s
            best_c = c.copy()
        
        # Soft-min optimization
        if time.time() - start_time > DEADLINE - 25:
            break
        try:
            res = minimize(_negative_soft_min, c.flatten(), method='L-BFGS-B',
                          bounds=[(-0.5, 0.5)] * (2*N),
                          options={'maxiter': 200, 'ftol': 1e-12})
            c_opt = res.x.reshape(N, 2)
            s_opt = _score(c_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_c = c_opt.copy()
        except Exception:
            pass
        
        # SLSQP polish on best found
        if time.time() - start_time > DEADLINE - 15:
            break
        if best_c is not None:
            try:
                c_polished = _slsqp_polish(best_c, maxiter=300)
                s_polished = _score(c_polished)
                if s_polished > best_score:
                    best_score = s_polished
                    best_c = c_polished.copy()
            except Exception:
                pass
    
    # Final polish with higher precision
    if best_c is not None and time.time() - start_time < DEADLINE - 10:
        try:
            c_final = _slsqp_polish(best_c, maxiter=1000, ftol=1e-16)
            s_final = _score(c_final)
            if s_final > best_score:
                best_score = s_final
                best_c = c_final.copy()
        except Exception:
            pass
    
    return best_c


def construct():
    """Construct centers of 50 equal circles in the unit square [-1/2, 1/2]^2."""
    start_time = time.time()
    
    # Multi-start optimization
    c = _multi_start_optimize()
    
    # Fallback to grid if optimization failed
    if c is None or not np.all(np.isfinite(c)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        c = grid[:N]
    
    # Ensure valid output
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        c = grid[:N]
    
    # Clip to ensure inside square (safety)
    c = np.clip(c, -0.5, 0.5)
    
    return c
