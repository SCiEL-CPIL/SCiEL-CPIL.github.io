"""Pack 50 equal circles in the unit square centred at the origin."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial.distance import cdist

N = 50
SIDE = 1.0
HALF = 0.5

# Deadline guard: evaluator kills at 150s; keep a safety margin.
_START = time.time()
DEADLINE = 140.0  # seconds from module load; construct will check against this


def _elapsed():
    return time.time() - _START


def _remaining():
    return max(0.0, DEADLINE - _elapsed())


def _score(c):
    """Largest common radius for centres c (N,2)."""
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # distance to walls
    wall = HALF - np.abs(c)
    r_wall = np.min(wall)
    if r_wall < 0:
        return 0.0
    # pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    r_pair = np.min(d) / 2.0
    return min(r_pair, r_wall)


def _objective_and_grad(x):
    """
    Minimize -r  (i.e. maximize r)
    x = [c_0..c_{N-1}, r]  length 2N+1
    """
    c = x[:2 * N].reshape(N, 2)
    r = x[2 * N]

    # pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(d, np.inf)

    # wall distances
    wall = HALF - np.abs(c)
    wall_min = np.min(wall)

    # The objective is min over all constraints of (slack).
    # We want to maximize r, so minimize -r subject to d >= 2r and wall >= r.
    # Instead, use a smooth penalty / barrier approach via log-sum-exp on violations.
    # Simpler: use a smooth min via log-sum-exp of the slacks (positive when feasible).
    # slacks = [d_ij - 2r for all i<j] and [wall_i - r]
    # We want all slacks >= 0. Maximize r.
    
    # Alternative: directly minimize a smooth surrogate of the "min slack".
    # f = -log(sum(exp(-slack_i / tau)))  -> as tau->0, f -> -min(slack_i)
    # So minimizing f is like maximizing min(slack_i).
    # But we also want to push r up.
    
    # Let's use: objective = -r + penalty where penalty pushes slacks positive.
    # Better: use the epigraph form with smooth min.
    
    # Collect all slacks
    iu = np.triu_indices(N, k=1)
    slacks_pair = d[iu] - 2 * r
    slacks_wall = wall - r
    slacks = np.concatenate([slacks_pair, slacks_wall.flatten()])
    
    tau = 0.01
    # smooth min: -log(sum(exp(-s/tau)))
    # For numerical stability, subtract max
    s_shifted = slacks / tau
    s_max = np.max(s_shifted)
    log_sum = s_max + np.log(np.sum(np.exp(s_shifted - s_max)))
    smooth_min = -log_sum * tau
    
    # Objective: we want to maximize r and keep slacks positive.
    # Use: obj = -r - lambda * smooth_min  where smooth_min is negative when slacks are positive.
    # Actually smooth_min = -logsumexp(-s/tau)*tau. If all s > 0, logsumexp(-s/tau) < 0? No.
    # If s_i are positive, -s_i/tau are negative, exp is < 1, sum is < n, log is negative, -log is positive.
    # So smooth_min > 0 when slacks positive. We want to maximize smooth_min? No, we want slacks >= 0.
    
    # Let me reconsider. The constraint is s_i >= 0. The "margin" is min(s_i).
    # We want to maximize r subject to min(s_i) >= 0.
    # Lagrangian / penalty: minimize -r + (1/2) * sum(max(0, -s_i)^2)
    # This is a smooth (C1) objective.
    
    viol = np.maximum(0, -slacks)
    penalty = 0.5 * np.sum(viol ** 2)
    obj = -r + penalty
    
    # Gradient
    # d obj / d r = -1 + sum over violations: d(-s_i)/dr * (-s_i) for violated
    # For pair (i,j): s = d_ij - 2r, d(-s)/dr = 2, so contrib = 2 * (-s) if violated
    # For wall i: s = w_i - r, d(-s)/dr = 1, contrib = 1 * (-s) if violated
    
    grad_r = -1.0
    if len(viol) > 0:
        # viol is same shape as slacks
        # pair part: first len(iu[0]) entries
        n_pair = len(iu[0])
        viol_pair = viol[:n_pair]
        viol_wall = viol[n_pair:]
        grad_r += 2.0 * np.sum(viol_pair)
        grad_r += 1.0 * np.sum(viol_wall)
    
    # Gradient w.r.t. c
    grad_c = np.zeros_like(c)
    # For each violated pair (i,j): s = d_ij - 2r, -s < 0, viol = -s
    # d obj / d c_i = d(-s)/d c_i * (-s) = (-(c_i - c_j)/d_ij) * (-s) = s * (c_i - c_j)/d_ij
    # Wait: obj = -r + 0.5*sum(viol^2), viol = max(0, -s)
    # d obj / d c_i = sum_j [d(viol_ij^2)/d c_i] = sum_j [2*viol_ij * d(viol_ij)/d c_i]
    # If violated: viol = -s, d(viol)/d c_i = -d(s)/d c_i = -(c_i - c_j)/d_ij
    # So term = 2*(-s) * (-(c_i-c_j)/d_ij) = 2*s*(c_i-c_j)/d_ij... but s < 0, so this is negative * (c_i-c_j)/d_ij
    # Actually let's just do: if violated, contrib to grad c_i = 2 * viol * d(viol)/d c_i = 2*(-s)*(-(c_i-c_j)/d_ij) = 2*s*(c_i-c_j)/d_ij
    # Since s < 0, this pushes c_i toward c_j? That's wrong. Let me recheck.
    # s = d_ij - 2r. If s < 0, circles overlap. We want to increase d_ij.
    # d(s)/d c_i = (c_i - c_j)/d_ij. To increase s, we move c_i in direction (c_i-c_j)/d_ij (away from j).
    # grad_c_i should point in the direction that increases s, i.e., (c_i-c_j)/d_ij.
    # The penalty gradient: d(0.5*viol^2)/d c_i = 2*viol * d(viol)/d c_i
    # viol = -s (when violated), d(viol)/d c_i = -d(s)/d c_i = -(c_i-c_j)/d_ij
    # So 2*(-s)*(-(c_i-c_j)/d_ij) = 2*s*(c_i-c_j)/d_ij. Since s<0, this is in direction -(c_i-c_j)/d_ij = (c_j-c_i)/d_ij, which moves i toward j. WRONG.
    
    # I think the issue is: we're minimizing obj. The penalty term 0.5*viol^2 is positive.
    # To minimize it, we want viol -> 0, i.e., s -> 0 from below.
    # The gradient of the penalty w.r.t. c_i should push to increase s.
    # Let me just compute it numerically-safe way: use the subgradient of max(0,-s).
    # Actually, the gradient of max(0, -s) w.r.t. x is -grad(s) if s<0, else 0.
    # So d(0.5*max(0,-s)^2)/d x = max(0,-s) * d(max(0,-s))/dx = max(0,-s) * (-grad(s)) if s<0 else 0
    # = -max(0,-s) * grad(s) if s<0
    # For pair: grad(s)/dc_i = (c_i-c_j)/d_ij
    # So term = -viol * (c_i-c_j)/d_ij * 2 (from chain rule with 0.5*...^2 -> 2*viol*d(viol))
    # Wait: d(0.5*v^2)/dx = v * dv/dx. v = max(0,-s). dv/dx = -ds/dx if s<0.
    # So = v * (-ds/dx) = -v * ds/dx.
    # For pair i,j: ds/dx_i = (c_i-c_j)/d_ij.
    # So grad_c_i += -viol_ij * (c_i-c_j)/d_ij
    # Since viol_ij = -s_ij > 0 and (c_i-c_j)/d_ij points away from j, -viol*(away) points toward j. STILL WRONG for minimization?
    
    # No wait. If we're minimizing, and the penalty is large when overlapping, the gradient should point in the direction that DECREASES the penalty. The penalty decreases when we separate the circles. So the gradient should point in the direction that increases d_ij, which is (c_i-c_j)/d_ij. But I'm getting the negative.
    
    # Let me just verify with a simple 1D case. Two points x1, x2, constraint x2-x1 >= 2r. s = x2-x1-2r.
    # If s < 0, viol = -s = 2r - x2 + x1.
    # obj = -r + 0.5*viol^2
    # d obj/d x1 = 2*viol * d(viol)/dx1 = 2*viol * 1 = 2*viol > 0.
    # So gradient w.r.t. x1 is positive, meaning we should DECREASE x1 to minimize. Decreasing x1 increases x2-x1, which increases s. CORRECT.
    # In 2D: d(s)/d c_i = (c_i - c_j)/d_ij. This is the unit vector from j to i.
    # d(viol)/d c_i = -d(s)/d c_i = (c_j - c_i)/d_ij.
    # d(0.5*viol^2)/d c_i = 2*viol * (c_j-c_i)/d_ij.
    # So grad_c_i += 2*viol_ij * (c_j-c_i)/d_ij.
    # This points from i toward j. For minimization, we move AGAINST the gradient, so we move from j toward i, i.e., away from j. CORRECT.
    
    # OK so the formula is: grad_c_i += 2 * viol_ij * (c_j - c_i) / d_ij for each violated pair.
    
    # Let's implement it properly.
    for idx in range(n_pair):
        if viol_pair[idx] > 0:
            i, j = iu[0][idx], iu[1][idx]
            dij = d[i, j]
            if dij > 1e-12:
                diff_vec = c[j] - c[i]
                grad_c[i] += 2.0 * viol_pair[idx] * diff_vec / dij
                grad_c[j] += 2.0 * viol_pair[idx] * (-diff_vec) / dij
    
    # Wall constraints: s = w_i - r, w_i = HALF - |c_ik|
    # d(s)/d c_ik = -sign(c_ik)
    # d(viol)/d c_ik = sign(c_ik)
    # grad_c_ik += 2 * viol_wall * sign(c_ik)
    for i in range(N):
        for k in range(2):
            if viol_wall[i, k] > 0:
                grad_c[i, k] += 2.0 * viol_wall[i, k] * np.sign(c[i, k])
    
    grad = np.concatenate([grad_c.flatten(), [grad_r]])
    return obj, grad


def construct():
    """Return centres of N equal circles in [-1/2,1/2]^2."""
    # Start from a good initial guess: hexagonal-like grid
    # For 50 circles, use a 7x7 grid and trim, or use known packing patterns.
    # Let's start with a square grid of 7x7=49 and add one, or 8x8=64 trim to 50.
    
    # Better: use a hexagonal packing initial guess.
    # Hexagonal: rows offset.
    # For 50 circles, try rows of 7,7,7,7,7,7,7,1 = 50? No.
    # 7*7 = 49, plus 1 = 50. Or 8 rows of varying counts.
    
    # Let's just use a square grid 7x7 (49) + 1 extra, or 8x8 trimmed.
    # Actually, let's use the known best packing structure as a hint.
    # The best known has radius ~0.07137.
    
    # Initialize with a square grid
    m = 8  # 8x8 = 64, take 50
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    c0 = grid[:N].copy()
    
    # Add some randomness to break symmetry
    np.random.seed(42)
    c0 = c0 + np.random.normal(0, 0.01, c0.shape)
    
    # Clip to be inside
    c0 = np.clip(c0, -HALF + 0.01, HALF - 0.01)
    
    r0 = 0.05
    x0 = np.concatenate([c0.flatten(), [r0]])
    
    best_x = x0.copy()
    best_score = _score(c0)
    
    # Optimize
    try:
        res = minimize(
            _objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 2000,
                'ftol': 1e-14,
                'gtol': 1e-8,
                'maxls': 50,
            }
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            c_opt = res.x[:2*N].reshape(N, 2)
            s = _score(c_opt)
            if s > best_score:
                best_score = s
                best_x = res.x.copy()
    except Exception:
        pass
    
    # Polish with a second pass if time remains
    if _remaining() > 10.0:
        try:
            c_start = best_x[:2*N].reshape(N, 2)
            r_start = best_x[2*N]
            x_start = np.concatenate([c_start.flatten(), [r_start]])
            res2 = minimize(
                _objective_and_grad,
                x_start,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 3000,
                    'ftol': 1e-16,
                    'gtol': 1e-10,
                    'maxls': 50,
                }
            )
            if res2.x is not None and np.all(np.isfinite(res2.x)):
                c_opt = res2.x[:2*N].reshape(N, 2)
                s = _score(c_opt)
                if s > best_score:
                    best_score = s
                    best_x = res2.x.copy()
        except Exception:
            pass
    
    c_final = best_x[:2*N].reshape(N, 2)
    # Ensure validity: clip slightly inside
    c_final = np.clip(c_final, -HALF + 1e-9, HALF - 1e-9)
    return c_final
