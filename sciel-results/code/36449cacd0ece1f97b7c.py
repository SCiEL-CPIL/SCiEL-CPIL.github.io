"""
Pack 200 equal circles in a unit square ([-0.5, 0.5]^2).
Strategy:
1. Start from a hexagonal lattice packing (better than square grid).
2. Use a force-directed relaxation / optimization to maximize the minimum distance
   (both between circles and to the walls).
3. Use scipy.optimize.minimize with L-BFGS-B on a smooth surrogate (soft-min) of
   the objective, or directly optimize the radius via a feasibility approach.

The score is r = min( min_{i<j} |c_i - c_j|/2, min_i min_k (0.5 - |c_ik|) )
We want to maximize r.

Approach:
- Parameterize positions x (N,2).
- Objective to minimize: -r, but r is non-smooth. Use a soft-min approximation.
- Alternatively, use a two-phase approach:
  Phase 1: Maximize sum of pairwise distances (or use a repulsion potential) to spread circles.
  Phase 2: Polish with a direct optimization of the min-distance using subgradient or
           a smooth approximation.

Given time constraints (150s), we'll use:
- Hexagonal initial layout.
- scipy.optimize.minimize with L-BFGS-B on a smooth objective that approximates
  the negative of the minimum clearance.

Actually, a better approach for this type of problem:
- Use the "radius" as an auxiliary variable and solve a sequence of feasibility problems,
  or use a direct optimization where we maximize r subject to constraints.

Let's use a direct approach:
- Objective: maximize r
- Constraints: |c_i - c_j| >= 2r for all i<j, and |c_ik| <= 0.5 - r for all i,k.

This is a non-linear programming problem. We can use scipy.optimize.minimize with
SLSQP or trust-constr, but with N=200, the number of constraints is O(N^2) which is ~20000.
This might be slow.

Alternative: Use a repulsion-based energy minimization.
Define energy E = sum_{i<j} max(0, 2r - |c_i - c_j|)^2 + sum_i sum_k max(0, |c_ik| - (0.5 - r))^2
But r is also a variable. We can fix r and check feasibility, then binary search on r.

Better: Use a smooth approximation to the min function.
Let f(x) = min(g_1(x), ..., g_m(x)) where g_j are the clearance functions.
We can approximate f with a soft-min: f_soft = -logsumexp(-g_j / tau) * tau
Then maximize f_soft by minimizing -f_soft.

But we need to be careful with the scale.

Let me try a different approach that's known to work well for circle packing:
1. Start with a good initial guess (hexagonal lattice).
2. Use a force-directed method where circles repel each other and are attracted to fill space,
   with a constraint that they don't overlap and stay in the box.
3. Gradually increase the repulsion strength or use an annealing schedule.

Actually, let's use scipy.optimize.minimize with a smooth objective:
- Let d_ij = |c_i - c_j|
- Let w_i = 0.5 - max(|c_ix|, |c_iy|)  (distance to nearest wall)
- We want to maximize min( min_{i<j} d_ij/2, min_i w_i )

Use soft-min: 
  logsumexp for the negative: 
  soft_min(a, b) = -log(exp(-a/tau) + exp(-b/tau)) * tau  ... this is for two values.
  For many values: soft_min(x_1, ..., x_m) = -tau * log(sum_j exp(-x_j/tau))

So define:
  a = min_{i<j} d_ij/2  -> approximate with soft_min over all pairs
  b = min_i w_i -> approximate with soft_min over all circle-wall distances

  r_soft = soft_min(soft_min_pairs, soft_min_walls)

But computing soft_min over all pairs (19900 pairs) every iteration might be slow.

Alternative: Use a repulsion potential that peaks when circles are too close.
E = sum_{i<j} exp(-d_ij^2 / sigma^2) + lambda * sum_i exp(-(0.5 - |c_ik|)^2 / sigma_w^2)
Minimizing E will push circles apart and away from walls. Then the minimum clearance
will be a good estimate.

Let's go with a combined approach:
1. Hexagonal initialization.
2. Minimize a repulsion energy with scipy L-BFGS-B.
3. Compute the actual score.

For the repulsion energy, use:
  E = sum_{i<j} (1/d_ij)^p  for d_ij < threshold, plus wall terms.

Actually, a standard approach is to use:
  E = sum_{i<j} max(0, 2r_target - d_ij)^2  but r_target is unknown.

Let me just use a simple repulsion:
  E = sum_{i<j} 1/d_ij^2  (with a floor to avoid singularity)
  + mu * sum_i (1/(0.5 - |c_ix|)^2 + 1/(0.5 - |c_iy|)^2)  with floors

This will naturally push circles apart and away from walls.

Let's implement this.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 200


def _hex_init():
    """Generate a hexagonal lattice packing in [-0.5, 0.5]^2."""
    points = []
    # Hexagonal lattice: spacing s, rows offset by s/2
    # We want to fit as many as possible, then take first N.
    # Estimate spacing: for N circles in area 1, roughly s ~ 1/sqrt(N) * sqrt(4/sqrt(3))
    # For hex packing, density is pi/(2*sqrt(3)) ~ 0.9069
    # Area per circle ~ 1/N, so radius ~ 0.5/sqrt(N * density_factor)
    # Let's just try a range of spacings.
    
    # Generate a grid of hex lattice points
    max_pts = 500
    for s in np.linspace(0.03, 0.06, 30):
        pts = []
        row = 0
        y = -0.5 + s/2
        while y <= 0.5:
            if row % 2 == 0:
                x = -0.5 + s/2
            else:
                x = -0.5 + s
            while x <= 0.5:
                if abs(x) <= 0.5 and abs(y) <= 0.5:
                    pts.append((x, y))
                x += s
            y += s * np.sqrt(3)/2
            row += 1
        pts = np.array(pts)
        if len(pts) >= N:
            # We have enough, use this spacing
            break
    else:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        pts = np.array([(x, y) for y in ticks for x in ticks])
    
    # Shuffle and take first N for a more uniform random selection
    idx = np.random.permutation(len(pts))[:N]
    return pts[idx]


def _energy(pos, sigma=0.02, mu=1.0, floor=1e-6):
    """Compute repulsion energy."""
    pos = pos.reshape(-1, 2)
    n = len(pos)
    E = 0.0
    
    # Pairwise repulsion
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            d = max(d, floor)
            E += 1.0 / (d**2)
    
    # Wall repulsion
    for i in range(n):
        for k in range(2):
            dist = 0.5 - abs(pos[i, k])
            dist = max(dist, floor)
            E += mu * 1.0 / (dist**2)
    
    return E


def _energy_grad(pos, sigma=0.02, mu=1.0, floor=1e-6):
    """Compute gradient of repulsion energy (numerical)."""
    # Use numerical gradient for simplicity
    eps = 1e-7
    n = len(pos)
    grad = np.zeros_like(pos)
    E0 = _energy(pos)
    for i in range(n):
        for k in range(2):
            pos[i, k] += eps
            E1 = _energy(pos)
            pos[i, k] -= 2*eps
            E2 = _energy(pos)
            pos[i, k] += eps
            grad[i, k] = (E1 - E2) / (2*eps)
    return grad


def _compute_score(pos):
    """Compute the actual score (radius)."""
    pos = np.asarray(pos).reshape(-1, 2)
    n = len(pos)
    
    # Check bounds
    if np.any(np.abs(pos) > 0.5 + 1e-9):
        return 0.0
    
    # Min pairwise distance
    min_pair = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < min_pair:
                min_pair = d
    
    # Min wall distance
    min_wall = np.inf
    for i in range(n):
        for k in range(2):
            d = 0.5 - abs(pos[i, k])
            if d < min_wall:
                min_wall = d
    
    r = min(min_pair / 2.0, min_wall)
    return r


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Initial position: hexagonal lattice
    np.random.seed(42)
    pos = _hex_init().astype(float)
    
    # Optimize using L-BFGS-B with numerical gradient
    # We'll do a few iterations of L-BFGS-B
    
    best_pos = pos.copy()
    best_score = _compute_score(best_pos)
    
    # Try optimization
    try:
        result = minimize(
            _energy,
            pos,
            method='L-BFGS-B',
            jac=_energy_grad,
            bounds=[(-0.5, 0.5)] * (2*N),
            options={'maxiter': 200, 'gtol': 1e-5, 'maxls': 50}
        )
        if result.fun < _energy(pos):
            pos_opt = result.x.reshape(-1, 2)
            score_opt = _compute_score(pos_opt)
            if score_opt > best_score:
                best_pos = pos_opt
                best_score = score_opt
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_pos
    
    # Second round: use the optimized position as start, with tighter settings
    try:
        result2 = minimize(
            _energy,
            best_pos,
            method='L-BFGS-B',
            jac=_energy_grad,
            bounds=[(-0.5, 0.5)] * (2*N),
            options={'maxiter': 300, 'gtol': 1e-6, 'maxls': 100}
        )
        if result2.fun < _energy(best_pos):
            pos_opt2 = result2.x.reshape(-1, 2)
            score_opt2 = _compute_score(pos_opt2)
            if score_opt2 > best_score:
                best_pos = pos_opt2
                best_score = score_opt2
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_pos
    
    # Third round: even tighter
    try:
        result3 = minimize(
            _energy,
            best_pos,
            method='L-BFGS-B',
            jac=_energy_grad,
            bounds=[(-0.5, 0.5)] * (2*N),
            options={'maxiter': 500, 'gtol': 1e-7, 'maxls': 200}
        )
        if result3.fun < _energy(best_pos):
            pos_opt3 = result3.x.reshape(-1, 2)
            score_opt3 = _compute_score(pos_opt3)
            if score_opt3 > best_score:
                best_pos = pos_opt3
                best_score = score_opt3
    except Exception:
        pass
    
    # Ensure all points are strictly inside
    best_pos = np.clip(best_pos, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_pos
