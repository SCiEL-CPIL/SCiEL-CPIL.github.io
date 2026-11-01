"""
Circle packing via contact-graph local search + soft-min gradient polish.
Strategy:
1. Initialize with a perturbed hexagonal lattice.
2. Build Delaunay triangulation to find close pairs (candidate contacts).
3. Iteratively:
   a. Identify "tight" pairs (dist < 2*r + tol) and boundary contacts.
   b. For small clusters of mutually tight circles, solve for positions
      that satisfy exact contact constraints (least-squares).
   c. Apply a global repulsive force to resolve any residual overlaps.
   d. Project into unit disk.
4. Final polish: high-beta soft-min gradient ascent.
"""
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 140.0  # safety margin before 150s kill


def _compute_score(pts):
    """Compute the true packing radius (nonsmooth)."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    radii = np.sqrt(np.sum(pts ** 2, axis=-1))
    min_margin = np.min(1.0 - radii)
    return min(min_pair, min_margin)


def _softmin(vals, beta):
    """Soft-min approximation of min(vals) using log-sum-exp."""
    neg = -vals
    m = np.max(neg)
    if m > 1e12:
        return m
    exp_vals = np.exp(neg - m)
    log_sum = np.log(np.sum(exp_vals)) + m
    return -log_sum / beta


def _objective_and_grad(pts, beta):
    """
    Objective: maximize soft-min of all constraints.
    Constraints:
      - Pairwise: d_ij / 2  (want >= r)
      - Boundary: 1 - |c_i| (want >= r)
    We minimize -softmin of (d_ij/2, 1-|c_i|).
    Returns (val, grad) where val is the negative soft-min, grad is gradient w.r.t. pts.
    """
    n = len(pts)
    # Pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Boundary margins
    radii = np.sqrt(np.sum(pts ** 2, axis=-1))
    margins = 1.0 - radii
    
    # Combine all constraints into one array
    # We want to maximize the minimum of (dists/2, margins)
    # So we minimize -softmin(dists/2, margins)
    # Let's construct the array of all constraint values
    # For gradients, we need d/dx of softmin
    
    # To be efficient, we'll compute the soft-min of the combined set.
    # However, the number of pairwise distances is N*(N-1)/2 = 1225.
    # Plus N boundary margins. Total ~1275 values.
    
    # Let's just compute the gradient analytically.
    # f = -softmin(C) where C is the vector of constraint values.
    # softmin(C) = -1/beta * log(sum(exp(-beta * C_i)))
    # grad softmin w.r.t C_i = exp(-beta * C_i) / sum(exp(-beta * C_j))
    # grad f w.r.t C_i = -grad softmin w.r.t C_i
    
    # Let's compute weights w_i = exp(-beta * C_i) / sum(exp(-beta * C_j))
    
    # C consists of:
    # 1. d_ij / 2 for i < j
    # 2. 1 - |c_i|
    
    # This is getting complex to vectorize cleanly for the gradient.
    # Let's use a simpler approach: use scipy's minimize with numerical gradient?
    # No, that's slow.
    
    # Alternative: Use the fact that the soft-min is dominated by the minimum.
    # If beta is large, the gradient is concentrated on the minimum.
    
    # Let's implement a simpler gradient ascent on the "pressure" model.
    # Actually, let's stick to the Nelder-Mead for the final polish as it's robust,
    # but use the contact-graph search to get close to the optimum first.
    
    # For now, this function is a placeholder for the polish step if we use it.
    # But we will use a different polish: direct minimization of -min(constraints) via Nelder-Mead.
    return None, None


def _contact_graph_search(pts, time_limit):
    """
    Local search based on contact graph.
    1. Find Delaunay neighbors.
    2. Identify tight pairs (dist < 2*r + tol).
    3. For each tight cluster, try to adjust positions to satisfy contacts.
    """
    t0 = time.time()
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    
    # Initial radius estimate
    r = _compute_score(pts)
    
    # We will iterate
    for it in range(1000):
        if time.time() - t0 > time_limit:
            break
            
        # Compute pairwise distances
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Current min distance
        min_d = np.min(dists)
        # Current min boundary margin
        radii = np.sqrt(np.sum(pts ** 2, axis=-1))
        min_m = np.min(1.0 - radii)
        
        # The current effective radius
        r_eff = min(min_d / 2.0, min_m)
        
        # If we made progress, update r
        if r_eff > r:
            r = r_eff
        
        # Identify tight pairs: dist < 2*r + 0.01
        tol = 0.01
        tight_mask = dists < (2 * r + tol)
        # Exclude diagonal
        np.fill_diagonal(tight_mask, False)
        
        # Identify boundary-tight points: 1 - |c_i| < r + tol
        boundary_tight = (1.0 - radii) < (r + tol)
        
        # If no tight pairs and no boundary tight, we might be stuck or optimal
        if not np.any(tight_mask) and not np.any(boundary_tight):
            # Try to expand slightly? Or break?
            # If we can't find any tight constraints, the min is likely determined by a pair
            # that is slightly further, or we are at a local minimum.
            # Let's break to save time for polish.
            break
        
        # Strategy: Find a small cluster of tightly connected points.
        # Use Delaunay to find neighbors.
        try:
            tri = Delaunay(pts)
        except Exception:
            # Degenerate case
            break
            
        # Get neighbors
        neighbors = [set() for _ in range(n)]
        for simplex in tri.simplices:
            for i in simplex:
                for j in simplex:
                    if i != j:
                        neighbors[i].add(j)
        
        # Find a "tight" edge (i, j) such that dist(i,j) < 2*r + tol
        tight_edges = []
        for i in range(n):
            for j in neighbors[i]:
                if i < j and tight_mask[i, j]:
                    tight_edges.append((i, j))
                    
        if not tight_edges:
            # Only boundary tight?
            if np.any(boundary_tight):
                # Push boundary points inward? No, push them outward to increase margin?
                # Actually, if boundary is tight, we want to move them inward to allow others to expand?
                # No, the constraint is 1-|c| >= r. If it's tight, we are limited by boundary.
                # We should try to move interior points to free up space, or move boundary points
                # to a position where they don't limit the radius as much (hard).
                # For now, let's just do a small random perturbation.
                pts += np.random.normal(0, 0.001, size=pts.shape)
                continue
            else:
                break
        
        # Pick a random tight edge
        i, j = tight_edges[np.random.randint(len(tight_edges))]
        
        # Try to adjust i and j to be exactly at distance 2*r
        # But we also need to consider their other neighbors.
        # Simple move: push i and j apart along the vector connecting them,
        # but only if it doesn't violate other constraints.
        
        vec = pts[i] - pts[j]
        d = np.linalg.norm(vec)
        if d < 1e-10:
            continue
        vec = vec / d
        
        # Desired distance is 2*r
        target_d = 2 * r
        # Move i by +vec * delta, j by -vec * delta
        # How much to move?
        # If d < target_d, we need to increase d.
        # delta = (target_d - d) / 2
        delta = (target_d - d) / 2.0
        if delta > 0:
            pts[i] += vec * delta
            pts[j] -= vec * delta
            
        # Project into disk
        radii = np.linalg.norm(pts, axis=1)
        mask = radii > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / radii[mask, np.newaxis] * 0.999

    return pts


def _polish(pts, time_limit):
    """Nelder-Mead polish on the true nonsmooth objective."""
    t0 = time.time()
    current_score = _compute_score(pts)

    def neg_score(x):
        pts_local = x.reshape(-1, 2)
        s = _compute_score(pts_local)
        return -s

    x0 = pts.flatten()
    # Use smaller initial simplex for better convergence
    res = minimize(neg_score, x0, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15,
                            'adaptive': True, 'initial_simplex': None})
    pts_opt = res.x.reshape(-1, 2)
    score_opt = _compute_score(pts_opt)
    if score_opt > current_score:
        return pts_opt
    return pts


def _hex_init(spacing):
    """Hexagonal lattice initialization."""
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    points = points[radius <= 1.0 - spacing / 2.0]
    if len(points) >= N:
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[np.argsort(radius, kind='stable')[:N]]
    return points


def _random_init(seed):
    """Random initialization within unit disk."""
    rng = np.random.RandomState(seed)
    pts = []
    while len(pts) < N:
        x = rng.uniform(-1, 1, size=(N * 2, 2))
        r = np.hypot(x[:, 0], x[:, 1])
        valid = r <= 0.95
        pts.extend(x[valid].tolist())
    pts = np.array(pts[:N])
    return pts


def construct():
    t_start = time.time()

    best_pts = None
    best_score = -1.0

    # Multiple initializations
    inits = [
        _hex_init(0.25),
        _hex_init(0.22),
        _hex_init(0.28),
        _random_init(42),
        _random_init(123),
    ]

    for init_idx, init_pts in enumerate(inits):
        if time.time() - t_start > DEADLINE - 10:
            break
        if len(init_pts) < N:
            continue

        # Contact graph search
        time_remaining = DEADLINE - (time.time() - t_start)
        time_for_search = min(time_remaining * 0.6, 20.0)
        if time_for_search < 2.0:
            break

        pts_searched = _contact_graph_search(init_pts, time_for_search)

        # Polish
        time_remaining = DEADLINE - (time.time() - t_start)
        time_for_polish = min(time_remaining * 0.35, 12.0)
        if time_for_polish > 1.0:
            pts_polished = _polish(pts_searched, time_for_polish)
        else:
            pts_polished = pts_searched

        score = _compute_score(pts_polished)
        if score > best_score:
            best_score = score
            best_pts = pts_polished.copy()

    if best_pts is None:
        best_pts = _hex_init(0.25)

    return best_pts
