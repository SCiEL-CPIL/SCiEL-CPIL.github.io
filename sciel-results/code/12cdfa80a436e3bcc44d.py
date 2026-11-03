"""Heilbronn triangles: maximize the smallest triangle area formed by 11 points in an equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Vertices of the equilateral triangle
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])

# Precompute all triangle triples
TRIPLES = [(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)]
NTRIP = len(TRIPLES)


def _soft_min_area(flat, beta):
    """Compute smooth soft-min of triangle areas and its gradient.
    Returns (value, grad) where value approximates the min area.
    """
    pts = flat.reshape(N, 2)
    areas = np.empty(NTRIP)
    for idx, (i, j, k) in enumerate(TRIPLES):
        dx1 = pts[j, 0] - pts[i, 0]
        dy1 = pts[j, 1] - pts[i, 1]
        dx2 = pts[k, 0] - pts[i, 0]
        dy2 = pts[k, 1] - pts[i, 1]
        det = dx1 * dy2 - dy1 * dx2
        areas[idx] = 0.5 * abs(det)
    
    # Soft-min: f = (1/beta) * log(sum(exp(-beta * area_i)))
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    value = log_sum / beta
    
    # Gradient
    exp_vals = np.exp(neg_beta_areas - max_val)
    w = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros(2 * N)
    for idx, (i, j, k) in enumerate(TRIPLES):
        wi = w[idx]
        if wi < 1e-15:
            continue
        dx1 = pts[j, 0] - pts[i, 0]
        dy1 = pts[j, 1] - pts[i, 1]
        dx2 = pts[k, 0] - pts[i, 0]
        dy2 = pts[k, 1] - pts[i, 1]
        det = dx1 * dy2 - dy1 * dx2
        sign_det = 1.0 if det >= 0 else -1.0
        factor = 0.5 * sign_det * wi
        grad[2 * i + 0] += factor * (-dy2)
        grad[2 * i + 1] += factor * (dx2)
        grad[2 * j + 0] += factor * (dy2)
        grad[2 * j + 1] += factor * (-dx2)
        grad[2 * k + 0] += factor * (-dy1)
        grad[2 * k + 1] += factor * (dx1)
    
    return value, grad


def _true_min_area(pts):
    """Compute the true minimum triangle area."""
    min_a = np.inf
    for (i, j, k) in TRIPLES:
        dx1 = pts[j, 0] - pts[i, 0]
        dy1 = pts[j, 1] - pts[i, 1]
        dx2 = pts[k, 0] - pts[i, 0]
        dy2 = pts[k, 1] - pts[i, 1]
        a = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
        if a < min_a:
            min_a = a
    return min_a


def _in_triangle(pts, eps=1e-9):
    """Check if all points are inside or on the triangle."""
    for p in pts:
        x, y = p
        # Check three half-planes
        # 1. y >= 0
        if y < -eps:
            return False
        # 2. y + sqrt(3)*x <= sqrt(3)  (line from (1,0) to (0.5, sqrt(3)/2))
        if y + np.sqrt(3) * x > np.sqrt(3) + eps:
            return False
        # 3. y - sqrt(3)*x <= 0  (line from (0,0) to (0.5, sqrt(3)/2))
        if y - np.sqrt(3) * x > eps:
            return False
    return True


def _clamp_to_triangle(pts, margin=0.002):
    """Project points inside the triangle with a small margin."""
    for idx in range(N):
        x, y = pts[idx]
        # Barycentric coordinates: p = a*V0 + b*V1 + c*V2
        # x = b + 0.5*c, y = c*sqrt(3)/2
        c = 2 * y / np.sqrt(3)
        b = x - c / 2
        a = 1 - b - c
        # Clamp to [margin, 1-2*margin]
        if a < margin:
            a = margin
        if b < margin:
            b = margin
        if c < margin:
            c = margin
        s = a + b + c
        a /= s
        b /= s
        c /= s
        pts[idx, 0] = b + 0.5 * c
        pts[idx, 1] = c * np.sqrt(3) / 2
    return pts


def _penalty(flat):
    """Penalty for points outside the triangle."""
    pts = flat.reshape(N, 2)
    penalty = 0.0
    for p in pts:
        x, y = p
        if y < 0:
            penalty += y ** 2 * 1000
        if y + np.sqrt(3) * x > np.sqrt(3):
            penalty += (y + np.sqrt(3) * x - np.sqrt(3)) ** 2 * 1000
        if y - np.sqrt(3) * x > 0:
            penalty += (y - np.sqrt(3) * x) ** 2 * 1000
    return penalty


def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    
    best_flat = None
    best_score = -np.inf
    
    # Initial configurations
    initial_configs = []
    
    # Config 1: structured barycentric grid
    bary_points = [
        (0.15, 0.15, 0.70),
        (0.70, 0.15, 0.15),
        (0.15, 0.70, 0.15),
        (0.50, 0.50, 0.00),
        (0.50, 0.00, 0.50),
        (0.00, 0.50, 0.50),
        (1/3, 1/3, 1/3),
        (0.30, 0.30, 0.40),
        (0.40, 0.30, 0.30),
        (0.30, 0.40, 0.30),
        (0.40, 0.40, 0.20),
    ]
    pts = np.array([[b + 0.5 * c, c * np.sqrt(3) / 2] for (a, b, c) in bary_points])
    initial_configs.append(pts)
    
    # Random initializations
    rng = np.random.RandomState(42)
    for _ in range(8):
        bary = rng.dirichlet(np.ones(3), size=N)
        pts = np.column_stack([bary[:, 1] + 0.5 * bary[:, 2], bary[:, 2] * np.sqrt(3) / 2])
        initial_configs.append(pts)
    
    # Optimize each initial config
    for init_pts in initial_configs:
        if time.time() > deadline - 5:
            break
        
        init_flat = init_pts.flatten().copy()
        
        # Anneal beta from low to high
        betas = [3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
            
            def neg_obj(flat):
                val, _ = _soft_min_area(flat, beta)
                return -(val + _penalty(flat))
            
            result = minimize(
                neg_obj,
                init_flat,
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-9, 'maxls': 50}
            )
            
            if result.x is not None:
                res_pts = result.x.reshape(N, 2)
                res_pts = _clamp_to_triangle(res_pts)
                if _in_triangle(res_pts):
                    min_a = _true_min_area(res_pts)
                    if min_a > best_score:
                        best_score = min_a
                        best_flat = res_pts.flatten().copy()
                init_flat = result.x
    
    if best_flat is None:
        return _clamp_to_triangle(initial_configs[0])
    
    return best_flat.reshape(N, 2)
