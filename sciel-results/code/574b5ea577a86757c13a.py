# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute all triangle index triplets for vectorized area computation
_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]
_I, _J, _K = _TRIPLETS.T

def _triangle_areas_vectorized(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    a = pts[_I]
    b = pts[_J]
    c = pts[_K]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    return 0.5 * np.abs(cross)

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    areas = _triangle_areas_vectorized(pts)
    if len(areas) == 0:
        return 0.0
    return np.min(areas) / hull_area

def _soft_min_objective(x, beta=500.0):
    """Soft-min of triangle areas / hull area, negative for minimization.
    Higher beta makes it closer to the true min."""
    pts = x.reshape(-1, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 1e6
    areas = _triangle_areas_vectorized(pts)
    if len(areas) == 0:
        return 1e6
    
    # Scale areas to prevent underflow/overflow in exp
    # We want to minimize -soft_min(areas) / hull_area
    # soft_min(areas) approx min(areas)
    # Let's work with -areas to use log-sum-exp for max
    scaled = -areas * beta
    max_scaled = np.max(scaled)
    soft_min = -max_scaled / beta - np.log(np.sum(np.exp(scaled - max_scaled))) / beta
    return -soft_min / hull_area

def _halton_sequence(n, base):
    """Generate first n Halton sequence values in base."""
    result = []
    f = 1.0
    i = 1
    for _ in range(n):
        result.append(f)
        f /= base
        i += 1
        while i % base == 0:
            i //= base
            f /= base
    return result

def _halton_points(n, offset=0.0):
    """Generate n Halton sequence points in [0,1]^2."""
    x = _halton_sequence(n, 2)
    y = _halton_sequence(n, 3)
    pts = np.column_stack([x, y])
    # Center and scale to fit in unit circle roughly
    pts = (pts - 0.5) * 1.5
    return pts + offset

def _ring_init(n_points, n_rings=3):
    """Initialize points in concentric rings."""
    pts = []
    remaining = n_points
    for r in range(n_rings):
        # Distribute points evenly in remaining rings
        # Simple heuristic: more points on outer rings
        if r == n_rings - 1:
            n_in_ring = remaining
            radius = 1.0
        else:
            # Approximate distribution
            n_in_ring = max(1, int(remaining / (n_rings - r)))
            remaining -= n_in_ring
            radius = (r + 1) / n_rings
        
        angles = np.linspace(0, 2 * np.pi, n_in_ring, endpoint=False)
        if r > 0:
            # Rotate rings to avoid alignment
            angles += np.pi / n_in_ring * (r % 2)
        
        ring_pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        pts.append(ring_pts)
    
    return np.vstack(pts)

def _regular_polygon_init(n_points):
    """Initialize points as a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_score = 0.0
    best_pts = None
    
    # Generate diverse initial candidates
    inits = []
    
    # 1. Regular Polygon
    inits.append(_regular_polygon_init(N))
    
    # 2. Halton Sequence
    inits.append(_halton_points(N))
    
    # 3. Concentric Rings
    inits.append(_ring_init(N, n_rings=3))
    
    # 4. Random perturbations of Regular Polygon
    rng = np.random.RandomState(42)
    for i in range(5):
        base = _regular_polygon_init(N)
        noise = rng.normal(0, 0.1, size=base.shape)
        inits.append(base + noise)
    
    # 5. Random perturbations of Halton
    for i in range(5):
        base = _halton_points(N)
        noise = rng.normal(0, 0.1, size=base.shape)
        inits.append(base + noise)
        
    # 6. Random perturbations of Rings
    for i in range(5):
        base = _ring_init(N, n_rings=3)
        noise = rng.normal(0, 0.1, size=base.shape)
        inits.append(base + noise)

    # Optimization Phase
    # We use Nelder-Mead on the soft-min objective.
    # Nelder-Mead is robust to non-differentiable functions and doesn't require gradients.
    
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        x0_flat = x0.flatten()
        
        # Try a few different betas to balance smoothness vs. fidelity
        betas = [100.0, 500.0, 1000.0]
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
                
            try:
                # Nelder-Mead settings
                result = minimize(
                    _soft_min_objective,
                    x0_flat,
                    method='Nelder-Mead',
                    args=(beta,),
                    options={
                        'maxiter': 1000,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
                
                pts = result.x.reshape(-1, 2)
                score = _score(pts)
                
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
                    
            except Exception as e:
                continue
                
    # Final Polish: Use L-BFGS-B on the best candidate found, with a very high beta
    # to approximate the true min more closely.
    if best_pts is not None and time.time() < deadline - 5:
        x0_flat = best_pts.flatten()
        try:
            result = minimize(
                _soft_min_objective,
                x0_flat,
                method='L-BFGS-B',
                args=(2000.0,),
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            pts = result.x.reshape(-1, 2)
            score = _score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            pass

    if best_pts is None:
        best_pts = _regular_polygon_init(N)
        
    return best_pts
# EVOLVE-BLOCK-END
