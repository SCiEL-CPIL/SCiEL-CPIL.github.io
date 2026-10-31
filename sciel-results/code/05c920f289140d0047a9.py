import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 16
D = 2

def _pairwise_distances(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))

def _score(pts):
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    dmin = d.min()
    dmax = d.max()
    if dmin <= 1e-12:
        return np.inf
    return (dmax / dmin) ** 2

def _objective(pts_flat):
    pts = pts_flat.reshape(N, 2)
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    dmin = d.min()
    dmax = d.max()
    if dmin <= 1e-12:
        return 1e10
    return (dmax / dmin) ** 2

def _soft_objective_sq(pts_flat, beta):
    """Smooth surrogate using squared distances for numerical stability."""
    pts = pts_flat.reshape(N, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, 1)
    d_sq = dists_sq[iu]
    d_sq = np.maximum(d_sq, 1e-12)
    
    max_d_sq = np.max(d_sq)
    # Soft max of d_sq
    soft_max_sq = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
    
    min_d_sq = np.min(d_sq)
    # Soft min of d_sq
    soft_min_sq = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
    
    if soft_min_sq < 1e-10:
        return 1e10
    return soft_max_sq / soft_min_sq

def generate_hex_cluster(n_points=19):
    """Generate the 19-point hexagonal cluster (center + 2 shells)."""
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    
    pts = []
    # Axial coordinates (q, r)
    # We need to cover the 19 points:
    # Center (0,0)
    # 1st shell: 6 points
    # 2nd shell: 12 points
    # Total 19.
    
    # Generate all points with dist <= 2 in axial coords
    # Axial dist: sqrt(q^2 + r^2 + qr)
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Check if in hexagon of radius 2
            # Condition: max(|q|, |r|, |q+r|) <= 2 ?
            # Or simply check distance
            d_axial = np.sqrt(q*q + r*r + q*r)
            if d_axial <= 2 + 1e-6:
                p = q * v1 + r * v2
                pts.append(p)
                
    pts = np.array(pts)
    # Center it
    pts -= pts.mean(axis=0)
    return pts

def _make_ring(n, radius, angle_offset=0.0):
    """Create n points on a ring of given radius."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False) + angle_offset
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_diverse_seeds():
    """Generate a diverse set of initial configurations."""
    seeds = []
    rng = np.random.default_rng(42)
    
    # 1. Hexagonal subsets (from current strategy)
    hex_19 = generate_hex_cluster(19)
    if len(hex_19) == 19:
        indices = np.arange(19)
        for removed_indices in itertools.combinations(indices, 3):
            mask = np.ones(19, dtype=bool)
            mask[list(removed_indices)] = False
            subset = hex_19[mask]
            seeds.append(subset)
    
    # 2. Ring-based configurations (from inspiration)
    # Single ring
    seeds.append(_make_ring(16, 1.0))
    
    # Two rings
    for r_in in [0.5, 0.6, 0.7]:
        for r_out in [1.2, 1.4, 1.6]:
            for offset in [0, np.pi/8, np.pi/4]:
                inner = _make_ring(8, r_in, angle_offset=0.0)
                outer = _make_ring(8, r_out, angle_offset=offset)
                seeds.append(np.vstack([inner, outer]))
    
    # Three rings
    configs = [(6, 5, 5), (4, 6, 6), (5, 6, 5)]
    radii_sets = [
        (0.4, 1.0, 1.6),
        (0.5, 1.1, 1.8),
        (0.3, 0.9, 1.5)
    ]
    for n1, n2, n3 in configs:
        for r1, r2, r3 in radii_sets:
            a1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
            a2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + np.pi/n2
            a3 = np.linspace(0, 2*np.pi, n3, endpoint=False) + np.pi/n3
            p1 = np.column_stack([r1*np.cos(a1), r1*np.sin(a1)])
            p2 = np.column_stack([r2*np.cos(a2), r2*np.sin(a2)])
            p3 = np.column_stack([r3*np.cos(a3), r3*np.sin(a3)])
            seeds.append(np.vstack([p1, p2, p3]))
    
    # Random seeds
    for seed_val in [42, 123, 456, 789, 101112]:
        rng = np.random.RandomState(seed_val)
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2*np.pi, N)
        seeds.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
        
    # Perturbed versions of best seeds
    scored = [(_score(s), s) for s in seeds]
    scored.sort(key=lambda x: x[0])
    for _, s in scored[:5]:
        for _ in range(2):
            perturbed = s + rng.normal(0, 0.1, size=(16, 2))
            seeds.append(perturbed)
            
    return seeds

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_pts = None
    best_score = np.inf
    
    # 1. Generate diverse seeds
    seeds = _generate_diverse_seeds()
    
    # Score all seeds and sort
    scored_seeds = [(_score(s), s) for s in seeds]
    scored_seeds.sort(key=lambda x: x[0])
    
    # Process top seeds with multi-stage optimization
    num_to_optimize = 15
    
    for i in range(min(num_to_optimize, len(scored_seeds))):
        if time.time() > deadline - 15:
            break
            
        curr_score, pts = scored_seeds[i]
        if curr_score < best_score:
            best_score = curr_score
            best_pts = pts.copy()
            
        current_pts = pts.copy()
        
        # Stage 1: Powell on exact objective
        if time.time() < deadline - 12:
            try:
                res1 = minimize(lambda p: _score(p.reshape(N, D)), current_pts.flatten(), 
                                method='Powell', options={'maxiter': 100, 'maxfev': 1000})
                sc1 = _score(res1.x.reshape(N, D))
                if sc1 < best_score:
                    best_score = sc1
                    best_pts = res1.x.reshape(N, D).copy()
                current_pts = res1.x.reshape(N, D)
            except Exception:
                pass
        
        # Stage 2: L-BFGS-B on soft objective (beta=50)
        if time.time() < deadline - 10:
            try:
                res2 = minimize(lambda p: _soft_objective_sq(p, 50.0), current_pts.flatten(),
                                method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
                sc2 = _score(res2.x.reshape(N, D))
                if sc2 < best_score:
                    best_score = sc2
                    best_pts = res2.x.reshape(N, D).copy()
                current_pts = res2.x.reshape(N, D)
            except Exception:
                pass
                
        # Stage 3: Powell again on exact objective
        if time.time() < deadline - 8:
            try:
                res3 = minimize(lambda p: _score(p.reshape(N, D)), current_pts.flatten(), 
                                method='Powell', options={'maxiter': 200, 'maxfev': 2000})
                sc3 = _score(res3.x.reshape(N, D))
                if sc3 < best_score:
                    best_score = sc3
                    best_pts = res3.x.reshape(N, D).copy()
                current_pts = res3.x.reshape(N, D)
            except Exception:
                pass
                
        # Stage 4: L-BFGS-B on higher beta soft objective (beta=200)
        if time.time() < deadline - 5:
            try:
                res4 = minimize(lambda p: _soft_objective_sq(p, 200.0), current_pts.flatten(),
                                method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-14})
                sc4 = _score(res4.x.reshape(N, D))
                if sc4 < best_score:
                    best_score = sc4
                    best_pts = res4.x.reshape(N, D).copy()
                current_pts = res4.x.reshape(N, D)
            except Exception:
                pass
                
        # Stage 5: Final Nelder-Mead polish
        if time.time() < deadline - 2:
            try:
                res5 = minimize(lambda p: _score(p.reshape(N, D)), current_pts.flatten(),
                                method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                sc5 = _score(res5.x.reshape(N, D))
                if sc5 < best_score:
                    best_score = sc5
                    best_pts = res5.x.reshape(N, D).copy()
            except Exception:
                pass

    # 5. Fallback
    if best_pts is None:
        # Simple fallback: hexagonal cluster
        hex_19 = generate_hex_cluster(19)
        if len(hex_19) >= 16:
            return hex_19[:16]
        else:
            x, y = np.meshgrid(np.arange(4), np.arange(4))
            pts = np.column_stack([x.ravel(), y.ravel()]).astype(float)
            pts -= pts.mean(axis=0)
            return pts[:16]
        
    return best_pts

def make_hexagonal_cluster(n, spread=5):
    """Fallback generator."""
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    pts = []
    for i in range(-spread, spread + 1):
        for j in range(-spread, spread + 1):
            p = i * v1 + j * v2
            pts.append(p)
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    return pts[order[:n]]
