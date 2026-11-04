"""
Optimized sphere covering with 130 points via Spectral Uniformity & Force-Directed Optimization.

Strategy:
1. Avoid ConvexHull during optimization (too slow/fragile).
2. Use a spectral energy proxy: minimize variance of pairwise distances to enforce uniformity.
3. Use a vectorized force-directed method:
   - Repulsive forces between all points (short range) to prevent clustering.
   - Attractive forces towards the "holes" (regions of high local density deficit).
   - This is approximated by moving each point away from its nearest neighbors and towards the center of mass of the "empty space" relative to its Voronoi cell (approximated by inverse distance weighting of neighbors).
4. Final polish: A few iterations of a precise local search using the actual covering radius (ConvexHull) on the best candidate found by the fast proxy.
"""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130
_DEADLINE = time.time() + 140  # 10s safety margin

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _time_ok(margin=5.0):
    return time.time() < _DEADLINE - margin

def _covering_radius(pts):
    """Compute the covering radius in radians via convex hull facet circumcenters."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi / 2

    max_r = 0.0
    for simplex in hull.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        v1, v2 = b - a, c - a
        cross = np.cross(v1, v2)
        cnorm = np.linalg.norm(cross)
        if cnorm < 1e-12:
            continue
        x = cross / cnorm
        centroid = (a + b + c) / 3.0
        if np.dot(x, centroid) < 0:
            x = -x
        dot_val = np.clip(np.dot(x, a), -1.0, 1.0)
        r = np.arccos(dot_val)
        if r > max_r:
            max_r = r
    return max_r

def _covering_radius_fast(pts):
    """Alias for _covering_radius."""
    return _covering_radius(pts)

def _fibonacci_spiral(n, jitter=0.0, seed=0):
    """Fibonacci spiral on the unit sphere with optional jitter."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    if jitter > 0:
        i = i + rng.randn(n) * jitter
        i = np.clip(i, 0.5, n - 0.5)
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _spectral_force_step(pts, lr=0.05, beta=2.0):
    """
    Vectorized force-directed step using Coulomb repulsion on the sphere.
    Maximizes minimum pairwise distance via inverse-cube repulsion.
    """
    pts = _normalize(pts.copy())
    
    # Compute 3D distance vectors
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist3_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist3_sq, 1.0)
    dist3 = np.sqrt(dist3_sq)
    
    # Coulomb force: F_i = sum_j (pts[i] - pts[j]) / dist3^3
    inv_dist3 = 1.0 / (dist3 ** 3)
    inv_dist3[dist3 < 1e-6] = 0.0
    F_coulomb = (diff * inv_dist3[:, :, np.newaxis]).sum(axis=1)
    
    # Small centering correction to prevent drift to one hemisphere
    centroid = np.mean(pts, axis=0)
    F_spread = -0.05 * centroid
    
    F = F_coulomb + F_spread
    
    # Normalize force magnitude for stable steps
    F_norms = np.linalg.norm(F, axis=1, keepdims=True)
    F_norms[F_norms < 1e-12] = 1.0
    F = F / F_norms
    
    new_pts = pts + lr * F
    new_pts = _normalize(new_pts)
    
    return new_pts

def _local_refine(pts, max_iters=200, lr_init=0.02, lr_decay=0.98):
    """
    Fine-tune using smooth surrogate: soft-max over facet circumradii.
    Moves points of large facets towards their circumcenters with weighted forces.
    """
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_r = _covering_radius_fast(pts)
    lr = lr_init
    temp = 0.005  # Start sharper
    stall_count = 0
    
    for it in range(max_iters):
        if not _time_ok(3.0):
            break
            
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        n_facets = len(hull.simplices)
        if n_facets == 0:
            break
            
        circumradii = np.zeros(n_facets)
        circumcenters = np.zeros((n_facets, 3))
        
        for idx, simplex in enumerate(hull.simplices):
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            v1, v2 = b - a, c - a
            cross = np.cross(v1, v2)
            cnorm = np.linalg.norm(cross)
            if cnorm < 1e-12:
                circumradii[idx] = 0.0
                continue
            x = cross / cnorm
            centroid = (a + b + c) / 3.0
            if np.dot(x, centroid) < 0:
                x = -x
            dot_val = np.clip(np.dot(x, a), -1.0, 1.0)
            r = np.arccos(dot_val)
            circumradii[idx] = r
            circumcenters[idx] = x
        
        # Soft-max weights
        r_max = np.max(circumradii)
        exp_vals = np.exp((circumradii - r_max) / temp)
        weights = exp_vals / np.sum(exp_vals)
        
        # Accumulate forces
        force = np.zeros_like(pts)
        for idx, simplex in enumerate(hull.simplices):
            w = weights[idx]
            if w < 1e-10:
                continue
            cc = circumcenters[idx]
            for i in simplex:
                force[i] += w * (cc - pts[i])
        
        force_norms = np.linalg.norm(force, axis=1, keepdims=True)
        force_norms[force_norms < 1e-12] = 1.0
        force = force / force_norms
        
        # Try the move
        new_pts = pts + lr * force
        new_pts = _normalize(new_pts)
        cr = _covering_radius_fast(new_pts)
        
        if cr < best_r:
            best_r = cr
            best_pts = new_pts.copy()
            pts = new_pts
            stall_count = 0
        else:
            stall_count += 1
            # Adaptive: reduce lr more aggressively when stalling
            lr *= (0.8 if stall_count < 3 else 0.6)
            if lr < 0.0003:
                break
        
        # Decay temperature gradually
        temp *= 0.995
        
        # Decay base learning rate
        lr *= lr_decay
        if lr < 0.0003:
            break
            
    return best_pts, best_r

def construct():
    """
    1. Generate diverse initial configs.
    2. Run fast spectral force-directed optimization (no hull).
    3. Refine the best candidate with slow hull-based local search.
    """
    start_time = time.time()
    best_pts = None
    best_r = np.pi
    
    # Initial configs
    inits = []
    for seed in [42, 123, 7, 999, 31415, 271828, 161803, 111, 222, 333]:
        inits.append(_fibonacci_spiral(N, jitter=0.05, seed=seed))
    inits.append(_fibonacci_spiral(N, jitter=0.0, seed=0))
    rng = np.random.RandomState(555)
    inits.append(_normalize(rng.randn(N, 3)))
    
    # Sort by initial covering radius (fast check)
    init_scores = []
    for p in inits:
        try:
            r = _covering_radius_fast(p)
        except Exception:
            r = np.pi
        init_scores.append((r, p))
    init_scores.sort(key=lambda x: x[0])
    
    # Phase 1: Fast Spectral Optimization with adaptive stopping
    for r0, p0 in init_scores:
        if not _time_ok(12.0):
            break
            
        pts = p0.copy()
        prev_cr = _covering_radius_fast(pts)
        # Adaptive step count: stop when improvement plateaus
        for it in range(500):
            if not _time_ok(8.0):
                break
            pts = _spectral_force_step(pts, lr=0.008, beta=2.0)
            
            # Check every 50 iterations
            if (it + 1) % 50 == 0:
                cr_check = _covering_radius_fast(pts)
                if abs(cr_check - prev_cr) < 1e-5:
                    break
                prev_cr = cr_check
        
        cr = _covering_radius_fast(pts)
        if cr < best_r:
            best_r = cr
            best_pts = pts.copy()
            
    # Phase 2: Slow Refinement of the Best
    if best_pts is not None and _time_ok(8.0):
        best_pts, best_r = _local_refine(best_pts, max_iters=200, lr_init=0.012, lr_decay=0.985)
        
    # Phase 3: High-precision polish pass with revert on failure
    if best_pts is not None and _time_ok(4.0):
        polish_lr = 0.002
        best_polish_pts = best_pts.copy()
        best_polish_r = best_r
        stall = 0
        for polish_iter in range(40):
            if not _time_ok(2.0):
                break
            
            try:
                hull = ConvexHull(best_polish_pts)
            except Exception:
                break
                
            # Find top 3 largest facets
            facet_radii = []
            for idx, simplex in enumerate(hull.simplices):
                a, b, c = best_polish_pts[simplex[0]], best_polish_pts[simplex[1]], best_polish_pts[simplex[2]]
                v1, v2 = b - a, c - a
                cross = np.cross(v1, v2)
                cnorm = np.linalg.norm(cross)
                if cnorm < 1e-12:
                    continue
                x = cross / cnorm
                centroid = (a + b + c) / 3.0
                if np.dot(x, centroid) < 0:
                    x = -x
                dot_val = np.clip(np.dot(x, a), -1.0, 1.0)
                r = np.arccos(dot_val)
                facet_radii.append((r, idx, x))
            
            if not facet_radii:
                break
                
            facet_radii.sort(key=lambda x: -x[0])
            top_facets = facet_radii[:3]
            
            # Try move
            trial_pts = best_polish_pts.copy()
            for r, idx, cc in top_facets:
                simplex = hull.simplices[idx]
                for i in simplex:
                    trial_pts[i] = trial_pts[i] + polish_lr * (cc - trial_pts[i])
            
            trial_pts = _normalize(trial_pts)
            cr = _covering_radius_fast(trial_pts)
            
            if cr < best_polish_r:
                best_polish_r = cr
                best_polish_pts = trial_pts
                best_pts = trial_pts.copy()
                best_r = cr
                stall = 0
            else:
                stall += 1
                polish_lr *= 0.7
                if stall > 5 or polish_lr < 0.0002:
                    break
            
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        
    # Final validation: ensure all points are on the unit sphere
    return _normalize(best_pts)

# Test
if __name__ == "__main__":
    t0 = time.time()
    pts = construct()
    print("Shape:", pts.shape)
    print("Covering radius (deg):", np.degrees(_covering_radius_fast(pts)))
    print("Time: %.1f s" % (time.time() - t0))
