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
    Vectorized force-directed step.
    Force = -grad(E) where E is a proxy for non-uniformity.
    We use a combination of:
    1. Repulsion: Points push each other away (inverse square law, short range).
    2. Attraction to 'holes': Points move towards the center of mass of their 'empty space'.
       Approximation: For each point, the 'hole' is in the direction opposite to the weighted sum of its neighbors.
    """
    pts = _normalize(pts.copy())
    # Compute pairwise dot products (cosines of angular distances)
    G = pts @ pts.T
    np.fill_diagonal(G, 0.0)
    
    # Angular distances (approximate for small angles, exact via arccos)
    # Using arccos for accuracy
    d = np.arccos(np.clip(G, -1.0, 1.0))
    
    # Repulsive force: F_rep[i] = sum_{j != i} (d_ij)^(-2) * (pts[i] - pts[j]) / ||pts[i] - pts[j]||
    # To avoid singularities, add epsilon
    d_safe = np.where(d > 1e-6, d, 1e-6)
    inv_d2 = 1.0 / (d_safe ** 2)
    np.fill_diagonal(inv_d2, 0.0)
    
    # Vectorized repulsion:
    # F_rep[i] = sum_j inv_d2[i,j] * (pts[i] - pts[j])
    # Note: (pts[i] - pts[j]) is the vector from j to i.
    # We want to push i away from j.
    # Let's compute the vector differences.
    # diff[i,j] = pts[i] - pts[j]
    # This is (N, N, 3).
    # To save memory, we can compute:
    # F_rep[i] = sum_j w_ij * pts[i] - sum_j w_ij * pts[j]
    # sum_j w_ij * pts[i] = pts[i] * (sum_j w_ij)
    # sum_j w_ij * pts[j] = W @ pts, where W = inv_d2
    
    W = inv_d2
    row_sums = W.sum(axis=1, keepdims=True) # (N, 1)
    
    F_rep = pts * row_sums - (W @ pts)
    
    # Attractive force towards 'holes':
    # The "hole" is where the density is lowest. 
    # A simple proxy: Move each point away from the "center of mass" of its neighbors, 
    # but weighted by their proximity. Actually, repulsion already does this.
    # A better spectral proxy: Minimize the variance of the degrees in the Delaunay graph?
    # Let's stick to a pure repulsion + global centering force.
    # Global centering: Ensure points don't drift to one hemisphere.
    # F_center = -lambda * pts[i] (pulls towards origin? No, that collapses).
    # We want to maximize the minimum distance.
    # Let's add a "spread" force: Move points away from the global centroid (which is ~0 for uniform).
    # Actually, the repulsion with inverse square is quite strong.
    
    # Let's try a different force: 
    # Minimize sum_{i,j} exp(-beta * d_ij^2). This is a Gaussian kernel density.
    # Gradient w.r.t pts[i]: -2 * beta * sum_j exp(-beta * d_ij^2) * (pts[i] - pts[j])
    # This is attractive at long range? No, exp is max at 0. 
    # Wait, minimizing sum of exp(-beta d^2) means we want d to be LARGE? No, exp(-beta d^2) is small for large d.
    # So minimizing this sum makes points CLOSER. That's wrong.
    # We want to MAXIMIZE the minimum distance.
    # So we should MAXIMIZE the sum of exp(-beta d^2)? No, that concentrates them.
    # We want to MINIMIZE the sum of 1/d^2? That's what repulsion does.
    
    # Let's use a "Thomson problem" style force:
    # F_i = sum_{j != i} (pts[i] - pts[j]) / ||pts[i] - pts[j]||^3
    # This is the standard Coulomb repulsion.
    
    # Let's combine:
    # 1. Coulomb repulsion (long range)
    # 2. Short range hard repulsion (to prevent overlap)
    
    # Compute 3D distances
    # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    # dist3 = np.linalg.norm(diff, axis=2)
    # This is (N, N, 3) -> (N, N). Memory heavy for N=130? 130*130*3 ~ 50k floats. Fine.
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist3_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist3_sq, 1.0) # Avoid div by 0
    dist3 = np.sqrt(dist3_sq)
    
    # Coulomb force: F_i = sum_j (pts[i] - pts[j]) / dist3^3
    # Vectorized:
    # F_i = sum_j diff[i,j] / dist3[i,j]^3
    # inv_dist3 = 1.0 / dist3**3
    # inv_dist3[dist3 < 1e-6] = 0.0
    # F = (diff * inv_dist3[:, :, np.newaxis]).sum(axis=1)
    
    # This is O(N^2). For N=130, this is very fast.
    inv_dist3 = 1.0 / (dist3 ** 3)
    inv_dist3[dist3 < 1e-6] = 0.0
    F_coulomb = (diff * inv_dist3[:, :, np.newaxis]).sum(axis=1)
    
    # Add a small "spread" force to encourage uniformity on the sphere surface
    # F_spread = -lambda * pts[i] (if centroid is not 0)
    centroid = np.mean(pts, axis=0)
    F_spread = -0.1 * centroid # Small correction
    
    # Total force
    F = F_coulomb + F_spread
    
    # Normalize force magnitude to prevent large steps
    F_norms = np.linalg.norm(F, axis=1, keepdims=True)
    F_norms[F_norms < 1e-12] = 1.0
    F = F / F_norms
    
    # Update
    new_pts = pts + lr * F
    new_pts = _normalize(new_pts)
    
    return new_pts

def _local_refine(pts, max_iters=50, lr_init=0.02, lr_decay=0.95):
    """
    Fine-tune using the actual covering radius gradient (approximated).
    Uses smooth surrogate: soft-max over facet circumradii to get gradients for ALL facets.
    Moves points of large facets towards their circumcenters with weighted forces.
    """
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_r = _covering_radius_fast(pts)
    lr = lr_init
    
    # Temperature for soft-max (controls smoothness)
    temp = 0.01
    
    for it in range(max_iters):
        if not _time_ok(2.0):
            break
            
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        # Compute circumradius for each facet
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
        
        # Soft-max weights: w_i = exp(r_i / temp) / sum(exp(r_j / temp))
        # This gives higher weight to larger facets
        r_max = np.max(circumradii)
        exp_vals = np.exp((circumradii - r_max) / temp)
        weights = exp_vals / np.sum(exp_vals)
        
        # For each point, compute the weighted average of circumcenters of facets containing it
        # and move it towards that center
        force = np.zeros_like(pts)
        
        for idx, simplex in enumerate(hull.simplices):
            w = weights[idx]
            if w < 1e-8:  # Skip negligible contributions
                continue
            cc = circumcenters[idx]
            for i in simplex:
                # Move point i towards circumcenter of facet idx
                force[i] += w * (cc - pts[i])
        
        # Normalize force per point
        force_norms = np.linalg.norm(force, axis=1, keepdims=True)
        force_norms[force_norms < 1e-12] = 1.0
        force = force / force_norms
        
        # Update
        new_pts = pts + lr * force
        new_pts = _normalize(new_pts)
        
        cr = _covering_radius_fast(new_pts)
        if cr < best_r:
            best_r = cr
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            # If no improvement, try smaller step
            lr *= 0.7
            if lr < 0.0005:
                break
            new_pts = pts + lr * force
            new_pts = _normalize(new_pts)
            cr = _covering_radius_fast(new_pts)
            if cr < best_r:
                best_r = cr
                best_pts = new_pts.copy()
                pts = new_pts
            else:
                lr *= 0.5
                if lr < 0.0005:
                    break
        
        # Decay temperature to make soft-max sharper over time
        temp *= 0.95
        
        # Also decay base learning rate
        lr *= lr_decay
        if lr < 0.0005:
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
    
    # Phase 1: Fast Spectral Optimization
    for r0, p0 in init_scores:
        if not _time_ok(15.0):
            break
            
        pts = p0.copy()
        # Run many fast steps
        for it in range(1000):
            if not _time_ok(10.0):
                break
            pts = _spectral_force_step(pts, lr=0.01, beta=2.0)
            
        # Evaluate
        cr = _covering_radius_fast(pts)
        if cr < best_r:
            best_r = cr
            best_pts = pts.copy()
            
    # Phase 2: Slow Refinement of the Best
    if best_pts is not None and _time_ok(10.0):
        best_pts, best_r = _local_refine(best_pts, max_iters=150, lr_init=0.015, lr_decay=0.985)
        
    # Phase 3: High-precision polish pass
    if best_pts is not None and _time_ok(5.0):
        # Final fine-tuning with very small steps and multiple restarts
        polish_lr = 0.003
        for polish_iter in range(50):
            if not _time_ok(2.0):
                break
            
            try:
                hull = ConvexHull(best_pts)
            except Exception:
                break
                
            # Find top 5 largest facets
            facet_radii = []
            for idx, simplex in enumerate(hull.simplices):
                a, b, c = best_pts[simplex[0]], best_pts[simplex[1]], best_pts[simplex[2]]
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
            top_facets = facet_radii[:5]
            
            # Move points of top facets towards their circumcenters
            for r, idx, cc in top_facets:
                simplex = hull.simplices[idx]
                for i in simplex:
                    best_pts[i] = best_pts[i] + polish_lr * (cc - best_pts[i])
            
            best_pts = _normalize(best_pts)
            cr = _covering_radius_fast(best_pts)
            if cr < best_r:
                best_r = cr
            else:
                polish_lr *= 0.8
                if polish_lr < 0.0005:
                    break
            
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        
    return _normalize(best_pts)

# Test
if __name__ == "__main__":
    t0 = time.time()
    pts = construct()
    print("Shape:", pts.shape)
    print("Covering radius (deg):", np.degrees(_covering_radius_fast(pts)))
    print("Time: %.1f s" % (time.time() - t0))
