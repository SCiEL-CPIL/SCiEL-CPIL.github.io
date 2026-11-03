"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START
def _normalize(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return pts / n

def _covering_radius(pts):
    """Compute covering radius in radians via convex hull facets."""
    try:
        hull = ConvexHull(pts)
        max_r = 0.0
        for simplex in hull.simplices:
            # Facet defined by 3 points on unit sphere
            p0, p1, p2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            # Normal to the facet (outward)
            n = np.cross(p1 - p0, p2 - p0)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n = n / nn
            # The farthest point on the sphere from this facet's plane
            # is in the direction of n (outward normal)
            # Distance from facet center to that point
            # Facet circumcenter: for a triangle on unit sphere, the circumcenter
            # of the spherical triangle is along the normal direction
            # The angular distance from any point on the facet to the antipodal normal
            # Actually, the covering radius contribution from this facet is the
            # angular distance from the facet's circumcenter (on sphere) to the
            # farthest vertex, which equals the circumradius of the spherical triangle.
            # For a triangle with vertices p0, p1, p2 on unit sphere, the circumradius
            # (spherical) can be computed as follows:
            # The circumcenter of the planar triangle projected onto the sphere
            # gives the center of the spherical cap that covers the triangle.
            # The angular radius is arccos(dot(circumcenter, vertex))
            
            # Compute circumcenter of triangle p0, p1, p2 in 3D (not on sphere)
            # Then normalize to get point on sphere
            a = p1 - p0
            b = p2 - p0
            c = np.cross(a, b)
            # Solve for circumcenter: C = p0 + s*a + t*b such that |C-p0|^2 = |C-p1|^2 = |C-p2|^2
            # |C-p1|^2 = |C-p0-a|^2 = |C-p0|^2 - 2*a.(C-p0) + |a|^2
            # So: -2*a.(C-p0) + |a|^2 = 0 => a.(C-p0) = |a|^2/2
            # Similarly: b.(C-p0) = |b|^2/2
            # Let d = C - p0 = s*a + t*b
            # a.(s*a + t*b) = |a|^2/2 => s*|a|^2 + t*(a.b) = |a|^2/2
            # b.(s*a + t*b) = |b|^2/2 => s*(a.b) + t*|b|^2 = |b|^2/2
            aa = np.dot(a, a)
            bb = np.dot(b, b)
            ab = np.dot(a, b)
            det = aa * bb - ab * ab
            if abs(det) < 1e-15:
                continue
            s = (bb * 0.5 * aa - ab * 0.5 * bb) / det  # Wait, let me redo
            # s*aa + t*ab = 0.5*aa
            # s*ab + t*bb = 0.5*bb
            # From first: s = (0.5*aa - t*ab)/aa
            # Substitute: (0.5*aa - t*ab)*ab/aa + t*bb = 0.5*bb
            # 0.5*ab - t*ab^2/aa + t*bb = 0.5*bb
            # t*(bb - ab^2/aa) = 0.5*(bb - ab)
            # t = 0.5*(bb - ab) / (bb - ab^2/aa) = 0.5*aa*(bb - ab) / (aa*bb - ab^2)
            t = 0.5 * aa * (bb - ab) / det
            s = (0.5 * aa - t * ab) / aa
            d = s * a + t * b
            C = p0 + d
            # Normalize C to get point on sphere
            cn = np.linalg.norm(C)
            if cn < 1e-12:
                continue
            C_unit = C / cn
            # Angular distance from C_unit to each vertex
            r0 = np.arccos(np.clip(np.dot(C_unit, p0), -1, 1))
            r1 = np.arccos(np.clip(np.dot(C_unit, p1), -1, 1))
            r2 = np.arccos(np.clip(np.dot(C_unit, p2), -1, 1))
            facet_r = max(r0, r1, r2)
            if facet_r > max_r:
                max_r = facet_r
        return max_r
    except Exception:
        return np.pi  # invalid

def _lloyd_step(pts, n_iters=10):
    """Lloyd's algorithm: move each point to centroid of its Voronoi cell."""
    pts = _normalize(pts.copy())
    for _ in range(n_iters):
        try:
            hull = ConvexHull(pts)
            # For each point, find its Voronoi cell = intersection of half-spaces
            # from adjacent facets. Simpler: use the facet-based approach.
            # Actually, for Lloyd on sphere, we can use the fact that the Voronoi
            # cell of point i is the set of points closer to i than to any other.
            # The centroid of the Voronoi cell can be approximated by averaging
            # the circumcenters of adjacent facets.
            
            # Alternative simpler approach: for each point, find all facets that
            # contain it, compute the spherical centroid of those facets' circumcenters
            # weighted by facet area, and move toward that centroid.
            
            # Simpler still: just do a gradient-free local search
            # For each point, find the farthest facet it's part of and move away
            # from the circumcenter of that facet.
            
            new_pts = pts.copy()
            for i in range(N):
                # Find facets containing point i
                facet_circumcenters = []
                for simplex in hull.simplices:
                    if i in simplex:
                        p0, p1, p2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                        a = p1 - p0
                        b = p2 - p0
                        aa = np.dot(a, a)
                        bb = np.dot(b, b)
                        ab = np.dot(a, b)
                        det = aa * bb - ab * ab
                        if abs(det) < 1e-15:
                            continue
                        t = 0.5 * aa * (bb - ab) / det
                        s = (0.5 * aa - t * ab) / aa
                        d = s * a + t * b
                        C = p0 + d
                        cn = np.linalg.norm(C)
                        if cn < 1e-12:
                            continue
                        facet_circumcenters.append(C / cn)
                
                if facet_circumcenters:
                    # Average the circumcenters (spherical mean)
                    centroid = np.mean(facet_circumcenters, axis=0)
                    cn = np.linalg.norm(centroid)
                    if cn > 1e-12:
                        centroid = centroid / cn
                        # Move point toward centroid
                        new_pts[i] = 0.7 * pts[i] + 0.3 * centroid
                        new_pts[i] = new_pts[i] / np.linalg.norm(new_pts[i])
        except Exception:
            pass
        pts = _normalize(new_pts)
    return pts

def _nelder_mead_polish(pts, max_iters=200, tol=1e-6):
    """Polish the configuration using Nelder-Mead on the worst points."""
    pts = _normalize(pts.copy())
    best_r = _covering_radius(pts)
    
    for it in range(max_iters):
        # Find the worst point (part of the largest facet)
        try:
            hull = ConvexHull(pts)
            worst_facet = None
            worst_r = 0.0
            for simplex in hull.simplices:
                p0, p1, p2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                a = p1 - p0
                b = p2 - p0
                aa = np.dot(a, a)
                bb = np.dot(b, b)
                ab = np.dot(a, b)
                det = aa * bb - ab * ab
                if abs(det) < 1e-15:
                    continue
                t = 0.5 * aa * (bb - ab) / det
                s = (0.5 * aa - t * ab) / aa
                d = s * a + t * b
                C = p0 + d
                cn = np.linalg.norm(C)
                if cn < 1e-12:
                    continue
                C_unit = C / cn
                r0 = np.arccos(np.clip(np.dot(C_unit, p0), -1, 1))
                r1 = np.arccos(np.clip(np.dot(C_unit, p1), -1, 1))
                r2 = np.arccos(np.clip(np.dot(C_unit, p2), -1, 1))
                facet_r = max(r0, r1, r2)
                if facet_r > worst_r:
                    worst_r = facet_r
                    worst_facet = simplex
        except Exception:
            break
        
        if worst_facet is None:
            break
        
        # Try small perturbations of the points in the worst facet
        improved = False
        for idx in worst_facet:
            for scale in [0.01, 0.005, 0.001]:
                for axis in range(3):
                    for sign in [-1, 1]:
                        trial = pts.copy()
                        trial[idx][axis] += sign * scale
                        trial[idx] = trial[idx] / np.linalg.norm(trial[idx])
                        r = _covering_radius(trial)
                        if r < best_r - 1e-9:
                            pts = trial
                            best_r = r
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return pts, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_r = _covering_radius(pts)
    
    # Phase 1: Lloyd's algorithm
    pts = _lloyd_step(pts, n_iters=50)
    r = _covering_radius(pts)
    if r < best_r:
        best_pts = pts.copy()
        best_r = r
    
    # Check time
    if time.time() > deadline - 30:
        return best_pts
    
    # Phase 2: Nelder-Mead polishing
    pts, r = _nelder_mead_polish(pts, max_iters=100)
    if r < best_r:
        best_pts = pts.copy()
        best_r = r
    
    # Check time
    if time.time() > deadline - 20:
        return best_pts
    
    # Phase 3: More Lloyd iterations
    pts = _lloyd_step(best_pts, n_iters=30)
    r = _covering_radius(pts)
    if r < best_r:
        best_pts = pts.copy()
        best_r = r
    
    # Check time
    if time.time() > deadline - 10:
        return best_pts
    
    # Phase 4: Final polishing
    pts, r = _nelder_mead_polish(best_pts, max_iters=50)
    if r < best_r:
        best_pts = pts.copy()
        best_r = r
    
    return best_pts
# EVOLVE-BLOCK-END
