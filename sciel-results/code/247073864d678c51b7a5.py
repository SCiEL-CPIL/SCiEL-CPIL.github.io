import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import product

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def hex_to_xy(q, r):
        x = 1.5 * q
        y = (SQRT3 / 2.0) * q + SQRT3 * r
        return x, y

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)]) @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def min_enclosing_hex_side(points):
        """
        Given a set of 2D points, find the minimal side length S of a regular hexagon
        that can contain all points. The hexagon center and orientation are free.
        Returns (S, center, angle_rad).
        """
        # To find the minimal enclosing hexagon, we can observe that for a fixed orientation theta,
        # the minimal center is the midpoint of the "bounding box" of the projections,
        # and the minimal apothem is the max over all axes of the half-range.
        # S = 2 * apothem / sqrt(3).
        
        # Since the hexagon is symmetric, we only need to check orientations in [0, pi/6) (30 deg).
        # Due to symmetry of the regular hexagon, the function is periodic with pi/3.
        
        best_S = float('inf')
        best_center = np.array([0.0, 0.0])
        best_theta = 0.0
        
        # Coarse to fine search over orientation
        # The optimal orientation aligns with the "principal axis" of the point cloud.
        # We will sample orientations.
        
        for theta in np.linspace(0, math.pi/6, 37):
            normals = []
            for k in range(6):
                ang_n = theta + math.pi/6 + k * math.pi/3
                normals.append([math.cos(ang_n), math.sin(ang_n)])
            normals = np.array(normals)
            
            # Project points onto normals
            projs = points @ normals.T # (N, 6)
            max_p = np.max(projs, axis=0)
            min_p = np.min(projs, axis=0)
            
            # For a center c, the constraint is max_p - c.n <= a and c.n - min_p <= a
            # So 2a >= max_p - min_p = range
            # Thus minimal a for this orientation is max_k (range_k / 2)
            ranges = max_p - min_p
            a_req = np.max(ranges) / 2.0
            
            S = 2.0 * a_req / SQRT3
            
            if S < best_S:
                best_S = S
                best_theta = theta
                
                # Find optimal center for this orientation
                # The center c must satisfy:
                # c.n_k <= min_p_k + a_req
                # c.n_k >= max_p_k - a_req
                # This is a linear program or we can solve for c in 2D.
                # Since the hexagon is centrally symmetric, the optimal center is the midpoint
                # of the intersection of the strips?
                # Actually, the feasible region for c is the intersection of 6 strips.
                # The center that minimizes the "max distance" is the Chebyshev center of this polygon.
                # But wait, a_req is fixed for this orientation. We just need ANY valid center.
                # To be safe, we can compute the center as the average of the midpoints of the ranges?
                # No, the midpoints are in the projection space.
                
                # Let's find the center by solving:
                # Minimize norm(c) such that c is in the intersection of strips.
                # Or simply, the center that centers the "bounding box" of the projections.
                # Let's use the fact that the hexagon is convex.
                # The optimal center for a fixed orientation and apothem a is the point that
                # minimizes the maximum violation.
                # Since we have slack (a_req is the max range/2, other ranges are <= a_req),
                # the center can be chosen to center the most constraining axis.
                
                # A simple robust choice: 
                # The feasible region for c is a convex hexagon (intersection of strips).
                # The "center" of this region (in the sense of the hexagon) is the solution to:
                # min max_i |c.n_i - mid_i| subject to c.n_i in [min_i - a_req, max_i + a_req]
                # Actually, since a_req is the max, the region is non-empty.
                
                # Let's just use the centroid of the vertices of the feasible polygon?
                # Or simpler: The optimal center for a regular hexagon enclosing a set of points
                # with fixed orientation is the midpoint of the "width" in the direction of the max projection?
                
                # Let's use a small Nelder-Mead to find the best center for this orientation.
                def center_obj(c):
                    c = np.array(c)
                    dists = np.abs(projs - np.outer(c, normals))
                    # Wait, projs are p.n. We want |p.n - c.n| <= a.
                    # So | (p-c).n | <= a.
                    # (p-c).n = p.n - c.n.
                    # Max over p, n of |p.n - c.n|.
                    # This is the max over k of max_p |p.n_k - c.n_k|.
                    # max_p |p.n_k - c.n_k| = max( max_p(p.n_k) - c.n_k, c.n_k - min_p(p.n_k) )
                    # = max( max_p_k - c.n_k, c.n_k - min_p_k )
                    # We want to minimize the max over k of this quantity.
                    vals = []
                    for k in range(6):
                        v1 = max_p[k] - c[0]*normals[k,0] - c[1]*normals[k,1]
                        v2 = c[0]*normals[k,0] + c[1]*normals[k,1] - min_p[k]
                        vals.append(max(v1, v2))
                    return max(vals)
                
                # Initial guess: centroid of points
                c0 = np.mean(points, axis=0)
                res_c = minimize(center_obj, c0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
                best_center = res_c.x

        return best_S, best_center, best_theta

    def check_disjoint(pos, ang):
        """Check if hexagons are disjoint. Returns True if disjoint."""
        polys = []
        for i in range(N):
            verts = get_hex_verts(pos[i,0], pos[i,1], 1.0, ang[i])
            polys.append(verts)
        
        for i in range(N):
            for j in range(i+1, N):
                # SAT check
                min_sep = float('inf')
                axes = []
                for k in range(6):
                    edge = polys[i][(k+1)%6] - polys[i][k]
                    n = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(n)
                    if norm > 1e-9: axes.append(n/norm)
                for k in range(6):
                    edge = polys[j][(k+1)%6] - polys[j][k]
                    n = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(n)
                    if norm > 1e-9: axes.append(n/norm)
                
                for axis in axes:
                    proj1 = np.dot(polys[i], axis)
                    proj2 = np.dot(polys[j], axis)
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    # Separation: positive if apart
                    sep = min(min1 - max2, min2 - max1)
                    # Actually, if they are apart, max1 < min2 or max2 < min1.
                    # Distance = min( max1 - min2, max2 - min1 ) is negative if apart?
                    # Let's define overlap_depth = max(0, min(max1, max2) - max(min1, min2))
                    # If overlap_depth > 0, they overlap.
                    # If min_sep (max separation) is negative, they overlap.
                    # sep = min(max1, max2) - max(min1, min2).
                    # If sep < 0, overlap.
                    sep = min(max1, max2) - max(min1, min2)
                    if sep < min_sep:
                        min_sep = sep
                
                if min_sep < 1e-6: # Overlapping or touching
                    return False
        return True

    # Generate candidate lattices
    candidates = []
    
    # 1. Honeycomb cluster of 12
    # Center (0,0), ring 1 (6), ring 2 (6 from 12)
    # Total 1 + 6 + 5 = 12?
    # Ring 2 has 12. We need 6 from ring 2.
    # This breaks symmetry.
    
    # Let's generate all subsets of size 12 from a large honeycomb patch.
    patch = []
    for q in range(-4, 5):
        for r in range(-4, 5):
            patch.append((q, r))
    
    # We need to select 12 points that are "connected" and "compact".
    # Instead of brute forcing subsets, let's use known compact shapes.
    
    # Shape 1: 1-6-5 (Center, Ring 1, 5 from Ring 2)
    # The 5 from Ring 2 can be arranged in various ways.
    # Due to symmetry of Ring 2 (12 positions), we just need to pick 5 contiguous ones?
    # Or any 5.
    
    ring2_points = []
    for q, r in [(-2, 0), (-1, -1), (1, -1), (2, 0), (1, 1), (-1, 1), (-3, 1), (-2, 2), (0, 2), (2, 2), (3, 1), (2, -2)]:
        # Wait, let's just generate ring 2 properly.
        pass
    
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,1), (1,2)]
    # Note: Ring 2 coordinates in axial system.
    
    # Let's just generate a large set of valid configurations by random sampling on the lattice.
    # And a few structured ones.
    
    # Structured 1: Full hexagon of radius 1 (7) + 5 from ring 2
    # We will try all combinations of 5 points from ring 2? 12 choose 5 is 792. That's small.
    # But we need to check disjointness. On the lattice, all are disjoint.
    # So we just need to evaluate the enclosing hexagon side.
    
    base_points = [(0,0)] + ring1
    for combo in product(range(len(ring2)), repeat=5):
        # Ensure distinct
        if len(set(combo)) < 5: continue
        pts = base_points + [ring2[i] for i in combo]
        candidates.append(pts)
        
    # Structured 2: No center. 6 from Ring 1 + 6 from Ring 2.
    # 6 choose 6 is 1. (All Ring 1)
    # So 6 from Ring 1 (all) + 6 from Ring 2.
    # 12 choose 6 is 924.
    for combo in product(range(len(ring2)), repeat=6):
        if len(set(combo)) < 6: continue
        pts = ring1 + [ring2[i] for i in combo]
        candidates.append(pts)
        
    # Structured 3: Random compact clusters
    np.random.seed(42)
    for _ in range(20):
        # Pick a random center
        # Grow a cluster
        pts = set()
        current = [(0,0)]
        pts.add((0,0))
        while len(pts) < 12 and len(current) > 0:
            next_current = []
            for p in current:
                q, r = p
                for nq, nr in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
                    np_ = (q+nq, r+nr)
                    if np_ not in pts:
                        pts.add(np_)
                        next_current.append(np_)
            current = next_current
        candidates.append(list(pts))
        
    best_side = float('inf')
    best_pts = None
    best_center = None
    best_theta = None
    
    # Evaluate candidates
    for i, pts in enumerate(candidates):
        if time.time() > deadline - 10:
            break
            
        if len(pts) != 12:
            continue
            
        # Convert to xy
        xy = np.array([hex_to_xy(q, r) for q, r in pts])
        
        # Check disjointness (should be true for lattice)
        # But wait, the task allows rotation. 
        # If we rotate the whole cluster, disjointness is preserved.
        # If we rotate individual hexagons, they might overlap.
        # However, the lattice configuration is the most regular.
        # Let's assume all hexagons are aligned with the lattice (angle 0).
        # This is a valid subset of the problem space.
        
        S, center, theta = min_enclosing_hex_side(xy)
        
        if S < best_side:
            best_side = S
            best_pts = pts
            best_center = center
            best_theta = theta
            
            # Update best
            # We can try to optimize individual angles?
            # If the cluster is tight, rotating individual hexagons might reduce the bounding box?
            # Unlikely to help with the outer hexagon if the lattice is already tight.
            # But it might allow a tighter fit if the outer hexagon shape doesn't match the lattice cluster shape.
            
    # Polish: Optimize individual angles and positions slightly?
    # The lattice positions are fixed.
    # We can optimize the angles of the inner hexagons.
    # And the outer center/angle are already optimized in min_enclosing_hex_side.
    
    # Let's try to optimize the angles of the inner hexagons for the best candidate.
    if best_pts is not None:
        xy = np.array([hex_to_xy(q, r) for q, r in best_pts])
        
        def angle_obj(angles):
            # angles: N radians
            verts_list = []
            for i in range(N):
                verts = get_hex_verts(xy[i,0], xy[i,1], 1.0, np.degrees(angles[i]))
                verts_list.append(verts)
            
            # Check overlap
            # If overlap, penalty
            # We need a fast overlap check.
            # For now, let's just compute the enclosing hexagon side.
            # If overlap occurs, the solution is invalid.
            # But if we start from 0, and perturb slightly, overlap is unlikely if the lattice is safe.
            
            # To be safe, we should check overlap.
            # But it's expensive.
            # Let's assume for now that small rotations don't cause overlap.
            # (Lattice hexagons are separated by a gap? No, they touch.)
            # If they touch, rotation WILL cause overlap.
            # So we must be very careful.
            
            # Actually, regular hexagons in a honeycomb lattice TOUCH.
            # So rotating them will cause overlap.
            # Thus, the only valid rotation for lattice hexagons is 0 (or 60k).
            # So we cannot optimize angles freely.
            
            # Therefore, the lattice configuration with angle 0 is the best we can get from this family.
            
            # Compute side
            S, center, theta = min_enclosing_hex_side(xy)
            return S
            
        # No optimization needed for angles if they must be 0.
        
        # However, what if we move the hexagons slightly?
        # The lattice is rigid in terms of contact.
        # We can't move them without overlap or breaking the lattice.
        # But we can move the WHOLE cluster.
        # That is handled by the outer center optimization.
        
    # Fallback
    if best_pts is None:
        # Use a simple 3x4 rectangle of hexes?
        # Just pick the first candidate or a default.
        best_pts = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (1,1), (0,2), (-1,2), (-2,1)]
        xy = np.array([hex_to_xy(q, r) for q, r in best_pts])
        best_side, best_center, best_theta = min_enclosing_hex_side(xy)

    # Construct return values
    # Inner hexagons are at the lattice positions, angle 0.
    # Outer center and angle are from the optimization.
    
    inner = []
    for q, r in best_pts:
        x, y = hex_to_xy(q, r)
        inner.append([x, y, 0.0])
        
    outer_center = tuple(best_center)
    outer_angle = np.degrees(best_theta)
    
    return inner, outer_center, best_side, outer_angle

if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Best side: {side}")
    print(f"Center: {center}")
    print(f"Angle: {angle}")
