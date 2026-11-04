import numpy as np
import itertools
from scipy.optimize import minimize

def construct():
    N = 12
    
    # Generate initial centers from a hexagonal tiling
    # This provides a good starting point
    centers = []
    # A 12-hexagon cluster often looks like a 3x3 grid with corners missing or similar.
    # Let's use a compact arrangement.
    # Coordinates for a hexagonal grid (axial coordinates q, r)
    # q in [-1, 1], r in [-1, 1] gives 9 hexes. Need 12.
    # Let's try a 4x3 rectangular block or similar, then optimize.
    # Or simply use the existing grid logic which creates 12 points.
    for i in range(-1, 2): # -1, 0, 1
        for j in range(-1, 3): # -1, 0, 1, 2
            x = i * np.sqrt(3)
            y = 2 * j + (1 if i % 2 != 0 else 0)
            centers.append([x, y])
            
    centers = np.array(centers)
    centers -= np.mean(centers, axis=0)
    
    # Initial angles: 0
    angles = np.zeros(N)
    
    # Variables: x1, y1, a1, x2, y2, a2, ...
    # Total 36 variables
    x0 = np.zeros(36)
    for i in range(N):
        x0[i*3] = centers[i, 0]
        x0[i*3+1] = centers[i, 1]
        x0[i*3+2] = 0.0 # angle
        
    # Precompute vertices of a unit hexagon (pointy top? or flat top?)
    # The code's calc_L assumes outer is flat-top.
    # Inner hexagons can be any orientation.
    # Let's define inner hexagon vertices relative to its center.
    # Standard unit hexagon (side 1) vertices:
    # If we align it with the outer (flat top), its vertices are at 0, 60...
    # But we can rotate it.
    # Let's use the same vertex definition as before:
    # angles = [90, 150, 210, 270, 330, 30] -> Pointy top?
    # Wait, 90 is top. So this is pointy-top.
    # If inner is pointy-top and outer is flat-top, they are rotated 30 deg relative to each other.
    # We will allow angle optimization.
    
    base_angles = np.deg2rad([90, 150, 210, 270, 330, 30])
    base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    def get_inner_verts(params):
        # params: [x1, y1, a1, x2, y2, a2, ...]
        verts_list = []
        for i in range(N):
            cx = params[i*3]
            cy = params[i*3+1]
            ang = params[i*3+2]
            c, s = np.cos(ang), np.sin(ang)
            R = np.array([[c, -s], [s, c]])
            v = (R @ base_verts.T).T + np.array([cx, cy])
            verts_list.append(v)
        return np.array(verts_list) # Shape (N, 6, 2)
        
    def calc_L_from_verts(verts):
        # verts: (N, 6, 2)
        # Outer hexagon: flat-top, centered at (0,0), side L.
        # Inequalities for point (x,y) to be inside flat-top hex of side L:
        # |y| <= L * sqrt(3)/2
        # |x| <= L
        # |y + x/sqrt(3)| <= L * sqrt(3)/2  ? No.
        
        # Let's derive the 6 inequalities for flat-top hex (vertices at 0, 60, ...):
        # 1. Right edge (x = L): No, vertex is at L. Edge is slanted.
        # Vertices: (L,0), (L/2, L*sqrt(3)/2)
        # Edge 1: between (L,0) and (L/2, L*sqrt(3)/2).
        # Line: y - 0 = (L*sqrt(3)/2 - 0) / (L/2 - L) * (x - L) = -sqrt(3) (x - L)
        # y = -sqrt(3)x + L*sqrt(3)
        # sqrt(3)x + y = L*sqrt(3)
        # Constraint: sqrt(3)x + y <= L*sqrt(3)  => L >= x + y/sqrt(3)
        
        # Edge 2: between (L/2, L*sqrt(3)/2) and (-L/2, L*sqrt(3)/2).
        # Horizontal line y = L*sqrt(3)/2.
        # Constraint: y <= L*sqrt(3)/2 => L >= 2y/sqrt(3)
        
        # Edge 3: between (-L/2, L*sqrt(3)/2) and (-L, 0).
        # Symmetric to Edge 1.
        # -sqrt(3)x + y = L*sqrt(3) => L >= -x + y/sqrt(3)
        
        # Edge 4: Bottom left.
        # -sqrt(3)x - y = L*sqrt(3) => L >= -x - y/sqrt(3)
        
        # Edge 5: Bottom horizontal.
        # y = -L*sqrt(3)/2 => L >= -2y/sqrt(3)
        
        # Edge 6: Bottom right.
        # sqrt(3)x - y = L*sqrt(3) => L >= x - y/sqrt(3)
        
        # So L_req for a point (x,y) is:
        # max( x + y/sqrt(3), 2y/sqrt(3), -x + y/sqrt(3), -x - y/sqrt(3), -2y/sqrt(3), x - y/sqrt(3) )
        # This can be simplified to:
        # max( |x| + |y|/sqrt(3) , 2|y|/sqrt(3) ) ? No.
        # Let's just compute the 6 terms.
        
        x = verts[:, :, 0]
        y = verts[:, :, 1]
        
        term1 = x + y / np.sqrt(3)
        term2 = 2 * y / np.sqrt(3)
        term3 = -x + y / np.sqrt(3)
        term4 = -x - y / np.sqrt(3)
        term5 = -2 * y / np.sqrt(3)
        term6 = x - y / np.sqrt(3)
        
        L_points = np.max(np.stack([term1, term2, term3, term4, term5, term6]), axis=0)
        return np.max(L_points)

    def sat_min_distance(v1, v2):
        """
        Calculate the minimum distance between two convex polygons v1 and v2.
        Returns negative value if they overlap (penetration depth).
        Uses Separating Axis Theorem.
        """
        min_dist = float('inf')
        
        # Get all edge normals for both polygons
        # For a convex polygon, the separating axes are the normals to the edges.
        
        def get_normals(verts):
            normals = []
            n = len(verts)
            for i in range(n):
                p1 = verts[i]
                p2 = verts[(i+1) % n]
                edge = p2 - p1
                # Normal is perpendicular to edge
                # For CCW polygon, normal is (edge_y, -edge_x) ? 
                # Let's just use both directions to be safe, or normalize.
                norm = np.array([edge[1], -edge[0]])
                norm_len = np.linalg.norm(norm)
                if norm_len > 1e-10:
                    normals.append(norm / norm_len)
            return normals
        
        normals1 = get_normals(v1)
        normals2 = get_normals(v2)
        
        all_normals = normals1 + normals2
        
        for n in all_normals:
            # Project both polygons onto the normal
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for overlap on this axis
            if max1 < min2 or max2 < min1:
                # They are separated on this axis
                # Distance is the gap
                dist = max(min2 - max1, min1 - max2)
                min_dist = min(min_dist, dist)
            else:
                # They overlap on this axis. 
                # For SAT, if they overlap on ALL axes, they intersect.
                # The "distance" in the context of a penalty is usually 0 if separated, 
                # and we need a measure of overlap if not.
                # Standard SAT returns boolean. To get penetration depth, it's more complex.
                # However, for a penalty, we can approximate the overlap depth.
                # If they overlap on an axis, it doesn't mean they intersect globally.
                # If they intersect, min_dist should be negative.
                # A simple approximation for penalty: 
                # If no separating axis is found, they intersect.
                # We can return a large negative number? No, we need differentiability.
                # Let's refine: 
                # If they are separated on any axis, min_dist is the min gap.
                # If they are NOT separated on any axis, they intersect.
                # We need a continuous function.
                # Let's use the minimum "separation" where separation can be negative.
                # Separation on axis n = min(max1 - min2, max2 - min1) ? No.
                # Gap = min(min2 - max1, min1 - max2) if disjoint.
                # If they overlap, this gap is negative.
                # Let's calculate the "separation" s = min( min2 - max1, min1 - max2 ).
                # If s > 0, they are disjoint, distance is s.
                # If s < 0, they overlap on this axis.
                # The true distance is min over all axes of the gap IF disjoint.
                # If they intersect, the distance is 0 (or negative penetration).
                # For a penalty, we want to push them apart.
                # If they intersect, we want a large penalty.
                # Let's define penalty_axis = max(0, -min_gap) ?
                
                # Actually, if they intersect, there is NO separating axis.
                # So `min_dist` will remain inf if we only update on separation.
                # We need to detect intersection.
                # If we finish the loop and min_dist is inf, they intersect.
                # In that case, we should return a negative value representing overlap.
                # How to estimate overlap? 
                # A common trick: return -epsilon or a fixed penalty? 
                # Better: Use the fact that if they intersect, the "minimum distance" is 0.
                # But we need to know HOW MUCH they overlap to guide the optimizer.
                # Approximation: If they intersect, return a value based on the center distance?
                # Or, we can calculate the minimum translation vector (MTV) magnitude.
                
                # Let's stick to a simpler robust penalty:
                # If they intersect, return -1.0 (or some constant) to indicate strong penalty.
                # But this is not differentiable.
                
                # Alternative: Use a smooth approximation of the distance function.
                # Distance between two convex sets is convex.
                # We can use the fact that distance is 0 if intersecting.
                # For optimization, a penalty of `max(0, 1 - dist)^2` is common.
                # If dist < 1, penalty > 0.
                # But if they intersect, dist is 0 (or undefined in simple SAT).
                
                # Let's use a different strategy for the penalty function itself.
                # Instead of exact SAT distance, use a "soft" collision detection.
                # If center distance < threshold, apply penalty.
                # But the directive asks for "smooth, differentiable penalty function derived from the pairwise minimum distances".
                
                # Let's implement a proper penetration depth estimation.
                # If no separating axis, they intersect.
                # We can estimate the overlap by projecting the centers?
                # Let's just return a large negative number if they intersect?
                # No, let's use the "min gap" logic but allow it to be negative.
                # Gap on axis n: 
                # If max1 < min2: gap = min2 - max1 (positive)
                # If max2 < min1: gap = min1 - max2 (positive)
                # Else: gap = min(max1 - min2, max2 - min1) (negative)
                # The true distance is min(gap) over all axes ONLY IF the min gap is positive.
                # If any gap is negative, they might still be disjoint on other axes? No, if they are disjoint, ALL axes have positive gap.
                # If they are intersecting, SOME axes have negative gap.
                # So, min_gap = min over all axes of (gap_on_axis).
                # If min_gap > 0, distance = min_gap.
                # If min_gap < 0, they intersect. The "penetration" is roughly -min_gap?
                # This is an approximation but works well for convex shapes.
                
                # Let's compute gap for each axis and take the minimum.
                pass # We will implement this in the loop below.

        # Re-implementing the loop to track min_gap which can be negative
        min_gap = float('inf')
        
        for n in all_normals:
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Calculate the "gap" on this axis.
            # If they are separated, gap is positive.
            # If they overlap, gap is negative.
            # gap = min( min2 - max1, min1 - max2 )
            # If max1 < min2, min2 - max1 > 0, min1 - max2 < 0. Min is negative? No.
            # If max1 < min2, the interval [min1, max1] is to the left of [min2, max2].
            # Distance is min2 - max1.
            # Formula: 
            # If max1 < min2: gap = min2 - max1
            # If max2 < min1: gap = min1 - max2
            # Else: gap = min(max1 - min2, max2 - min1) # This will be negative
            
            if max1 < min2:
                gap = min2 - max1
            elif max2 < min1:
                gap = min1 - max2
            else:
                gap = min(max1 - min2, max2 - min1)
                
            min_gap = min(min_gap, gap)
            
        # If min_gap is inf, something went wrong (shouldn't happen with valid polygons)
        if min_gap == float('inf'):
            return 0.0
            
        return min_gap

    def overlap_penalty(params):
        verts = get_inner_verts(params)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = sat_min_distance(verts[i], verts[j])
                # dist is positive if separated, negative if overlapping.
                # We want to penalize negative dist.
                # Use a smooth penalty: max(0, -dist)^2
                # To make it differentiable at 0, we can use a soft max or just rely on the optimizer.
                # max(0, x)^2 is not differentiable at x=0.
                # Use 0.5 * (sqrt(x^2 + eps^2) - eps)^2 ? No.
                # Let's use: if dist < 0: penalty += (-dist)**2
                # This is standard. L-BFGS-B with numerical gradients can handle the kink if we are careful, 
                # or we can use a smooth approximation: 
                # soft_max(x, beta) = logsumexp? 
                # Let's stick to the hard penalty but ensure the optimizer takes small steps.
                if dist < 0:
                    penalty += (-dist)**2
        return penalty

    def objective(params):
        verts = get_inner_verts(params)
        L = calc_L_from_verts(verts)
        pen = overlap_penalty(params)
        # Weight for penalty
        # If L is around 5-6, and overlap is small, we need a high weight.
        return L + 100.0 * pen
        
    # Optimize
    # L-BFGS-B is good for bound-constrained problems, but we have no bounds here.
    # Powell is derivative-free and robust.
    # Let's try L-BFGS-B first.
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-6})
    
    best_params = res.x
    best_L = calc_L_from_verts(get_inner_verts(best_params))
    
    inner = []
    for i in range(N):
        cx = best_params[i*3]
        cy = best_params[i*3+1]
        ang_deg = np.rad2deg(best_params[i*3+2])
        inner.append([cx, cy, ang_deg])
        
    return inner, (0.0, 0.0), best_L, 0.0
