"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon."""
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def compute_S_for_container(vertices_list, ocx, ocy, oangle):
        """Compute the minimum side length S of the outer hexagon to contain all vertices."""
        cos_a = np.cos(oangle)
        sin_a = np.sin(oangle)
        S = 0.0
        apothem = np.sqrt(3) / 2
        for verts in vertices_list:
            for v in verts:
                lx = v[0] - ocx
                ly = v[1] - ocy
                rx = lx * cos_a + ly * sin_a
                ry = -lx * sin_a + ly * cos_a
                for k in range(6):
                    theta_n = np.pi / 6 + k * np.pi / 3
                    proj = rx * np.cos(theta_n) + ry * np.sin(theta_n)
                    S = max(S, abs(proj))
        return S / apothem

    def sat_penalty(v1, v2):
        """
        Compute a smooth, differentiable penalty for overlap between two convex polygons.
        Uses the sum of squared penetrations along all separating axes.
        """
        polys = [v1, v2]
        penalty = 0.0
        for poly in polys:
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                # Normalize normal for consistent scaling? 
                # Actually, for penalty magnitude consistency, normalization helps.
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal = normal / norm
                
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                
                # Depth of penetration along this axis
                # If proj1.max() > proj2.min(), they overlap on this axis
                # Penetration = proj1.max() - proj2.min()
                # But we must consider both directions?
                # If they overlap, proj1.max() > proj2.min() AND proj2.max() > proj1.min()
                # The "gap" is min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())? No.
                
                # Let's use:
                # overlap_1 = max(0, proj1.max() - proj2.min())
                # overlap_2 = max(0, proj2.max() - proj1.min())
                # If they are disjoint, at least one of these will be 0? No.
                # If disjoint, either proj1.max() < proj2.min() (overlap_1=0, overlap_2>0? No, proj2.max() > proj1.min() is true, so overlap_2 > 0).
                # Wait. If proj1 is to the left of proj2:
                # proj1.max() < proj2.min()
                # overlap_1 = max(0, proj1.max() - proj2.min()) = 0
                # overlap_2 = max(0, proj2.max() - proj1.min()) = proj2.max() - proj1.min() > 0
                # So summing both is wrong for disjoint cases.
                
                # Correct logic:
                # They are disjoint if there exists an axis where proj1.max() < proj2.min() OR proj2.max() < proj1.min().
                # We want a penalty that is 0 if disjoint, and >0 if overlapping.
                # A common trick:
                # Let d = min(proj1.max() - proj2.min(), proj2.max() - proj1.min())? No.
                
                # Let's look at the separation distance.
                # sep = min(proj1.min(), proj2.min()) - max(proj1.max(), proj2.max())? No.
                
                # Simpler:
                # If proj1.max() < proj2.min(): sep = proj2.min() - proj1.max() (positive)
                # If proj2.max() < proj1.min(): sep = proj1.min() - proj2.max() (positive)
                # Else: sep = 0 (overlap)
                
                # We can compute sep = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())?
                # No.
                
                # Let's use:
                # sep_1 = proj2.min() - proj1.max()
                # sep_2 = proj1.min() - proj2.max()
                # If sep_1 > 0 or sep_2 > 0, they are separated.
                # Penalty = max(0, -sep_1)^2 + max(0, -sep_2)^2 ?
                # If separated by sep_1 > 0, sep_2 is likely very negative.
                # max(0, -sep_1) = 0.
                # max(0, -sep_2) = -sep_2 (large positive).
                # This would penalize disjoint objects!
                
                # We only want to penalize if they are NOT separated by ANY axis.
                # This is a global property, not local to one axis.
                # The SAT penalty must be computed over ALL axes.
                # If ANY axis has separation > 0, penalty is 0.
                # If NO axis has separation > 0, penalty is sum of squared penetrations?
                
                # This makes the function non-differentiable at the boundary (min over axes).
                # However, for optimization, we can approximate.
                # Or, we can sum the penalties from all axes, but weight them such that if one axis separates, the total penalty drops.
                
                # Alternative: Use the minimum distance between the two polygons.
                # If distance > 0, penalty 0.
                # If distance < 0 (overlap), penalty = distance^2.
                # Computing min distance between convex polygons is differentiable almost everywhere.
                
                # For now, let's use a simpler heuristic that is differentiable:
                # Penalty = max(0, dist_centers - min_dist_required)^2 ? No, too weak.
                
                # Let's try the "Sum of Max(0, -sep)" approach but only if it's the MINIMUM separation?
                # Actually, `scipy` doesn't have a built-in differentiable SAT penalty.
                
                # Let's stick to the previous approach but make it smooth.
                # We calculate the minimum overlap width $w$.
                # If $w > 0$ (overlap), penalty = $w^2$.
                # $w$ is the minimum over all axes of the "separation" if we define separation as negative when overlapping.
                # Let $s_k$ be the separation along axis $k$.
                # $s_k = \min(\text{proj}_1.\max - \text{proj}_2.\min, \text{proj}_2.\max - \text{proj}_1.\min)$? No.
                
                # Let's use:
                # $s_k = \min(\text{proj}_2.\min - \text{proj}_1.\max, \text{proj}_1.\min - \text{proj}_2.\max)$
                # If $s_k > 0$, separated.
                # If $s_k < 0$, overlapping on this axis.
                # The "depth" of overlap is $\max(0, -s_k)$.
                # The total overlap depth is $\min_k (-s_k)$? No, if they overlap, all $s_k < 0$.
                # The minimum depth of penetration is $\min_k (-s_k)$.
                # So penalty = $\max(0, \min_k (-s_k))^2$.
                # This is differentiable almost everywhere.
                
                # Let's implement this.
                pass # Placeholder for logic below

    # ... rest of code
# EVOLVE-BLOCK-END
