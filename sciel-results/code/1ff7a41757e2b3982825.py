# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay


def _all_triangle_areas(pts):
    """Compute areas of all C(n,3) triangles. Returns (areas, indices)."""
    n = len(pts)
    idx = []
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                a = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                idx.append((i, j, k))
                areas.append(a)
    return np.array(areas), idx


def _score(pts):
    """Compute the Heilbronn score: A_min / A_hull."""
    n = len(pts)
    if n < 3:
        return 0.0
    areas, idx = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    a_min = np.min(areas)
    if a_min < 1e-15:
        return 0.0
    try:
        hull = ConvexHull(pts)
        a_hull = hull.volume
    except Exception:
        return 0.0
    if a_hull < 1e-15:
        return 0.0
    return a_min / a_hull


def _compute_forces(pts, areas, idx, n):
    """
    Compute forces on each point.
    For each triangle with area < threshold, push its vertices apart.
    Also add a gentle outward force for hull points to maintain convexity.
    """
    forces = np.zeros_like(pts)
    a_min = np.min(areas)
    
    # Threshold: triangles that are "small" relative to the minimum
    # We want to increase the smallest triangles
    threshold = a_min * 1.5
    
    for (i, j, k), a in zip(idx, areas):
        if a < threshold and a > 1e-15:
            # Force proportional to how "small" this triangle is
            # Push vertices apart from the centroid
            centroid = (pts[i] + pts[j] + pts[k]) / 3.0
            strength = (threshold - a) / threshold  # 0 to 1
            
            for v in [i, j, k]:
                diff = pts[v] - centroid
                norm = np.linalg.norm(diff)
                if norm > 1e-10:
                    forces[v] += strength * diff / norm * 0.01
    
    # Hull points: gentle outward push to maintain convex shape
    try:
        hull = ConvexHull(pts)
        hull_verts = hull.vertices
        # Compute outward normals for hull edges
        hv = pts[hull_verts]
        m = len(hv)
        for ii in range(m):
            p1 = hv[ii]
            p2 = hv[(ii + 1) % m]
            edge = p2 - p1
            # Outward normal (perpendicular, pointing away from centroid)
            normal = np.array([-edge[1], edge[0]])
            centroid = np.mean(hv, axis=0)
            mid = (p1 + p2) / 2.0
            if np.dot(normal, mid - centroid) < 0:
                normal = -normal
            normal = normal / (np.linalg.norm(normal) + 1e-10)
            # Apply gentle outward force to both hull vertices
            idx1 = hull_verts[ii]
            idx2 = hull_verts[(ii + 1) % m]
            forces[idx1] += normal * 0.002
            forces[idx2] += normal * 0.002
    except Exception:
        pass
    
    return forces


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area, using physics-based relaxation.
    """
    deadline = time.time() + 600.0
    n = 13
    
    # Start with a regular 13-gon as initial configuration
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = _score(best_pts)
    
    # Also try a few different initial configurations
    initial_configs = [best_pts.copy()]
    
    # Configuration 2: 12 points on circle + 1 center
    angles12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts2 = np.column_stack([np.cos(angles12), np.sin(angles12)])
    pts2 = np.vstack([pts2, np.array([[0.0, 0.0]])])
    initial_configs.append(pts2)
    
    # Configuration 3: Two concentric rings (7 + 6)
    angles7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles6 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    pts3 = np.vstack([
        np.column_stack([np.cos(angles7) * 1.0, np.sin(angles7) * 1.0]),
        np.column_stack([np.cos(angles6) * 0.5, np.sin(angles6) * 0.5])
    ])
    initial_configs.append(pts3)
    
    # Configuration 4: Three concentric rings (5 + 4 + 4)
    angles5 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    angles4a = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    angles4b = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    pts4 = np.vstack([
        np.column_stack([np.cos(angles5) * 1.0, np.sin(angles5) * 1.0]),
        np.column_stack([np.cos(angles4a) * 0.6, np.sin(angles4a) * 0.6]),
        np.column_stack([np.cos(angles4b) * 0.3, np.sin(angles4b) * 0.3])
    ])
    initial_configs.append(pts4)
    
    for init_pts in initial_configs:
        if time.time() > deadline - 30:
            break
        
        pts = init_pts.copy()
        current_score = _score(pts)
        
        # Physics-based relaxation
        step_size = 0.05
        cooling = 1.0
        
        max_iters = 5000
        for it in range(max_iters):
            if time.time() > deadline - 30:
                break
            
            # Compute areas and forces
            areas, idx = _all_triangle_areas(pts)
            forces = _compute_forces(pts, areas, idx, n)
            
            # Update positions
            pts = pts + step_size * cooling * forces
            
            # Keep points in a reasonable bounded region
            # Project back if any point goes too far
            max_norm = np.max(np.linalg.norm(pts, axis=1))
            if max_norm > 10:
                pts = pts / max_norm * 10
            
            # Compute new score
            new_score = _score(pts)
            
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
            else:
                # Revert if score decreased
                pts = pts - step_size * cooling * forces
                cooling *= 0.95
                step_size *= 0.98
            
            if cooling < 0.001:
                break
        
        # Final local refinement with smaller steps
        pts = best_pts.copy()
        current_score = _score(pts)
        step_size = 0.01
        cooling = 1.0
        
        for it in range(3000):
            if time.time() > deadline - 20:
                break
            
            areas, idx = _all_triangle_areas(pts)
            forces = _compute_forces(pts, areas, idx, n)
            
            pts_new = pts + step_size * cooling * forces
            
            # Bound check
            max_norm = np.max(np.linalg.norm(pts_new, axis=1))
            if max_norm > 10:
                pts_new = pts_new / max_norm * 10
            
            new_score = _score(pts_new)
            
            if new_score > current_score:
                current_score = new_score
                pts = pts_new
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
            else:
                cooling *= 0.97
                step_size *= 0.99
            
            if cooling < 0.001:
                break
    
    # Ensure valid output
    if not np.all(np.isfinite(best_pts)):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts


# EVOLVE-BLOCK-END
