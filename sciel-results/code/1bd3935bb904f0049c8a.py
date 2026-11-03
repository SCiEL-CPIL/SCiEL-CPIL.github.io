"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, ConvexHull
import time


def _voronoi_areas(points, boundary_radius=1.0):
    """
    Compute the area of the Voronoi cell for each point, clipped to the unit disk.
    For points on the boundary, we use the area of the polygon formed by the point,
    the intersection of its Voronoi edges with the unit circle, etc.
    
    This is an approximation: we compute the Voronoi diagram in the plane, then for each cell,
    we intersect it with the unit disk to get the effective area.
    
    For efficiency, we approximate by:
    1. Computing Voronoi vertices
    2. For each region, if it's unbounded, we clip with the unit circle
    3. Computing polygon area
    """
    n = len(points)
    if n < 3:
        return np.ones(n)
    
    try:
        vor = Voronoi(points)
    except Exception:
        return np.ones(n)
    
    areas = np.zeros(n)
    
    # For each point, get its Voronoi region vertices
    for i in range(n):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if len(region) == 0:
            # Degenerate case
            areas[i] = 0.01
            continue
        
        vertices = vor.vertices[region]
        
        # Check if any vertex is at infinity (indicated by -1 in region)
        if -1 in region:
            # Unbounded region - clip with unit circle
            # Find the finite vertices
            finite_mask = np.array([v != -1 for v in region])
            finite_vertices = vor.vertices[region[finite_mask]]
            
            if len(finite_vertices) == 0:
                areas[i] = 0.01
                continue
            
            # For unbounded regions, we need to find where the Voronoi edges
            # intersect the unit circle. This is complex, so we approximate
            # by adding points on the unit circle that are "visible" from this point.
            
            # Simple approximation: the area is roughly the area of the polygon
            # formed by the finite vertices plus the arc of the unit circle
            # that faces this point.
            
            # For now, use a rough estimate based on distance from center
            dist = np.linalg.norm(points[i])
            # Approximate area as sector of circle
            # This is very rough
            areas[i] = max(0.005, 0.5 * (2 * np.pi * (1 - dist) ** 2) / n)
        else:
            # Bounded region
            if len(vertices) < 3:
                areas[i] = 0.005
                continue
            
            # Compute polygon area
            x = vertices[:, 0]
            y = vertices[:, 1]
            
            # Shoelace formula
            area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
            
            # Check if polygon is inside unit circle; if not, we might need to clip
            # For simplicity, if any vertex is outside, scale down
            max_r = np.max(np.hypot(x, y))
            if max_r > boundary_radius:
                # Rough clip: reduce area
                scale = boundary_radius / max_r
                area *= scale * scale
            
            areas[i] = max(0.001, area)
    
    return areas


def _log_det_cov_gradient(points, areas, dt=0.001):
    """
    Compute gradient of sum of log(Voronoi cell areas) with respect to point positions.
    
    For a Voronoi diagram, the gradient of the area of cell i with respect to point i
    is related to the sum of outward normals of the Voronoi edges weighted by edge length.
    
    Specifically, d(area_i)/d(point_i) = -sum of (outward normal * edge length) for each edge.
    
    For log(area_i), gradient is (1/area_i) * d(area_i)/d(point_i).
    """
    n = len(points)
    if n < 3:
        return np.zeros_like(points)
    
    try:
        vor = Voronoi(points)
    except Exception:
        return np.zeros_like(points)
    
    grads = np.zeros_like(points)
    
    for i in range(n):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if len(region) == 0 or -1 in region:
            continue
        
        if len(region) < 3:
            continue
        
        vertices = vor.vertices[region]
        
        if len(vertices) < 3:
            continue
        
        # The gradient of the area with respect to the point position
        # is the negative of the sum of (outward normal * length) for each edge
        # Equivalently, it's related to the centroid of the Voronoi cell
        # Actually, for a convex polygon, the gradient of area w.r.t. translation
        # is zero, but w.r.t. the vertex that defines the cell, it's different.
        
        # Better approach: use the fact that for Voronoi, the gradient of the
        # area of cell i w.r.t. point i is:
        # sum over edges j of (normal_j * length_j) where normal points outward
        
        # Compute edges
        m = len(vertices)
        edge_grad = np.zeros(2)
        
        for k in range(m):
            v1 = vertices[k]
            v2 = vertices[(k + 1) % m]
            
            # Edge vector
            edge_vec = v2 - v1
            edge_len = np.linalg.norm(edge_vec)
            
            if edge_len < 1e-10:
                continue
            
            # Outward normal: for a convex polygon with vertices in CCW order,
            # the outward normal of edge (v1, v2) is (dy, -dx) normalized
            # First, check if vertices are in CCW order
            edge_dir = edge_vec / edge_len
            # Normal perpendicular to edge, pointing outward
            # For CCW polygon, outward normal is (dy, -dx) where (dx, dy) is edge direction
            normal = np.array([edge_dir[1], -edge_dir[0]])
            
            # Check if this normal points away from the centroid
            centroid = np.mean(vertices, axis=0)
            edge_midpoint = (v1 + v2) / 2
            to_centroid = centroid - edge_midpoint
            
            if np.dot(normal, to_centroid) > 0:
                normal = -normal
            
            edge_grad += normal * edge_len
        
        # The gradient of area w.r.t. point position is -edge_grad
        # (moving the point in the direction of edge_grad increases area)
        # Actually, it's: d(area)/d(point) = -edge_grad
        # So for log(area), gradient is -edge_grad / area
        
        area_i = areas[i]
        if area_i > 1e-10:
            grads[i] = -edge_grad / area_i
        else:
            grads[i] = -edge_grad * 100
    
    return grads


def construct():
    """
    Pack N equal circles in a unit disk using gradient ascent on log Voronoi cell areas.
    """
    np.random.seed(42)
    
    # Initialize on hexagonal lattice
    spacing = 0.06
    points = []
    for j in range(-20, 21):
        for i in range(-20, 21):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y < 0.95:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have too many or too few, adjust
    if len(points) > N:
        # Keep the first N (centered ones)
        dists = np.hypot(points[:, 0], points[:, 1])
        points = points[np.argsort(dists)[:N]]
    elif len(points) < N:
        # Add random points
        while len(points) < N:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, 0.9)
            points = np.vstack([points, [r * np.cos(angle), r * np.sin(angle)]])
    
    points = points[:N]
    
    # Optimize
    best_score = -np.inf
    best_points = points.copy()
    
    for iteration in range(500):
        # Compute Voronoi areas
        areas = _voronoi_areas(points, boundary_radius=1.0)
        
        # Ensure positive areas
        areas = np.maximum(areas, 1e-6)
        
        # Compute gradient of sum of log(areas)
        grads = _log_det_cov_gradient(points, areas)
        
        # Gradient ascent with step size
        # Use adaptive step size
        step = 0.002 / (1 + iteration * 0.001)
        
        new_points = points + step * grads
        
        # Project back inside unit disk
        radii = np.hypot(new_points[:, 0], new_points[:, 1])
        mask = radii > 0.98
        if np.any(mask):
            new_points[mask] *= 0.98 / radii[mask]
        
        # Check for overlaps
        # Compute minimum distance between pairs
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(new_points[i] - new_points[j])
                if d < min_dist:
                    min_dist = d
        
        score = min(min_dist / 2.0, np.min(1.0 - np.hypot(new_points[:, 0], new_points[:, 1])))
        
        if score > best_score:
            best_score = score
            best_points = new_points.copy()
        
        points = new_points
        
        # Early stopping if improvement is small
        if iteration > 100 and score < best_score - 0.001:
            break
    
    # Final projection
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    mask = radii > 1.0
    if np.any(mask):
        best_points[mask] *= 1.0 / radii[mask]
    
    return best_points
# EVOLVE-BLOCK-END
