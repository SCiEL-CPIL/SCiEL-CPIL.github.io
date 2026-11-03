"""Voronoi-based packing optimization."""
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
import time

N = 50

def _clip_polygon_to_square(polygon, min_x=-0.5, max_x=0.5, min_y=-0.5, max_y=0.5):
    """
    Clip a convex polygon (list of (x, y)) to the square [min_x, max_x] x [min_y, max_y].
    Uses Sutherland-Hodgman algorithm.
    """
    if len(polygon) == 0:
        return []

    def clip_edge(poly, min_coord, max_coord, axis):
        output = []
        for i in range(len(poly)):
            current = poly[i]
            next_point = poly[(i + 1) % len(poly)]
            
            c_in = min_coord <= current[axis] <= max_coord
            n_in = min_coord <= next_point[axis] <= max_coord
            
            if c_in:
                output.append(current)
                if not n_in:
                    # Exit: interpolate
                    t = (max_coord - current[axis]) / (next_point[axis] - current[axis]) if axis == 0 else (max_coord - current[axis]) / (next_point[axis] - current[axis])
                    # Correct interpolation logic
                    if axis == 0:
                        x = max_coord
                        y = current[1] + t * (next_point[1] - current[1])
                    else:
                        x = current[0] + t * (next_point[0] - current[0])
                        y = max_coord
                    output.append((x, y))
            else:
                if n_in:
                    # Enter: interpolate
                    if axis == 0:
                        t = (min_coord - current[axis]) / (next_point[axis] - current[axis])
                        x = min_coord
                        y = current[1] + t * (next_point[1] - current[1])
                    else:
                        t = (min_coord - current[axis]) / (next_point[axis] - current[axis])
                        x = current[0] + t * (next_point[0] - current[0])
                        y = min_coord
                    output.append((x, y))
        return output

    poly = [tuple(p) for p in polygon]
    poly = clip_edge(poly, min_x, max_x, 0)
    if not poly: return []
    poly = clip_edge(poly, max_x, min_x, 0) # Note: S-H usually clips against one line at a time. 
    # My clip_edge above only handles one bound. I need to clip against 4 lines.
    
    # Let's implement proper S-H for 4 lines.
    def clip_line(poly, min_val, max_val, axis):
        if not poly: return []
        output = []
        for i in range(len(poly)):
            curr = poly[i]
            nxt = poly[(i+1)%len(poly)]
            
            curr_in = min_val <= curr[axis] <= max_val
            nxt_in = min_val <= nxt[axis] <= max_val
            
            if curr_in:
                output.append(curr)
                if not nxt_in:
                    # Crossing out
                    # Interpolate
                    if curr[axis] == nxt[axis]: continue
                    t = (max_val - curr[axis]) / (nxt[axis] - curr[axis])
                    new_pt = tuple(curr[d] + t*(nxt[d]-curr[d]) for d in range(2))
                    # Ensure it's on the boundary
                    new_pt = list(new_pt)
                    new_pt[axis] = max_val
                    output.append(tuple(new_pt))
            elif nxt_in:
                # Crossing in
                if curr[axis] == nxt[axis]: continue
                t = (min_val - curr[axis]) / (nxt[axis] - curr[axis])
                new_pt = tuple(curr[d] + t*(nxt[d]-curr[d]) for d in range(2))
                new_pt = list(new_pt)
                new_pt[axis] = min_val
                output.append(tuple(new_pt))
        
        return output

    # Clip against x >= -0.5
    poly = clip_line(poly, -0.5, 1000, 0)
    # Clip against x <= 0.5
    poly = clip_line(poly, -1000, 0.5, 0)
    # Clip against y >= -0.5
    poly = clip_line(poly, -0.5, 1000, 1)
    # Clip against y <= 0.5
    poly = clip_line(poly, -1000, 0.5, 1)
    
    return poly

def _inradius_of_polygon(polygon, center):
    """
    Calculate the minimum distance from center to the edges of the polygon.
    """
    if len(polygon) < 3:
        return 0.0
        
    min_dist = float('inf')
    n = len(polygon)
    for i in range(n):
        p1 = np.array(polygon[i])
        p2 = np.array(polygon[(i+1)%n])
        # Line segment p1-p2
        # Distance from center to line segment
        v = p2 - p1
        w = center - p1
        c1 = np.dot(w, v)
        if c1 <= 0:
            dist = np.linalg.norm(center - p1)
        else:
            c2 = np.dot(v, v)
            if c2 <= c1:
                dist = np.linalg.norm(center - p2)
            else:
                c3 = c1 / c2
                closest = p1 + c3 * v
                dist = np.linalg.norm(center - closest)
        if dist < min_dist:
            min_dist = dist
            
    return min_dist

def _calculate_voronoi_radius(centers):
    """
    Calculate the minimum inradius of the Voronoi cells clipped to the square.
    """
    try:
        vor = Voronoi(centers)
    except Exception:
        # Fallback to standard radius if Voronoi fails (e.g., coincident points)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        return min(min_pair, min_wall)

    min_r = float('inf')
    
    for i in range(N):
        region = vor.regions[i]
        if len(region) == 0:
            min_r = 0.0
            break
            
        # Get vertices of the Voronoi cell
        verts = []
        for idx in region:
            if idx == -1:
                # Unbounded vertex. We need to handle this by clipping to square.
                # For simplicity, if there's an unbounded vertex, the cell touches the boundary.
                # We will just use the bounded vertices and then clip the resulting polygon?
                # Actually, if a vertex is -1, the cell is unbounded.
                # We can't form a polygon directly.
                # Strategy: If any vertex is -1, we assume the cell extends to the square boundary.
                # We will collect the bounded vertices.
                pass
            else:
                verts.append(vor.vertices[idx])
                
        if len(verts) < 3:
            # Degenerate or unbounded without enough points to form a hull?
            # Fallback: calculate distance to neighbors and walls manually
            diffs = centers - centers[i]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            dists[i] = np.inf
            min_pair = np.min(dists) / 2.0
            wall_dists = 0.5 - np.abs(centers[i])
            min_wall = np.min(wall_dists)
            r_i = min(min_pair, min_wall)
            if r_i < min_r:
                min_r = r_i
            continue
            
        # If we have unbounded vertices, the simple polygon from verts might not be closed.
        # However, for points inside the square, the Voronoi cell is the intersection of half-planes.
        # The vertices we collected are the intersections of the bisectors.
        # If the cell is clipped by the square, the "corners" of the cell might be the square corners.
        # This approximation (using only Voronoi vertices) might underestimate the inradius if the cell is large and clipped.
        # But for packing, cells are small.
        
        # To be more robust, if the cell is unbounded, we should probably just use the neighbor/wall distance method.
        # Let's check if the region contains -1.
        if -1 in region:
            # Use the simple distance method for unbounded cells
            diffs = centers - centers[i]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            dists[i] = np.inf
            min_pair = np.min(dists) / 2.0
            wall_dists = 0.5 - np.abs(centers[i])
            min_wall = np.min(wall_dists)
            r_i = min(min_pair, min_wall)
            if r_i < min_r:
                min_r = r_i
            continue
            
        # Bounded cell. Clip to square just in case (should be inside if points are inside).
        poly = verts
        poly = _clip_polygon_to_square(poly)
        
        if len(poly) < 3:
             # Degenerate
             r_i = 0.0
        else:
             r_i = _inradius_of_polygon(poly, centers[i])
             
        if r_i < min_r:
            min_r = r_i
            
    return min_r

def _optimize_voronoi(centers, time_limit=100.0):
    """
    Optimize centers using Nelder-Mead on the Voronoi inradius objective.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 0.0
        c = params.reshape(N, 2)
        # Clip to valid range to avoid Voronoi issues
        c = np.clip(c, -0.49, 0.49)
        r = _calculate_voronoi_radius(c)
        return -r

    # Use Nelder-Mead as it doesn't require gradients and handles the non-smooth Voronoi topology changes better
    result = minimize(
        objective, 
        centers.reshape(-1), 
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8}
    )
    
    best_centers = result.x.reshape(N, 2)
    best_score = -result.fun
    
    # Final polish with L-BFGS-B on the smooth approximation if time permits
    if time.time() - start_time < time_limit - 10:
        def smooth_objective(params, k=500):
            c = params.reshape(N, 2)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            pair_terms = dists / 2.0
            wall_dists = 0.5 - np.abs(c)
            all_terms = np.concatenate([pair_terms.flatten(), wall_dists.flatten()])
            all_terms = all_terms[np.isfinite(all_terms)]
            if len(all_terms) == 0:
                return 0.0
            m = np.min(all_terms)
            shifted = all_terms - m
            exp_terms = np.exp(-k * shifted)
            log_sum = np.log(np.sum(exp_terms))
            soft_min = m - (1.0 / k) * log_sum
            return -soft_min

        try:
            res2 = minimize(
                smooth_objective,
                best_centers.reshape(-1),
                method='L-BFGS-B',
                bounds=[(-0.49, 0.49)] * (2*N),
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            c2 = res2.x.reshape(N, 2)
            score2 = _calculate_voronoi_radius(c2)
            if score2 > best_score:
                best_centers = c2
                best_score = score2
        except Exception:
            pass
            
    return best_centers, best_score

def construct():
    # Initialize with a hexagonal grid
    s = 0.14
    points = []
    range_x = np.arange(-0.8, 0.8, s)
    range_y = np.arange(-0.8, 0.8, s)
    for i, x in enumerate(range_x):
        offset = (s / 2) if i % 2 == 0 else 0
        for y in range_y:
            points.append((x + offset, y))
    points = np.array(points)
    mask = (np.abs(points[:, 0]) <= 0.49) & (np.abs(points[:, 1]) <= 0.49)
    points = points[mask]
    
    if len(points) < N:
        # Fallback to random
        np.random.seed(42)
        points = np.random.uniform(-0.45, 0.45, (N, 2))
    else:
        np.random.seed(42)
        indices = np.random.choice(len(points), N, replace=False)
        points = points[indices]
        
    best_centers = points
    best_score = _calculate_voronoi_radius(best_centers)
    
    # Run optimization
    centers, score = _optimize_voronoi(best_centers, time_limit=100.0)
    if score > best_score:
        best_centers = centers
        best_score = score
        
    # Final check and clip
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
