"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon (flat-top orientation at 0 deg)."""
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts[i, 0] = cx + side * math.cos(theta)
        verts[i, 1] = cy + side * math.sin(theta)
    return verts

def _outer_vertices(cx, cy, side, angle_deg):
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts[i, 0] = cx + side * math.cos(theta)
        verts[i, 1] = cy + side * math.sin(theta)
    return verts

def _project_range(verts, axis):
    """Min and max projection of polygon vertices onto a unit axis."""
    dots = verts @ axis
    return dots.min(), dots.max()

def _sat_overlap(v1, v2, tol=1e-9):
    """True if two convex polygons overlap (interior intersection)."""
    for poly in (v1, v2):
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            edge = poly[j] - poly[i]
            # Normal to edge (2D)
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            r1min, r1max = _project_range(v1, normal)
            r2min, r2max = _project_range(v2, normal)
            if r1max < r2min + tol or r2max < r1min + tol:
                return False
    return True

def _point_in_convex(px, py, poly, tol=1e-9):
    """Check if point is inside a convex polygon (given in CCW order)."""
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        # Cross product of edge (i->j) with point (i->p)
        ex = poly[j, 0] - poly[i, 0]
        ey = poly[j, 1] - poly[i, 1]
        px_rel = px - poly[i, 0]
        py_rel = py - poly[i, 1]
        cross = ex * py_rel - ey * px_rel
        if cross < -tol:
            return False
    return True

def _distance_to_boundary(px, py, outer_verts):
    """Calculate the distance from point (px, py) to the boundary of the outer hexagon.
    Returns negative if inside, positive if outside? 
    Actually, for SA, we want a penalty for being outside.
    We can calculate the minimum distance to any edge.
    """
    min_dist = float('inf')
    n = len(outer_verts)
    for i in range(n):
        p1 = outer_verts[i]
        p2 = outer_verts[i+1] if i+1 < n else outer_verts[0]
        
        # Vector p1->p2
        v = p2 - p1
        # Vector p1->p
        w = np.array([px, py]) - p1
        
        c1 = np.dot(w, v)
        if c1 <= 0:
            dist = np.linalg.norm(w)
        else:
            c2 = np.dot(v, v)
            if c2 <= c1:
                dist = np.linalg.norm(w - v)
            else:
                t = c1 / c2
                dist = np.linalg.norm(w - t * v)
        min_dist = min(min_dist, dist)
    return min_dist

def _penalty_function(placements, outer_side):
    """Calculate total penalty for a configuration.
    placements: list of (x, y, angle_deg)
    outer_side: side length of container
    """
    penalty = 0.0
    
    # 1. Boundary violation penalty
    outer_verts = _outer_vertices(0.0, 0.0, outer_side, 0.0)
    for x, y, a in placements:
        verts = _hex_vertices(x, y, a, 1.0)
        for v in verts:
            # Check if vertex is inside outer hex
            # If not, add penalty proportional to distance outside
            # Using _point_in_convex is boolean. We need distance.
            # Let's use a simpler check: if not inside, penalty += 10.0 * (distance violation)
            # To avoid expensive distance calc for every vertex if inside, 
            # we can just check inside first.
            if not _point_in_convex(v[0], v[1], outer_verts, tol=1e-6):
                 # Estimate violation depth
                 # Find closest point on boundary
                 # For speed, just add a large constant if outside
                 penalty += 100.0 
                 # More precise:
                 # dist = _distance_to_boundary(v[0], v[1], outer_verts)
                 # if dist < 0: penalty += 100 * (-dist) # dist is positive if outside in my previous def?
                 # My _distance_to_boundary returns min dist to segment. If point is outside, this is dist to boundary.
                 # So penalty += 100 * dist
                 d = _distance_to_boundary(v[0], v[1], outer_verts)
                 penalty += 100.0 * d

    # 2. Overlap penalty
    n_placed = len(placements)
    verts_list = [_hex_vertices(x, y, a, 1.0) for x, y, a in placements]
    for i in range(n_placed):
        for j in range(i + 1, n_placed):
            # Check overlap
            # SAT is boolean. For penalty, we need "depth" of overlap.
            # Approximation: If they overlap, add penalty based on center distance.
            # Better: Calculate min distance between vertices? 
            # Simple heuristic: If centers are too close, they overlap.
            # For unit hexagons, center distance < sqrt(3) implies overlap?
            # Min distance between centers of non-overlapping hexagons (aligned) is 2*inradius = sqrt(3) ~ 1.732.
            # If rotated, it can be less.
            # Let's use SAT to detect overlap, and if so, add a penalty.
            # To get a gradient, we can use the minimum separation distance.
            # But SAT only tells us if they are separated.
            # Let's use a simpler penalty: 
            # If they overlap, penalty += 100.0
            # This creates a "hard" constraint landscape. SA can still work.
            if _sat_overlap(verts_list[i], verts_list[j], tol=1e-6):
                 penalty += 100.0

    return penalty

def _simulate_annealing(outer_side, time_limit=2.0):
    """Run SA to find a feasible packing for the given outer side."""
    start_time = time.time()
    
    # Initial random placement
    # Place centers within the safe region of the outer hexagon
    # Inradius of outer = outer_side * sqrt(3) / 2
    # Inradius of inner = sqrt(3) / 2
    # Max distance from center for inner center = inradius_outer - inradius_inner
    max_r = max(0.1, (outer_side * math.sqrt(3.0) / 2.0) - (math.sqrt(3.0) / 2.0))
    
    placements = []
    for _ in range(N):
        r = max_r * math.sqrt(np.random.rand())
        theta = 2 * math.pi * np.random.rand()
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        a = np.random.rand() * 60.0 # 0 to 60 degrees
        placements.append([x, y, a])
    
    current_cost = _penalty_function(placements, outer_side)
    best_placements = placements.copy()
    best_cost = current_cost
    
    T = 10.0
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit and T > 1e-4:
        # Pick a random hexagon to move
        i = np.random.randint(N)
        
        # Generate a new position
        dx = np.random.randn() * 0.5
        dy = np.random.randn() * 0.5
        da = np.random.randn() * 10.0
        
        new_x = placements[i][0] + dx
        new_y = placements[i][1] + dy
        new_a = placements[i][2] + da
        
        # Keep within bounds (loose bounds, penalty handles strictness)
        # But don't go too far out
        if new_x**2 + new_y**2 > (outer_side * 2)**2:
            continue
            
        old_pos = placements[i].copy()
        placements[i] = [new_x, new_y, new_a]
        
        new_cost = _penalty_function(placements, outer_side)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_placements = [p.copy() for p in placements]
        else:
            # Accept with probability exp(-(new_cost - current_cost) / T)
            if np.random.rand() < math.exp(-(new_cost - current_cost) / T):
                current_cost = new_cost
            else:
                # Reject: revert
                placements[i] = old_pos
        
        T *= cooling_rate
        
    # Final check
    if best_cost < 1e-3: # Threshold for "feasible"
        return best_placements
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Bisection search on outer side length
    lo = 2.5
    hi = 15.0
    
    best_placements = None
    best_side = hi
    
    # First, find a feasible upper bound
    # Try a few values to bracket
    for test_side in [3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]:
        if time.time() > deadline - 5.0:
            break
        result = _simulate_annealing(test_side, time_limit=1.0)
        if result is not None:
            best_side = test_side
            best_placements = result
            hi = test_side
            break
    
    # If we didn't find any feasible, use a large value
    if best_placements is None:
        # Fallback: place in a line
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Bisection to find minimum feasible side
    # We have a feasible hi. We need to find the lowest lo that is infeasible? 
    # Or just bisect between lo (infeasible?) and hi (feasible).
    # We should ensure lo is infeasible.
    # Let's check if current lo is feasible.
    # If lo is feasible, we can try to go lower.
    
    # Check lower bound
    test_lo = _simulate_annealing(lo, time_limit=1.0)
    if test_lo is not None:
        # lo is feasible, try even lower
        # Area limit is sqrt(6) ~ 2.449
        while lo > 2.45:
            lo -= 0.1
            if time.time() > deadline - 5.0:
                break
            test_lo = _simulate_annealing(lo, time_limit=0.5)
            if test_lo is None:
                break
    
    # Now lo is likely infeasible, hi is feasible.
    # Bisect
    for _ in range(30):
        if time.time() > deadline - 5.0:
            break
        mid = (lo + hi) / 2.0
        result = _simulate_annealing(mid, time_limit=1.0)
        if result is not None:
            hi = mid
            best_side = mid
            best_placements = result
        else:
            lo = mid
    
    # Return the best found
    if best_placements is not None:
        return best_placements, (0.0, 0.0), best_side, 0.0
    else:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-END
