import numpy as np
import time
import random
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Helper functions for geometry
    
    def get_hex_verts(cx, cy, theta):
        """Get vertices of a unit hexagon."""
        angles = theta + np.arange(6) * np.pi / 3
        return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])
    
    def get_outer_verts(cx, cy, S, theta):
        """Get vertices of the outer hexagon."""
        angles = theta + np.arange(6) * np.pi / 3
        return np.column_stack([cx + S * np.cos(angles), cy + S * np.sin(angles)])
    
    def point_in_polygon(p, poly):
        """Check if point p is inside convex polygon poly using cross products."""
        # p: (2,), poly: (6, 2)
        # All cross products of (poly[i+1] - poly[i]) and (p - poly[i]) should have the same sign
        cross = np.cross(np.roll(poly, -1, axis=0) - poly, p - poly)
        return np.all(cross >= -1e-9) or np.all(cross <= 1e-9)
        
    def poly_overlap(p1, p2):
        """Check if two convex polygons overlap using SAT."""
        # p1, p2: (6, 2)
        # Get all edge normals
        normals = []
        for i in range(6):
            e1 = p1[(i+1)%6] - p1[i]
            normals.append(np.array([-e1[1], e1[0]]))
            e2 = p2[(i+1)%6] - p2[i]
            normals.append(np.array([-e2[1], e2[0]]))
        
        for n in normals:
            # Project both polygons onto normal
            proj1 = p1.dot(n)
            proj2 = p2.dot(n)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for gap
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return False
        return True

    def max_penetration(p1, p2):
        """Estimate maximum penetration depth for overlap."""
        # If not overlapping, return 0
        if not poly_overlap(p1, p2):
            return 0.0
        # Approximate penetration by max distance of vertices of one inside the other
        # This is not exact but good for optimization
        depth = 0.0
        for v in p1:
            if point_in_polygon(v, p2):
                # Distance to boundary of p2
                # For convex poly, min distance to edges
                min_d = float('inf')
                for i in range(6):
                    e = p2[(i+1)%6] - p2[i]
                    # Project v onto line segment
                    t = np.dot(v - p2[i], e) / np.dot(e, e)
                    t = np.clip(t, 0, 1)
                    closest = p2[i] + t * e
                    d = np.linalg.norm(v - closest)
                    min_d = min(min_d, d)
                depth = max(depth, min_d)
        for v in p2:
            if point_in_polygon(v, p1):
                min_d = float('inf')
                for i in range(6):
                    e = p1[(i+1)%6] - p1[i]
                    t = np.dot(v - p1[i], e) / np.dot(e, e)
                    t = np.clip(t, 0, 1)
                    closest = p1[i] + t * e
                    d = np.linalg.norm(v - closest)
                    min_d = min(min_d, d)
                depth = max(depth, min_d)
        return depth

    def containment_violation(inner_verts, outer_verts):
        """Calculate how much inner_verts violate outer_verts containment."""
        viol = 0.0
        for v in inner_verts:
            if not point_in_polygon(v, outer_verts):
                # Distance to outer boundary
                min_d = float('inf')
                for i in range(6):
                    e = outer_verts[(i+1)%6] - outer_verts[i]
                    t = np.dot(v - outer_verts[i], e) / np.dot(e, e)
                    t = np.clip(t, 0, 1)
                    closest = outer_verts[i] + t * e
                    d = np.linalg.norm(v - closest)
                    min_d = min(min_d, d)
                viol = max(viol, min_d)
        return viol

    def evaluate_config(config, S):
        """
        config: list of (x, y, theta)
        S: outer side length
        Returns total penalty
        """
        outer_verts = get_outer_verts(0.0, 0.0, S, 0.0)
        penalty = 0.0
        inner_verts_list = []
        
        # Check containment
        for i in range(N):
            x, y, th = config[i]
            verts = get_hex_verts(x, y, th)
            inner_verts_list.append(verts)
            penalty += containment_violation(verts, outer_verts)
            
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                pen = max_penetration(inner_verts_list[i], inner_verts_list[j])
                penalty += pen
                
        return penalty

    def local_search(config, S, max_iters=500):
        """
        Local search to minimize penalty for a given S.
        """
        current_penalty = evaluate_config(config, S)
        best_config = config[:]
        best_penalty = current_penalty
        
        temp = 0.5
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Pick random hexagon and perturb
            i = random.randint(0, N-1)
            x, y, th = config[i]
            
            dx = random.gauss(0, 0.1)
            dy = random.gauss(0, 0.1)
            dth = random.gauss(0, 0.1)
            
            new_config = config[:]
            new_config[i] = (x + dx, y + dy, th + dth)
            
            new_penalty = evaluate_config(new_config, S)
            
            if new_penalty < current_penalty:
                current_penalty = new_penalty
                config = new_config
                if current_penalty < best_penalty:
                    best_penalty = current_penalty
                    best_config = config[:]
            else:
                # Accept with probability
                if random.random() < np.exp(-(new_penalty - current_penalty) / temp):
                    current_penalty = new_penalty
                    config = new_config
            
            temp *= 0.999 # Cool down
            
            if best_penalty < 1e-5:
                break
                
        return best_config, best_penalty

    # Binary search for minimal S
    low = 2.7
    high = 4.0
    
    best_S = high
    best_result = None
    
    # Initial guess: Honeycomb
    # Center at 0,0. Neighbors at distance sqrt(3).
    # For N=11, we can try to fit them.
    
    for _ in range(20): # 20 binary search steps
        if time.time() > deadline - 10:
            break
        mid = (low + high) / 2.0
        
        # Try to find a feasible config for S=mid
        feasible = False
        found_config = None
        
        # Multi-start local search
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            # Initialize config
            config = []
            # Center
            config.append((0.0, 0.0, 0.0))
            # 6 neighbors
            for k in range(6):
                ang = k * np.pi / 3
                config.append((np.sqrt(3)*np.cos(ang), np.sqrt(3)*np.sin(ang), 0.0))
            # 4 more, place them roughly
            config.append((2*np.sqrt(3), 0, 0.0))
            config.append((-2*np.sqrt(3), 0, 0.0))
            config.append((0, 2*np.sqrt(3), 0.0))
            config.append((0, -2*np.sqrt(3), 0.0))
            
            # Randomize slightly
            for i in range(N):
                x, y, th = config[i]
                config[i] = (x + random.gauss(0, 0.2), y + random.gauss(0, 0.2), th + random.gauss(0, 0.2))
                
            # Local search
            res_config, pen = local_search(config, mid, max_iters=200)
            
            if pen < 1e-5:
                feasible = True
                found_config = res_config
                break
                
        if feasible:
            high = mid
            best_S = mid
            best_result = found_config
        else:
            low = mid
            
    if best_result is None:
        # Fallback to a known valid config if search failed
        # Use a very large S to be safe, or return the best attempt
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    # Convert best_result to required format
    inner = [[x, y, np.degrees(th) % 360] for x, y, th in best_result]
    return inner, (0.0, 0.0), best_S, 0.0
# EVOLVE-BLOCK-END
