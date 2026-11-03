import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    
    # Unit hexagon vertices (flat-top, centered at origin)
    def unit_hex_verts(cx, cy, angle_deg):
        a = np.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = a + k * np.pi / 3
            verts.append([cx + np.cos(theta), cy + np.sin(theta)])
        return np.array(verts)
    
    # Check if all points are inside outer hexagon with center (ox, oy), side S, rotation rot
    def all_inside(points, ox, oy, S, rot_deg):
        rot = np.radians(rot_deg)
        for px, py in points:
            # Translate to outer center
            dx, dy = px - ox, py - oy
            # Rotate by -rot to align with standard hexagon
            xr = dx * np.cos(-rot) - dy * np.sin(-rot)
            yr = dx * np.sin(-rot) + dy * np.cos(-rot)
            # Check 6 half-planes: |projection on each axis| <= S*sqrt(3)/2
            # The 6 normals are at angles 30, 90, 150, 210, 270, 330 degrees
            for k in range(6):
                ang = np.pi / 6 + k * np.pi / 3
                proj = xr * np.cos(ang) + yr * np.sin(ang)
                if abs(proj) > S * np.sqrt(3) / 2 + 1e-10:
                    return False
        return True
    
    # Compute minimum enclosing hexagon side for given points, center, rotation
    def min_hex_side(points, ox, oy, rot_deg):
        # Binary search on S
        lo, hi = 0.0, 100.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if all_inside(points, ox, oy, mid, rot_deg):
                hi = mid
            else:
                lo = mid
        return hi
    
    # Generate candidate arrangements
    def honeycomb_positions(n, spacing=1.001):
        """Generate honeycomb lattice positions for n hexagons"""
        # Hexagonal lattice vectors for hexagons with side 1
        # Center-to-center distance in honeycomb: sqrt(3) for adjacent
        d = spacing * np.sqrt(3)  # distance between adjacent hex centers
        positions = []
        # Use axial coordinates
        for q in range(-10, 11):
            for r in range(-10, 11):
                # Convert axial to Cartesian for hex grid
                x = d * (q + r / 2)
                y = d * (r * np.sqrt(3) / 2)
                positions.append((x, y))
                if len(positions) >= n + 50:
                    break
            if len(positions) >= n + 50:
                break
        return positions
    
    # Try different arrangements
    best_result = None
    best_score = float('inf')
    
    # Strategy 1: Honeycomb cluster arrangements
    # Generate various compact clusters
    def generate_clusters(n):
        clusters = []
        # Hexagonal number pattern: 1, 7, 19, ...
        # For 11: center + 6 + 4 from next ring
        # Axial coordinates for hex grid
        hex_dirs = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]
        
        # BFS to get n closest positions to origin
        from collections import deque
        visited = set()
        order = [(0,0)]
        visited.add((0,0))
        queue = deque([(0,0)])
        while len(order) < n:
            q, r = queue.popleft()
            for dq, dr in hex_dirs:
                nq, nr = q + dq, r + dr
                if (nq, nr) not in visited:
                    visited.add((nq, nr))
                    order.append((nq, nr))
                    queue.append((nq, nr))
        
        # Convert to Cartesian
        d = np.sqrt(3) * 1.001
        pts = []
        for q, r in order[:n]:
            x = d * (q + r / 2)
            y = d * (r * np.sqrt(3) / 2)
            pts.append((x, y))
        clusters.append(pts)
        
        # Also try linear arrangements
        for spacing_mult in [1.0, 1.001, 1.002]:
            d = np.sqrt(3) * spacing_mult
            # Rows: try different row splits
            for row_split in [(1,2,3,3,2,1), (2,3,3,2,1), (3,3,3,2), (4,3,3,1), (5,3,2,1), (6,3,1,1)]:
                if sum(row_split) != n:
                    continue
                pts = []
                y = 0.0
                for row_len in row_split:
                    for i in range(row_len):
                        x = (i - (row_len - 1) / 2) * d
                        pts.append((x, y))
                    y += d * np.sqrt(3) / 2
                # Center vertically
                max_y = max(p[1] for p in pts)
                min_y = min(p[1] for p in pts)
                offset = -(max_y + min_y) / 2
                pts = [(x, y + offset) for x, y in pts]
                clusters.append(pts)
        
        return clusters
    
    # Evaluate a cluster arrangement
    def evaluate_cluster(positions, all_angles, outer_rot):
        # Collect all vertices
        all_verts = []
        for i, (x, y) in enumerate(positions):
            angle = all_angles[i]
            verts = unit_hex_verts(x, y, angle)
            all_verts.extend(verts.tolist())
        
        # Compute centroid of all vertices as outer center
        cx = np.mean([v[0] for v in all_verts])
        cy = np.mean([v[1] for v in all_verts])
        
        S = min_hex_side(all_verts, cx, cy, outer_rot)
        return S, (cx, cy)
    
    # Try various configurations
    clusters = generate_clusters(N)
    
    for cluster in clusters:
        if time.time() > deadline - 10:
            break
        
        # Try all 0 degree rotation for inner hexagons
        angles = [0.0] * N
        for outer_rot in [0.0, 30.0]:
            S, center = evaluate_cluster(cluster, angles, outer_rot)
            if S < best_score:
                best_score = S
                inner = [[x, y, a] for (x, y), a in zip(cluster, angles)]
                best_result = (inner, center, S, outer_rot)
        
        # Try 30 degree rotation for inner hexagons (pointy-top)
        angles = [30.0] * N
        for outer_rot in [0.0, 30.0]:
            S, center = evaluate_cluster(cluster, angles, outer_rot)
            if S < best_score:
                best_score = S
                inner = [[x, y, a] for (x, y), a in zip(cluster, angles)]
                best_result = (inner, center, S, outer_rot)
    
    # Local optimization: try to adjust positions slightly
    if best_result is not None and time.time() < deadline - 20:
        inner, center, S, rot = best_result
        
        def objective(params):
            if time.time() > deadline - 5:
                return float('inf')
            new_positions = []
            for i in range(N):
                dx = params[2*i]
                dy = params[2*i+1]
                x = inner[i][0] + dx
                y = inner[i][1] + dy
                new_positions.append((x, y))
            
            all_verts = []
            for i, (x, y) in enumerate(new_positions):
                verts = unit_hex_verts(x, y, inner[i][2])
                all_verts.extend(verts.tolist())
            
            cx = np.mean([v[0] for v in all_verts])
            cy = np.mean([v[1] for v in all_verts])
            
            S_new = min_hex_side(all_verts, cx, cy, rot)
            return S_new
        
        x0 = np.zeros(2 * N)
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_score and time.time() < deadline - 5:
                new_positions = []
                for i in range(N):
                    x = inner[i][0] + res.x[2*i]
                    y = inner[i][1] + res.x[2*i+1]
                    new_positions.append((x, y))
                
                all_verts = []
                for i, (x, y) in enumerate(new_positions):
                    verts = unit_hex_verts(x, y, inner[i][2])
                    all_verts.extend(verts.tolist())
                
                cx = np.mean([v[0] for v in all_verts])
                cy = np.mean([v[1] for v in all_verts])
                S_new = min_hex_side(all_verts, cx, cy, rot)
                
                if S_new < best_score:
                    best_score = S_new
                    best_result = (new_positions, (cx, cy), S_new, rot)
        except:
            pass
    
    if best_result is None:
        # Fallback
        inner = [[0.0, 0.0, 0.0]]
        return inner, (0.0, 0.0), 10.0, 0.0
    
    inner, center, S, rot = best_result
    return [[x, y, a] for x, y, a in inner], center, S, rot

# EVOLVE-BLOCK-END
