import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    # Hexagon geometry constants
    SQRT3 = math.sqrt(3)
    
    def get_hex_vertices(cx, cy, angle_deg):
        """
        Get vertices of a unit regular hexagon centered at (cx, cy) with rotation angle_deg.
        Returns array of shape (6, 2).
        """
        a = math.radians(angle_deg)
        # Standard orientation: flat top? Pointy top?
        # Problem says "regular hexagon". Usually side length 1.
        # Vertices at angles 0, 60, ... or 30, 90, ...?
        # "Side length 1".
        # Let's use vertices at 0, 60, 120, 180, 240, 300 degrees relative to center.
        # This gives pointy-top if x-axis is horizontal.
        # Actually, for a regular hexagon of side 1, distance from center to vertex is 1.
        # Distance from center to side (apothem) is sqrt(3)/2.
        # Let's stick to standard: v_i = (cos(a + i*pi/3), sin(a + i*pi/3))
        thetas = a + np.arange(6) * (math.pi / 3)
        vx = cx + np.cos(thetas)
        vy = cy + np.sin(thetas)
        return np.column_stack((vx, vy))

    def check_collision(v1, v2):
        """
        Check if two convex polygons (hexagons) defined by their vertices intersect.
        v1, v2 are (6, 2) arrays.
        Returns True if they intersect (overlap).
        Uses Separating Axis Theorem (SAT).
        """
        # Get edges for v1
        edges1 = np.roll(v1, -1, axis=0) - v1
        # Get edges for v2
        edges2 = np.roll(v2, -1, axis=0) - v2
        
        # Normals
        normals1 = np.column_stack((-edges1[:, 1], edges1[:, 0]))
        normals2 = np.column_stack((-edges2[:, 1], edges2[:, 0]))
        
        all_normals = np.vstack((normals1, normals2))
        
        for normal in all_normals:
            # Project v1 and v2 onto normal
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            min1 = np.min(proj1)
            max1 = np.max(proj1)
            min2 = np.min(proj2)
            max2 = np.max(proj2)
            
            # Check for gap
            if max1 < min2 or max2 < min1:
                return False # Separating axis found, no intersection
        
        return True # No separating axis, they intersect

    def objective(params):
        """
        params: [cx1, cy1, a1, cx2, cy2, a2, ..., cx11, cy11, a11, ocx, ocy, oside, oang]
        a_i in degrees.
        ocx, ocy: outer center.
        oside: outer side length.
        oang: outer angle in degrees.
        
        Minimize: oside + penalty
        """
        if time.time() > deadline:
            return 1e10
            
        # Extract inner hexagon params
        inner_params = params[:N*3]
        inner_hexes = []
        for i in range(N):
            cx = inner_params[i*3]
            cy = inner_params[i*3+1]
            a = inner_params[i*3+2]
            verts = get_hex_vertices(cx, cy, a)
            inner_hexes.append(verts)
            
        ocx = params[N*3]
        ocy = params[N*3+1]
        oside = params[N*3+2]
        oang = params[N*3+3]
        
        # Ensure oside is positive
        if oside <= 0:
            oside = 1e-6
            
        # Calculate outer hex vertices
        outer_verts = get_hex_vertices(ocx, ocy, oang)
        # Scale outer hex? 
        # get_hex_vertices returns unit hex (side 1).
        # We need side length oside.
        # Vertices of unit hex are at distance 1.
        # Vertices of hex with side S are at distance S.
        # So scale outer_verts by oside.
        outer_verts = outer_verts * oside
        
        penalty = 0.0
        
        # 1. Containment penalty
        # Check if all inner vertices are inside outer hex.
        # Use SAT or point-in-poly.
        # Since outer is convex, we can check if any inner vertex is outside.
        # Or check if outer edges separate inner vertices.
        # Simpler: For each inner vertex, check if it is inside outer hex.
        # Point in convex polygon: check cross products.
        
        # To be differentiable-ish for gradient-based optimization, we can use a soft penalty.
        # However, L-BFGS-B doesn't strictly require gradients, it approximates them.
        # We can just use a hard penalty for violations.
        
        # For each inner hexagon, get its 6 vertices.
        for i in range(N):
            v_inner = inner_hexes[i]
            for j in range(6):
                px, py = v_inner[j]
                
                # Check if (px, py) is inside outer_hex
                # Transform to local frame of outer hex
                a_rad = math.radians(oang)
                cos_a = math.cos(a_rad)
                sin_a = math.sin(a_rad)
                
                # Rotate point relative to center
                dx = px - ocx
                dy = py - ocy
                
                rx = dx * cos_a + dy * sin_a
                ry = -dx * sin_a + dy * cos_a
                
                # Outer hex is centered at origin in local frame, scaled by oside.
                # Local coords of unit hex vertices: (1,0), (0.5, sqrt3/2), etc.
                # Inequalities for point (x,y) to be inside hex of side S (centered at 0, angle 0):
                # The hex is intersection of 3 strips.
                # Normals at 30, 90, 150 degrees.
                # Apothem = S * sqrt(3) / 2.
                # proj_n <= apothem for all 6 normals (or 3 with abs).
                
                apothem = oside * SQRT3 / 2
                
                # Normal 1: 30 deg
                n1 = (math.cos(math.radians(30)), math.sin(math.radians(30)))
                p1 = rx * n1[0] + ry * n1[1]
                if p1 > apothem:
                    penalty += (p1 - apothem)**2
                elif p1 < -apothem:
                    penalty += (-apothem - p1)**2
                    
                # Normal 2: 90 deg
                n2 = (0, 1)
                p2 = ry
                if p2 > apothem:
                    penalty += (p2 - apothem)**2
                elif p2 < -apothem:
                    penalty += (-apothem - p2)**2
                    
                # Normal 3: 150 deg
                n3 = (math.cos(math.radians(150)), math.sin(math.radians(150)))
                p3 = rx * n3[0] + ry * n3[1]
                if p3 > apothem:
                    penalty += (p3 - apothem)**2
                elif p3 < -apothem:
                    penalty += (-apothem - p3)**2

        # 2. Collision penalty
        # Check pairwise intersections of inner hexagons.
        # This is expensive.
        # Optimize: only check if bounding circles overlap.
        # Bounding circle radius is 1 (side 1 hex circumradius is 1).
        # If dist(centers) > 2, no collision.
        
        for i in range(N):
            for j in range(i+1, N):
                v1 = inner_hexes[i]
                v2 = inner_hexes[j]
                
                cx1, cy1 = v1[0,0] - 1, v1[0,1] # Rough center estimate? No, use params.
                # Better to use params directly for distance check
                cx1 = inner_params[i*3]
                cy1 = inner_params[i*3+1]
                cx2 = inner_params[j*3]
                cy2 = inner_params[j*3+1]
                
                dist_sq = (cx1-cx2)**2 + (cy1-cy2)**2
                if dist_sq > 4.0: # 2^2
                    continue
                
                # Check SAT
                if check_collision(v1, v2):
                    # They overlap. Add penalty.
                    # Penalty should be proportional to overlap depth?
                    # Hard to compute depth easily.
                    # Just add a large constant penalty to push them apart.
                    # But this creates a non-smooth landscape.
                    # Alternative: Minimize distance between closest points?
                    # For now, use a simple penalty: 100 * (1 - dist/2) if dist < 2?
                    # Or just a fixed penalty.
                    penalty += 100.0
                    
        return oside + penalty

    # Initial guess
    # Place hexagons in a loose cluster around origin.
    # Outer hex large enough to contain them.
    
    best_score = float('inf')
    best_params = None
    
    # Multi-start
    n_restarts = 20
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Random initial configuration
        # Centers within radius 3
        angles = np.random.uniform(0, 2*math.pi, N)
        radii = np.random.uniform(0, 2, N)
        
        x0 = []
        for i in range(N):
            cx = radii[i] * math.cos(angles[i])
            cy = radii[i] * math.sin(angles[i])
            a = np.random.uniform(0, 360)
            x0.extend([cx, cy, a])
            
        # Outer hex: center 0,0, side 10, angle 0
        x0.extend([0.0, 0.0, 10.0, 0.0])
        
        x0 = np.array(x0)
        
        # Bounds
        # Inner centers: -10 to 10
        # Inner angles: 0 to 360
        # Outer center: -10 to 10
        # Outer side: 1 to 20
        # Outer angle: 0 to 360
        
        bounds = []
        for i in range(N):
            bounds.append((-10, 10))
            bounds.append((-10, 10))
            bounds.append((0, 360))
            
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((1, 20))
        bounds.append((0, 360))
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6})
            
            if res.success or res.fun < best_score:
                # Verify validity
                # Check if penalty is 0
                # Re-evaluate
                f = res.fun
                # Extract oside
                oside = res.x[N*3+2]
                
                # Check if valid packing
                inner_params = res.x[:N*3]
                inner_hexes = []
                for i in range(N):
                    cx = inner_params[i*3]
                    cy = inner_params[i*3+1]
                    a = inner_params[i*3+2]
                    verts = get_hex_vertices(cx, cy, a)
                    inner_hexes.append(verts)
                    
                # Check collisions
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if check_collision(inner_hexes[i], inner_hexes[j]):
                            valid = False
                            break
                    if not valid:
                        break
                
                # Check containment
                if valid:
                    ocx = res.x[N*3]
                    ocy = res.x[N*3+1]
                    oang = res.x[N*3+3]
                    outer_verts = get_hex_vertices(ocx, ocy, oang) * oside
                    
                    for i in range(N):
                        for j in range(6):
                            px, py = inner_hexes[i][j]
                            # Check if inside outer
                            a_rad = math.radians(oang)
                            cos_a = math.cos(a_rad)
                            sin_a = math.sin(a_rad)
                            dx = px - ocx
                            dy = py - ocy
                            rx = dx * cos_a + dy * sin_a
                            ry = -dx * sin_a + dy * cos_a
                            
                            apothem = oside * SQRT3 / 2
                            p1 = rx * math.cos(math.radians(30)) + ry * math.sin(math.radians(30))
                            p2 = ry
                            p3 = rx * math.cos(math.radians(150)) + ry * math.sin(math.radians(150))
                            
                            if abs(p1) > apothem + 1e-6 or abs(p2) > apothem + 1e-6 or abs(p3) > apothem + 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    if oside < best_score:
                        best_score = oside
                        best_params = res.x
                        
        except Exception as e:
            pass

    # Fallback if no valid solution found
    if best_params is None:
        # Simple linear arrangement
        best_params = np.zeros(N*3 + 4)
        for i in range(N):
            best_params[i*3] = i * 2.0 # Spaced by 2 (circumradius*2)
            best_params[i*3+1] = 0.0
            best_params[i*3+2] = 0.0
        best_params[N*3] = (N-1) # Center
        best_params[N*3+1] = 0.0
        best_params[N*3+2] = N + 5 # Large side
        best_params[N*3+3] = 0.0
        best_score = best_params[N*3+2]
        
    # Format result
    inner_rows = []
    for i in range(N):
        cx = best_params[i*3]
        cy = best_params[i*3+1]
        a = best_params[i*3+2]
        inner_rows.append([cx, cy, a])
        
    ocx = best_params[N*3]
    ocy = best_params[N*3+1]
    oside = best_params[N*3+2]
    oang = best_params[N*3+3]
    
    return inner_rows, (ocx, ocy), oside, oang

if __name__ == "__main__":
    r = construct()
    print(r)
