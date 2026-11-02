"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Returns vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
        angle = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)

    def hexagons_disjoint_fast(h1, h2):
        """Fast check for disjointness using SAT (Separating Axis Theorem)."""
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # If dist > 2.0, definitely disjoint (max distance between any two points on unit hexagons)
        if dist_sq > 4.0:
            return True
        
        # Precise SAT check for all other cases
        v1 = hexagon_vertices(h1[0], h1[1], 1.0, h1[2])
        v2 = hexagon_vertices(h2[0], h2[1], 1.0, h2[2])
        
        # Check all 12 edge normals (6 from each hexagon)
        for i in range(6):
            # Edge normal from h1
            e1 = v1[(i+1) % 6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            nn1 = np.linalg.norm(n1)
            if nn1 > 1e-12:
                n1 = n1 / nn1
                proj1 = np.dot(v1, n1)
                proj2 = np.dot(v2, n1)
                if np.max(proj1) < np.min(proj2) - 1e-9 or np.max(proj2) < np.min(proj1) - 1e-9:
                    return True
            
            # Edge normal from h2
            e2 = v2[(i+1) % 6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            nn2 = np.linalg.norm(n2)
            if nn2 > 1e-12:
                n2 = n2 / nn2
                proj1 = np.dot(v1, n2)
                proj2 = np.dot(v2, n2)
                if np.max(proj1) < np.min(proj2) - 1e-9 or np.max(proj2) < np.min(proj1) - 1e-9:
                    return True
        return False

    def compute_min_side_fast(inner, center, angle):
        """Vectorized computation of minimum outer side length required to contain all inner hexagons."""
        sqrt3 = np.sqrt(3)
        max_extent = 0.0
        
        # Precompute all vertices for all inner hexagons
        all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        
        # Center offset
        offsets = all_verts - np.array(center)
        
        # Edge normals of outer hexagon
        for k in range(6):
            normal_angle = np.radians(angle) + np.radians(30) + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            
            # Dot product of all offsets with normal
            extents = np.dot(offsets, normal)
            max_e = np.max(extents)
            if max_e > max_extent:
                max_extent = max_e
        
        return max_extent * 2.0 / sqrt3

    def is_valid(inner):
        """Check if all hexagons are pairwise disjoint."""
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint_fast(inner[i], inner[j]):
                    return False
        return True

    def objective(params):
        """Objective: minimize outer side length while ensuring disjointness."""
        # Unpack
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+2]
        
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint_fast(inner[i], inner[j]):
                    return 1e6
        
        # Compute side
        side = compute_min_side_fast(inner, center, angle)
        
        return side

    def hexagonal_grid_init():
        """Create a 3x4 honeycomb-like grid initialization that is valid and compact."""
        params = np.zeros(3*N + 3)
        # Hexagonal grid: 3 rows of 4 hexagons
        # For hexagons of side 1, the distance between centers in a hex grid:
        # Horizontal neighbors: 2 * sqrt(3)/2 * 1 = sqrt(3) ~ 1.732
        # Vertical neighbors: 1.5 (3/2 of side length for flat-top)
        # Actually for flat-top hexagons:
        # Horizontal spacing: 3/2 * side * 2 = 3 * side = 3 (for touching)
        # No wait. Let me recalculate.
        # For flat-top regular hexagons of side s:
        # Width = 2s, Height = sqrt(3)*s
        # Horizontal distance between centers for touching: 3s/2 * 2 = 3s? No.
        # Actually, for flat-top hexagons:
        # The horizontal distance between centers of adjacent columns is 3s/2 * 2 = 3s? 
        # Let me think again. A flat-top hexagon has vertices at angles 0, 60, 120, 180, 240, 300.
        # The rightmost vertex is at (s, 0). The next hexagon to the right, if touching, 
        # would have its leftmost vertex at (s, 0). Its center would be at (s + s, 0) = (2s, 0)?
        # No, the leftmost vertex of a flat-top hexagon is at (-s, 0).
        # So if hexagon A is at (0,0), its rightmost point is (s, 0).
        # Hexagon B to the right, touching, has its leftmost point at (s, 0).
        # So center of B is at (s + s, 0) = (2s, 0).
        # Wait, that's for pointy-top. Let me check flat-top.
        # Flat-top: vertices at (s,0), (s/2, sqrt(3)s/2), (-s/2, sqrt(3)s/2), (-s, 0), (-s/2, -sqrt(3)s/2), (s/2, -sqrt(3)s/2)
        # So width is 2s, height is sqrt(3)s.
        # For two flat-top hexagons side by side (touching at the right/left vertices):
        # Center distance = 2s.
        # But they can also be offset vertically. In a honeycomb:
        # Column 0: y = 0, sqrt(3)s, 2*sqrt(3)s, ...
        # Column 1: y = sqrt(3)s/2, 3*sqrt(3)s/2, ... (offset by half)
        # Horizontal distance between columns: 3s/2 * 2 = 3s? No.
        # Actually, for flat-top hexagons in a honeycomb:
        # The horizontal distance between centers of adjacent columns is 3s/2.
        # Wait, let me look this up properly.
        # For flat-top hexagons of side s:
        # - Width (horizontal): 2s
        # - Height (vertical): sqrt(3)s
        # - Horizontal distance between centers in same row: 3s (since they touch at vertices? No, 2s)
        # Hmm, I'm getting confused. Let me just use a simple grid that's definitely valid.
        # Place hexagons in a 3x4 grid with spacing 2.1 (slightly more than 2.0 to ensure disjointness)
        # This gives a 3x4 grid = 12 hexagons.
        
        s = 1.0
        spacing = 2.1  # Slightly more than 2.0 to ensure disjointness
        
        positions = []
        for row in range(3):
            for col in range(4):
                x = col * spacing
                y = row * spacing
                positions.append((x, y, 0))
        
        # Center the grid
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        
        for i in range(N):
            params[3*i] = positions[i][0] - cx
            params[3*i+1] = positions[i][1] - cy
            params[3*i+2] = positions[i][2]
        
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        return params

    def nelder_mead(params, f, max_iter=2000, tol=1e-8, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0):
        """Nelder-Mead optimization."""
        n = len(params)
        simplex = [params.copy() for _ in range(n + 1)]
        for i in range(n):
            simplex[i+1] = params.copy()
            simplex[i+1][i] += 0.1 * (1.0 if abs(params[i]) < 1e-10 else params[i])
        
        fvals = [f(s) for s in simplex]
        
        for _ in range(max_iter):
            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if abs(fvals[0] - fvals[-1]) < tol:
                break
            
            # Centroid of all but worst point
            centroid = np.zeros(n)
            for i in range(n):
                centroid += simplex[i]
            centroid /= n
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[n])
            fr = f(xr)
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[n])
                fe = f(xe)
                if fe < fr:
                    simplex[n] = xe
                    fvals[n] = fe
                else:
                    simplex[n] = xr
                    fvals[n] = fr
            elif fr < fvals[n-1]:
                simplex[n] = xr
                fvals[n] = fr
            else:
                # Contraction
                if fr < fvals[n]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                    fc = f(xc)
                    if fc <= fr:
                        simplex[n] = xc
                        fvals[n] = fc
                    else:
                        # Shrink
                        for i in range(1, n+1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = f(simplex[i])
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[n])
                    fc = f(xc)
                    if fc <= fvals[n]:
                        simplex[n] = xc
                        fvals[n] = fc
                    else:
                        # Shrink
                        for i in range(1, n+1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = f(simplex[i])
        
        # Return best point
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]

    # Start with a valid hexagonal grid initialization
    best_params = hexagonal_grid_init()
    best_score = objective(best_params)
    
    # Run Nelder-Mead optimization
    if time.time() < deadline - 10.0:
        best_params, best_score = nelder_mead(best_params, objective, max_iter=5000)
    
    # Polish with simulated annealing
    if time.time() < deadline - 5.0:
        current_params = best_params.copy()
        current_score = best_score
        
        T = 0.5
        T_min = 0.001
        cooling_rate = 0.9995
        step_size = 0.1
        
        polish_start = time.time()
        while time.time() - polish_start < 10.0:
            new_params = current_params.copy()
            idx = random.randint(0, 3*N + 2)
            new_params[idx] += random.gauss(0, step_size * (0.5 + 0.5 * T))
            
            new_score = objective(new_params)
            
            if new_score < current_score or random.random() < np.exp(-(new_score - current_score) / T):
                current_params = new_params
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_params = current_params.copy()
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
    
    # Extract best
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    angle = float(best_params[3*N+2])
    side = compute_min_side_fast(inner, center, angle)
    
    # Final check: ensure validity
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not hexagons_disjoint_fast(inner[i], inner[j]):
                valid = False
                break
        if not valid: break
    
    if not valid:
        # Fallback: use the hexagonal grid
        inner = []
        spacing = 2.1
        positions = []
        for row in range(3):
            for col in range(4):
                x = col * spacing
                y = row * spacing
                positions.append((x, y, 0))
        
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        
        for pos in positions:
            inner.append([float(pos[0] - cx), float(pos[1] - cy), float(pos[2])])
        center = (0.0, 0.0)
        angle = 0.0
        side = compute_min_side_fast(inner, center, angle)

    return inner, center, side, angle
# EVOLVE-BLOCK-END
