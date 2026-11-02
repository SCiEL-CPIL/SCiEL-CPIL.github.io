"""Improved packing search with Nelder-Mead and honeycomb initialization."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

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

    def objective(params):
        """Objective: minimize outer side length while ensuring disjointness with smooth penalty."""
        # Unpack
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+2]
        
        # Check disjointness with soft penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint_fast(inner[i], inner[j]):
                    penalty += 10.0
        
        if penalty > 0:
            return 1e4 + penalty
        
        # Compute side
        side = compute_min_side_fast(inner, center, angle)
        
        return side

    def honeycomb_init():
        """Create a 3x4 honeycomb grid of hexagons - a valid compact starting configuration."""
        params = np.zeros(3*N + 3)
        # Honeycomb grid: 3 rows, 4 columns
        # Row 0: 4 hexagons
        # Row 1: 3 hexagons (offset)
        # Row 2: 4 hexagons
        # Actually for 12 hexagons, let's do a proper honeycomb pattern
        # Use axial coordinates for honeycomb
        
        # Simple approach: place in a rectangular grid with proper offsets
        # Hexagons in honeycomb: adjacent hexagons are 2*sqrt(3)/2 = sqrt(3) apart horizontally
        # and 3/2 apart vertically (for pointy-top) or similar
        
        # For flat-top hexagons of side 1:
        # Horizontal spacing: 2 (center to center for adjacent in same row)
        # Vertical spacing: sqrt(3) (for offset rows)
        
        # Let's use a 2-3-2-3-2 pattern or similar
        # Actually, simplest: just use a grid where we know it's valid
        
        # Place 12 hexagons in a 3x4 grid with honeycomb offsets
        positions = []
        # Row 0: 4 hexagons at y=0
        for col in range(4):
            positions.append((col * 2.0, 0.0, 0.0))
        # Row 1: 3 hexagons at y=sqrt(3), offset by 1
        for col in range(3):
            positions.append((col * 2.0 + 1.0, np.sqrt(3), 0.0))
        # Row 2: 4 hexagons at y=2*sqrt(3)
        for col in range(4):
            positions.append((col * 2.0, 2*np.sqrt(3), 0.0))
        # Row 3: 1 hexagon at y=3*sqrt(3), offset
        positions.append((1.0, 3*np.sqrt(3), 0.0))
        
        # That's 4+3+4+1 = 12
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
        
        # Center and angle
        params[3*N] = 2.0  # approximate center x
        params[3*N+1] = 1.5 * np.sqrt(3)  # approximate center y
        params[3*N+2] = 0.0
        
        return params

    def flower_init():
        """Create a flower pattern: 1 center, 6 ring, 5 outer ring."""
        params = np.zeros(3*N + 3)
        positions = []
        positions.append((0, 0, 0))
        for i in range(6):
            theta = i * np.pi / 3
            positions.append((2*np.cos(theta), 2*np.sin(theta), 0))
        for i in range(5):
            theta = i * 2*np.pi / 5 + np.pi/5
            positions.append((4*np.cos(theta), 4*np.sin(theta), 0))
        
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
        
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        return params

    # Try multiple initializations
    best_params = None
    best_score = np.inf
    
    inits = [honeycomb_init(), flower_init()]
    
    for init_params in inits:
        if time.time() > deadline:
            break
        
        # Try Nelder-Mead optimization
        try:
            result = minimize(
                objective,
                init_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
        
        # Also try simulated annealing from this init
        if time.time() > deadline - 20:
            continue
        
        current_params = init_params.copy()
        current_score = objective(current_params)
        if current_score < best_score:
            best_score = current_score
            best_params = current_params.copy()
        
        T = 0.5
        T_min = 0.001
        cooling_rate = 0.999
        step_size = 0.2
        max_steps = 30000
        
        for _ in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            new_params = current_params.copy()
            n_perturb = 1 if random.random() < 0.7 else 2
            for _ in range(n_perturb):
                idx = random.randint(0, 3*N + 2)
                current_step = step_size * (0.5 + 0.5 * T)
                new_params[idx] += random.gauss(0, current_step)
            
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

    # Polish pass
    if best_params is not None and time.time() < deadline - 3.0:
        polish_start = time.time()
        for _ in range(2000):
            if time.time() - polish_start > 8.0:
                break
            new_params = best_params.copy()
            idx = random.randint(0, 3*N + 2)
            new_params[idx] += random.gauss(0, 0.02)
            new_score = objective(new_params)
            if new_score < best_score:
                best_score = new_score
                best_params = new_params.copy()

    if best_params is None:
        # Fallback
        best_params = honeycomb_init()
        best_score = objective(best_params)

    # Extract best
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    angle = float(best_params[3*N+2])
    side = compute_min_side_fast(inner, center, angle)
    
    # Final validation
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not hexagons_disjoint_fast(inner[i], inner[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to honeycomb
        params = honeycomb_init()
        inner = []
        for i in range(N):
            inner.append([float(params[3*i]), float(params[3*i+1]), float(params[3*i+2])])
        center = (float(params[3*N]), float(params[3*N+1]))
        angle = float(params[3*N+2])
        side = compute_min_side_fast(inner, center, angle)

    return inner, center, side, angle
# EVOLVE-BLOCK-END
