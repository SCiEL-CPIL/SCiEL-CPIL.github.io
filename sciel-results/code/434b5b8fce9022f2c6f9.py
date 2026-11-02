"""Improved packing search with smooth SAT penalties and L-BFGS-B."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 12
SQRT3 = np.sqrt(3)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

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

    def get_edges(v):
        """Returns list of edge vectors and normals for a polygon."""
        edges = []
        normals = []
        n = len(v)
        for i in range(n):
            p1 = v[i]
            p2 = v[(i + 1) % n]
            e = p2 - p1
            # Normal pointing outward? For SAT, direction doesn't matter for gap calc if we take abs
            nrm = np.array([-e[1], e[0]])
            norm = np.linalg.norm(nrm)
            if norm > 1e-12:
                nrm = nrm / norm
            edges.append(e)
            normals.append(nrm)
        return edges, normals

    def sat_penalty(v1, v2):
        """
        Calculates a smooth penalty based on SAT projection overlaps.
        Returns 0 if disjoint, > 0 if overlapping.
        """
        # Get normals for both polygons
        e1, n1 = get_edges(v1)
        e2, n2 = get_edges(v2)
        
        all_normals = n1 + n2
        penalty = 0.0
        
        for nrm in all_normals:
            # Project both polygons onto the normal
            proj1 = np.dot(v1, nrm)
            proj2 = np.dot(v2, nrm)
            
            min1 = np.min(proj1)
            max1 = np.max(proj1)
            min2 = np.min(proj2)
            max2 = np.max(proj2)
            
            # Gap: if max1 < min2 or max2 < min1, they are separated
            # Overlap depth is negative gap
            gap1 = min2 - max1
            gap2 = min1 - max2
            
            # If gap > 0, no overlap on this axis.
            # If gap < 0, overlap.
            # We want to penalize overlap.
            # Smooth penalty: square of the negative gap
            # We take the most negative gap (deepest overlap) on this axis? 
            # Actually, for SAT, if ANY axis has a positive gap, they are disjoint.
            # If ALL axes have negative gaps, they overlap.
            # The "overlap depth" is roughly the minimum gap (most negative).
            # Let's use the minimum of the gaps (which will be <= 0 if overlapping on this axis).
            min_gap = min(gap1, gap2)
            
            if min_gap < 0:
                # Overlap on this axis
                # Penalty is proportional to the square of the overlap depth
                penalty += (min_gap * min_gap)
                
        return penalty

    def containment_penalty(inner_verts, center, angle_deg, side):
        """
        Penalty for inner vertices being outside the outer hexagon.
        """
        # Outer hexagon vertices
        outer_verts = hexagon_vertices(center[0], center[1], side, angle_deg)
        e_out, n_out = get_edges(outer_verts)
        
        # For each inner vertex, check if it's inside.
        # A point is inside if it is on the "inner" side of all edges.
        # For a convex polygon, we can check the signed distance to each edge.
        # If any distance is negative (outside), add penalty.
        
        penalty = 0.0
        for v in inner_verts:
            for i in range(6):
                # Edge from outer_verts[i] to outer_verts[i+1]
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                edge = p2 - p1
                # Normal pointing INWARDS?
                # Standard normal is (-dy, dx). 
                # For a CCW polygon, inward normal is (dy, -dx) normalized?
                # Let's determine orientation.
                # hexagon_vertices generates CCW.
                # So inward normal is (edge[1], -edge[0]) / norm.
                nrm = np.array([edge[1], -edge[0]])
                norm = np.linalg.norm(nrm)
                if norm > 1e-12:
                    nrm = nrm / norm
                    
                # Vector from p1 to v
                vec = v - p1
                dist = np.dot(vec, nrm)
                
                # If dist < 0, v is outside
                if dist < 0:
                    penalty += (dist * dist)
                    
        return penalty

    def objective(params):
        """
        Objective function combining disjointness and containment penalties.
        params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, cx, cy, c_angle, side]
        """
        inner = []
        all_inner_verts = []
        
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner.append((x, y, a))
            verts = hexagon_vertices(x, y, 1.0, a)
            all_inner_verts.append(verts)
            
        cx = params[3*N]
        cy = params[3*N+1]
        c_angle = params[3*N+2]
        side = params[3*N+3]
        
        # 1. Disjointness Penalty
        disp_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                # Fast check: distance between centers
                dx = inner[i][0] - inner[j][0]
                dy = inner[i][1] - inner[j][1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < 4.0: # Max dist between centers for touching hexagons is 2
                    v1 = all_inner_verts[i]
                    v2 = all_inner_verts[j]
                    disp_penalty += sat_penalty(v1, v2)
        
        # 2. Containment Penalty
        # Flatten all inner vertices
        flat_verts = np.vstack(all_inner_verts)
        cont_penalty = containment_penalty(flat_verts, np.array([cx, cy]), c_angle, side)
        
        # 3. Side length penalty (we want to minimize side)
        # We add side to the objective to drive it down
        # But we must ensure containment.
        # If containment is violated, penalty is huge.
        
        return side + 100.0 * disp_penalty + 100.0 * cont_penalty

    def honeycomb_init():
        """Create a 3x4 honeycomb grid of hexagons."""
        params = np.zeros(3*N + 4)
        positions = []
        # Row 0: 4 hexagons
        for col in range(4):
            positions.append((col * 2.0, 0.0, 0.0))
        # Row 1: 3 hexagons
        for col in range(3):
            positions.append((col * 2.0 + 1.0, SQRT3, 0.0))
        # Row 2: 4 hexagons
        for col in range(4):
            positions.append((col * 2.0, 2*SQRT3, 0.0))
        # Row 3: 1 hexagon
        positions.append((1.0, 3*SQRT3, 0.0))
        
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
            
        # Center and angle
        cx = 2.0
        cy = 1.5 * SQRT3
        params[3*N] = cx
        params[3*N+1] = cy
        params[3*N+2] = 0.0
        params[3*N+3] = 5.0 # Initial guess for side
        
        return params

    def flower_init():
        """Create a flower pattern."""
        params = np.zeros(3*N + 4)
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
        params[3*N+3] = 5.0
        
        return params

    def random_init():
        """Random initialization within a reasonable box."""
        params = np.zeros(3*N + 4)
        for i in range(N):
            params[3*i] = random.uniform(-3, 3)
            params[3*i+1] = random.uniform(-3, 3)
            params[3*i+2] = random.uniform(0, 60)
            
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        params[3*N+3] = 6.0
        return params

    # Main Search Loop
    best_params = None
    best_score = np.inf
    
    inits = [honeycomb_init(), flower_init()]
    # Add a few random inits
    for _ in range(3):
        inits.append(random_init())
        
    for init_params in inits:
        if time.time() > deadline:
            break
            
        # Optimize
        try:
            # Use L-BFGS-B for speed and handling of bounds (though we don't have strict bounds, it's good)
            # Or Nelder-Mead for robustness
            result = minimize(
                objective,
                init_params,
                method='L-BFGS-B',
                options={
                    'maxiter': 2000,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
                
        except Exception as e:
            pass

        # If L-BFGS-B fails or gets stuck, try Nelder-Mead from the result
        if time.time() > deadline - 10:
            continue
            
        try:
            start_nm = result.x if 'result' in locals() else init_params
            result_nm = minimize(
                objective,
                start_nm,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if result_nm.fun < best_score:
                best_score = result_nm.fun
                best_params = result_nm.x.copy()
        except Exception:
            pass

    # Final Polish
    if best_params is not None and time.time() < deadline - 5.0:
        # Local search around the best solution
        for _ in range(500):
            if time.time() > deadline - 2.0:
                break
            new_params = best_params.copy()
            # Perturb a random subset of variables
            for _ in range(3):
                idx = random.randint(0, 3*N + 3)
                new_params[idx] += random.gauss(0, 0.01)
            new_score = objective(new_params)
            if new_score < best_score:
                best_score = new_score
                best_params = new_params.copy()

    if best_params is None:
        # Fallback
        best_params = honeycomb_init()

    # Extract results
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
        
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    angle = float(best_params[3*N+2])
    
    # Recalculate exact minimal side for the final configuration
    # The optimizer found a 'side' that satisfied the soft constraints.
    # We should compute the *exact* minimal side that contains these hexagons.
    
    def compute_exact_min_side(inner, center, angle_deg):
        """
        Computes the minimal side length of a regular hexagon at (center, angle_deg)
        that contains all inner hexagons.
        """
        all_verts = []
        for i in range(N):
            v = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            all_verts.append(v)
            
        all_verts = np.vstack(all_verts)
        
        # The outer hexagon is defined by 6 half-planes.
        # For a regular hexagon of side S, center C, rotation A:
        # The distance from center to each side is S * sqrt(3) / 2.
        # The normals to the sides are at angles A + 30 + k * 60 degrees.
        
        min_side = 0.0
        for k in range(6):
            normal_angle = np.radians(angle_deg) + np.radians(30) + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            
            # Project all vertices onto this normal
            offsets = all_verts - np.array(center)
            projections = np.dot(offsets, normal)
            max_proj = np.max(projections)
            
            # The side length S must satisfy: S * sqrt(3) / 2 >= max_proj
            # So S >= max_proj * 2 / sqrt(3)
            required_side = max_proj * 2.0 / SQRT3
            if required_side > min_side:
                min_side = required_side
                
        return min_side

    side = compute_exact_min_side(inner, center, angle)
    
    # Final Validation
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            v1 = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            v2 = hexagon_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            # Use strict SAT check
            if sat_penalty(v1, v2) > 1e-6: # If penalty is non-zero, they overlap
                valid = False
                break
        if not valid:
            break
            
    # Check containment
    all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
    if containment_penalty(all_verts, np.array(center), angle, side) > 1e-6:
        valid = False

    if not valid:
        # If invalid, fallback to honeycomb
        params = honeycomb_init()
        inner = []
        for i in range(N):
            inner.append([float(params[3*i]), float(params[3*i+1]), float(params[3*i+2])])
        center = (float(params[3*N]), float(params[3*N+1]))
        angle = float(params[3*N+2])
        side = compute_exact_min_side(inner, center, angle)

    return inner, center, side, angle
# EVOLVE-BLOCK-END
