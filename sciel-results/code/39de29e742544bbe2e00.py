"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = math.sqrt(3)
    
    # Helper to compute support function of a unit regular hexagon
    # with angle alpha in direction phi
    def support_hex(alpha, phi):
        # The support function of a regular hexagon with circumradius 1
        # is cos(phi - alpha_vertex) where alpha_vertex is the closest vertex angle.
        # Vertices are at alpha + k*pi/3.
        # We can compute this by finding the nearest multiple of pi/3 to (phi - alpha)
        # and taking the cosine of the difference.
        # diff = (phi - alpha) % (pi/3)
        # if diff > pi/6: diff = pi/3 - diff
        # return cos(diff)
        # This works because the support function is symmetric and periodic.
        diff = (phi - alpha) % (math.pi / 3)
        if diff > math.pi / 6:
            diff = math.pi / 3 - diff
        return math.cos(diff)

    # Precompute outer hexagon normals (rotation 0)
    # Normals are at pi/6 + k*pi/3
    outer_normals = [math.pi / 6 + k * math.pi / 3 for k in range(6)]
    
    def objective(params):
        # params: [x1, y1, a1, ..., xN, yN, aN, s]
        # N hexagons, each 3 params, plus s
        n_params = 3 * N
        s = params[n_params]
        
        if s < 2.0:
            return 1e6
            
        # Extract positions and angles
        pos = np.empty((N, 2))
        angs = np.empty(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i + 1]
            angs[i] = params[3*i + 2]
        
        penalty = 0.0
        
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                
                # Check 12 directions: 6 normals of i, 6 normals of j
                # Normals of i: angs[i] + pi/6 + k*pi/3
                # Normals of j: angs[j] + pi/6 + k*pi/3
                
                # We can just check the 12 directions
                # Optimization: if distance between centers is large, skip?
                # Max extent of hexagon is 1. If dist > 2, no overlap.
                dist = math.hypot(dx, dy)
                if dist > 2.0:
                    continue
                
                for k in range(6):
                    # Direction from i
                    phi_i = angs[i] + math.pi / 6 + k * math.pi / 3
                    # Support of i in phi_i
                    h_i = math.sqrt(3) / 2 # Inradius, since phi_i is normal to edge
                    # Support of j in -phi_i
                    h_j = support_hex(angs[j], -phi_i)
                    
                    # Projection of vector (j-i) on phi_i
                    proj = dx * math.cos(phi_i) + dy * math.sin(phi_i)
                    
                    if h_i + h_j > proj:
                        penalty += (h_i + h_j - proj) ** 2
                    
                    # Direction from j
                    phi_j = angs[j] + math.pi / 6 + k * math.pi / 3
                    # Support of j in phi_j
                    h_j_val = math.sqrt(3) / 2
                    # Support of i in -phi_j
                    h_i_val = support_hex(angs[i], -phi_j)
                    
                    # Projection of vector (i-j) on phi_j
                    # which is - (dx * cos(phi_j) + dy * sin(phi_j))
                    proj_j = -(dx * math.cos(phi_j) + dy * math.sin(phi_j))
                    
                    if h_j_val + h_i_val > proj_j:
                        penalty += (h_j_val + h_i_val - proj_j) ** 2

        # Check containment
        inradius_outer = s * math.sqrt(3) / 2
        
        for i in range(N):
            for k in range(6):
                phi = outer_normals[k]
                # Support of inner hex i in direction phi
                h_i = support_hex(angs[i], phi)
                # Projection of center i on phi
                proj = pos[i, 0] * math.cos(phi) + pos[i, 1] * math.sin(phi)
                
                if h_i + proj > inradius_outer:
                    penalty += (h_i + proj - inradius_outer) ** 2
        
        return s + penalty

    # Initial guess: hexagonal packing
    # Rows: 3, 4, 4
    rows = [3, 4, 4]
    init_pos = []
    init_angs = []
    y_offset = 0.0
    for i, count in enumerate(rows):
        x_start = -(count - 1) * sqrt3 / 2
        for j in range(count):
            x = x_start + j * sqrt3
            y = y_offset
            init_pos.append([x, y])
            init_angs.append(0.0)
        y_offset += 1.5
    
    # Center
    xs = [p[0] for p in init_pos]
    ys = [p[1] for p in init_pos]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in init_pos:
        p[0] -= cx
        p[1] -= cy
        
    # Initial s estimate
    max_dist = max(math.hypot(p[0], p[1]) for p in init_pos)
    s_init = max_dist + 1.5
    
    bounds = []
    for i in range(N):
        bounds.append((-5, 5))
        bounds.append((-5, 5))
        bounds.append((-math.pi, math.pi))
    bounds.append((3.0, 15.0))
    
    # Multi-start DE
    best_s = float('inf')
    best_params = None
    
    for restart in range(5):
        if restart == 0:
            x0 = []
            for i in range(N):
                x0.append(init_pos[i][0])
                x0.append(init_pos[i][1])
                x0.append(init_angs[i])
            x0.append(s_init)
        else:
            x0 = None # Let DE choose
            
        try:
            result = differential_evolution(
                objective,
                bounds,
                x0=x0,
                maxiter=100,
                popsize=15,
                tol=1e-4,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42 + restart,
                polish=False
            )
            if result.fun < best_s:
                best_s = result.fun
                best_params = result.x
        except Exception:
            continue
            
    # If DE failed, use heuristic
    if best_params is None:
        pos = init_pos
        angs = init_angs
        s = s_init
    else:
        pos = []
        angs = []
        for i in range(N):
            pos.append([best_params[3*i], best_params[3*i + 1]])
            angs.append(best_params[3*i + 2])
        s = best_params[3*N]
        
    # Final check and cleanup
    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], math.degrees(angs[i])])
        
    return inner, (0.0, 0.0), s, 0.0
# EVOLVE-BLOCK-END
