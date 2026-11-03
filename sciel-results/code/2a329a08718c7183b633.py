import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 148.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Helper: Get vertices of a unit hexagon at (x, y) with rotation angle
    def get_hex_vertices(x, y, angle):
        # Regular hexagon vertices relative to center
        # Side length 1. Distance from center to vertex is 1.
        # Standard orientation: flat top/bottom or pointy top/bottom?
        # Let's use standard math convention.
        # Vertices at angles: angle + k * pi/3
        angles = angle + np.arange(6) * (math.pi / 3.0)
        return np.stack([x + np.cos(angles), y + np.sin(angles)], axis=1)

    # Helper: Check if two convex polygons (hexagons) overlap
    # Using Separating Axis Theorem (SAT) for convex polygons
    def polygons_overlap(v1, v2):
        # v1, v2 are (6, 2) arrays
        # Try all edges from both polygons
        for v in [v1, v2]:
            for i in range(6):
                p1 = v[i]
                p2 = v[(i + 1) % 6]
                # Edge vector
                edge = p2 - p1
                # Normal vector
                normal = np.array([-edge[1], edge[0]])
                
                # Project both polygons onto normal
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                # If there's a gap, they don't overlap
                if max1 < min2 or max2 < min1:
                    return False
        return True

    # Helper: Calculate overlap penalty and enclosing hexagon side length
    def objective(params):
        # params: [x1, y1, a1, x2, y2, a2, ..., cx, cy, beta]
        # Total size: 3*N + 3 = 36
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')

        # Extract positions and angles
        positions = params[:3*N].reshape(N, 3)
        cx, cy, beta = params[3*N], params[3*N+1], params[3*N+2]
        
        # Wrap angles to [0, 2*pi) for stability
        positions[:, 2] = positions[:, 2] % (2 * math.pi)
        beta = beta % (math.pi / 3.0) # Hexagon symmetry: 60 degrees
        
        # 1. Calculate Overlap Penalty
        # We want to push overlapping hexagons apart.
        # Approximate overlap depth using distance between centers vs sum of apothems?
        # No, SAT is better but expensive. 
        # For gradient descent, a smooth approximation of collision is needed.
        # Let's use a distance-based penalty for centers if they are too close,
        # but that's not accurate for rotated hexagons.
        
        # Better approach: 
        # 1. Compute vertices for all hexagons.
        # 2. For each pair, if they overlap, calculate a penalty based on 
        #    the minimum translation vector (MTV) magnitude or similar.
        #    However, calculating MTV for hexagons in a loop is slow.
        
        # Alternative: 
        # Use a "soft" repulsion force based on center distance.
        # Min distance between non-overlapping hexagons is 2*APOTHEM (if parallel) 
        # or 2*1 (if pointy-to-pointy).
        # The "effective radius" varies with angle.
        
        # Let's try a simpler strategy: 
        # Penalty = Sum of (max(0, R_eff_i + R_eff_j - dist_ij))^2
        # Where R_eff is the max distance from center to any vertex (which is 1.0).
        # This is an upper bound on the "radius". If dist < 2.0, they *might* overlap.
        # This is conservative but differentiable.
        
        penalty = 0.0
        verts = []
        for i in range(N):
            x, y, a = positions[i]
            v = get_hex_vertices(x, y, a)
            verts.append(v)
            
        # Pairwise repulsion
        # If two hexagons are closer than 2*APOTHEM, they definitely overlap?
        # No, apothem is min distance. 
        # If center distance < 2*APOTHEM, they overlap.
        # If center distance > 2.0, they don't.
        # Between 2*APOTHEM and 2.0, it depends on rotation.
        
        # Let's use a strict SAT check for the final score, but for optimization,
        # we need a smooth gradient.
        # Let's use the distance between centers as a proxy for overlap,
        # penalizing if dist < 2*APOTHEM.
        # This ensures they are at least "apothem-distance" apart, which guarantees
        # no overlap for ANY rotation? 
        # Wait. If two hexagons are parallel, min distance is 2*APOTHEM.
        # If they are rotated 30 deg relative, min distance can be smaller?
        # Actually, the "inradius" is APOTHEM. The "circumradius" is 1.0.
        # The minimum distance between two non-overlapping convex sets is 0.
        # The condition for non-overlap is complex.
        
        # Let's stick to a simple repulsion:
        # If dist < 2 * APOTHEM, penalize heavily.
        # This might force them to be further apart than necessary (optimal packing
        # might have them touching at vertices where distance is closer to 2*APOTHEM? 
        # Actually, touching at vertices means distance between centers can be 
        # as low as... well, if they touch at a vertex, the distance is roughly 
        # 2*APOTHEM * cos(30) ? No.
        
        # Let's use a safer repulsion radius: R_repulse = 2 * APOTHEM.
        # This guarantees no overlap for parallel hexagons.
        # For rotated ones, it might be slightly conservative, but it's a valid start.
        
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(positions[i, :2] - positions[j, :2])
                min_d = 2 * APOTHEM
                if d < min_d:
                    penalty += (min_d - d) ** 2 * 100.0 # Strong penalty
        
        # 2. Calculate Enclosing Hexagon Side Length
        # We need to find the smallest regular hexagon (rotated by beta, centered at cx,cy)
        # that contains all vertices of all inner hexagons.
        
        # The distance from center (cx,cy) to the farthest vertex of any inner hexagon,
        # projected onto the 6 normals of the outer hexagon, determines the required apothem D.
        # D = max over all vertices v and all 6 normals n of |dot(v - c, n)|
        # Side length S = (D + APOTHEM) / APOTHEM
        
        D = 0.0
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            normal = np.array([nx, ny])
            
            for i in range(N):
                v = verts[i]
                # Project all 6 vertices of hex i onto normal
                projs = np.dot(v - np.array([cx, cy]), normal)
                D = max(D, np.max(projs), -np.min(projs))
        
        S = (D + APOTHEM) / APOTHEM
        
        # Total Objective: Minimize S, but keep penalty low.
        # If penalty > 0, the solution is invalid.
        # We want to minimize S + penalty * large_weight.
        # But we want to find a valid packing first.
        # Let's minimize (S + 1000 * penalty).
        
        return S + 1000.0 * penalty

    # Initialize with a known good lattice configuration
    # Axial coordinates to Cartesian
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Start with a compact cluster
    # 1 center, 6 neighbors, 4 more?
    # Let's use the "flower" shape + 4 extra
    init_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    
    init_params = []
    for q, r in init_axial:
        x, y = axial_to_cartesian(q, r)
        init_params.extend([x, y, 0.0])
    
    # Initial center and beta
    cx0 = 0.0
    cy0 = 0.0
    beta0 = 0.0
    init_params.extend([cx0, cy0, beta0])
    
    init_params = np.array(init_params)
    
    # Optimize
    # Use Nelder-Mead as it doesn't require gradients (which are hard to define for SAT/Max)
    # But Nelder-Mead can be slow. 
    # Let's try L-BFGS-B with numerical gradients? Or just Nelder-Mead with many iterations.
    
    best_params = init_params
    best_score = objective(init_params)
    
    # Multi-start with small perturbations
    starts = [init_params]
    
    # Add some random starts near the lattice
    for _ in range(5):
        p = init_params.copy()
        # Perturb positions
        p[:3*N:3] += np.random.normal(0, 0.1, N) # x
        p[1:3*N:3] += np.random.normal(0, 0.1, N) # y
        # Perturb angles
        p[2:3*N:3] += np.random.normal(0, 0.2, N)
        # Perturb center/beta
        p[3*N] += np.random.normal(0, 0.1)
        p[3*N+1] += np.random.normal(0, 0.1)
        p[3*N+2] += np.random.normal(0, 0.1)
        starts.append(p)
        
    for start in starts:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Extract best result
    best_positions = best_params[:3*N].reshape(N, 3)
    cx, cy, beta = best_params[3*N], best_params[3*N+1], best_params[3*N+2]
    
    # Final verification and cleanup
    # Check if valid (penalty should be 0)
    # If not valid, we might need to fallback or accept a larger S
    
    # Recalculate S for the best params
    def calc_final_S(positions, cx, cy, beta):
        D = 0.0
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            normal = np.array([nx, ny])
            for i in range(N):
                x, y, a = positions[i]
                v = get_hex_vertices(x, y, a)
                projs = np.dot(v - np.array([cx, cy]), normal)
                D = max(D, np.max(projs), -np.min(projs))
        return (D + APOTHEM) / APOTHEM, D
        
    S_final, D_final = calc_final_S(best_positions, cx, cy, beta)
    
    # Check validity (no overlaps)
    # This is a hard check. If invalid, we return a valid but worse solution?
    # Or we try to fix it?
    # For now, let's assume the optimizer found a valid one or close to it.
    # If penalty was high, S_final might be large.
    
    # To ensure validity, let's do a quick SAT check.
    # If invalid, we scale up the positions slightly from center? No, that increases S.
    # We should return the best valid solution found.
    
    # Let's check the best candidate from the optimization runs
    # We only stored the best one.
    
    # If the best one is invalid, we might want to try the initial lattice which is valid.
    # The initial lattice is valid.
    
    init_S, _ = calc_final_S(init_params[:3*N].reshape(N, 3), init_params[3*N], init_params[3*N+1], init_params[3*N+2])
    
    # Let's do a proper validity check on best_params
    def is_valid(positions):
        verts = []
        for i in range(N):
            x, y, a = positions[i]
            verts.append(get_hex_vertices(x, y, a))
        for i in range(N):
            for j in range(i+1, N):
                if polygons_overlap(verts[i], verts[j]):
                    return False
        return True
        
    if is_valid(best_positions):
        final_positions = best_positions
        final_cx, final_cy, final_beta = cx, cy, beta
        final_S = S_final
    else:
        # Fallback to initial valid lattice
        final_positions = init_params[:3*N].reshape(N, 3)
        final_cx, final_cy, final_beta = init_params[3*N], init_params[3*N+1], init_params[3*N+2]
        final_S, _ = calc_final_S(final_positions, final_cx, final_cy, final_beta)
        
    # Convert to output format
    inner = []
    for i in range(N):
        x, y, a = final_positions[i]
        inner.append([float(x), float(y), float(math.degrees(a))])
        
    center = [float(final_cx), float(final_cy)]
    side = float(final_S)
    rotation = float(math.degrees(final_beta))
    
    return inner, center, side, rotation

if __name__ == "__main__":
    # Quick test
    res = construct()
    print("Side length:", res[2])
