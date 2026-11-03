"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Precompute constants
    SQRT3 = np.sqrt(3)
    # Outer hexagon normals (flat-top, apothem = S * SQRT3 / 2)
    # Normals point to midpoints of edges.
    # For flat-top hex (vertices at 0, 60, ...), normals are at 30, 90, ...
    outer_normals = np.array([[np.cos(np.pi/6 + k*np.pi/3), np.sin(np.pi/6 + k*np.pi/3)] for k in range(6)])
    
    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon."""
        thetas = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(thetas), cy + s * np.sin(thetas)])

    def compute_supports(verts, normals):
        """Compute support function h(n) = max(v dot n) for each normal in normals."""
        # verts: (6, 2), normals: (M, 2)
        # Returns (M,)
        return np.max(verts @ normals.T, axis=0)

    def objective(params):
        """
        params: 3*N array [cx0, cy0, ang0, cx1, cy1, ang1, ...]
        Returns: S_outer + penalty
        """
        n_hex = len(params) // 3
        S_outer = 0.0
        penalty = 0.0
        
        # Store vertices for all hexagons to reuse in penalty
        all_verts = []
        
        for i in range(n_hex):
            cx = params[3*i]
            cy = params[3*i+1]
            angle = params[3*i+2]
            verts = hex_vertices(cx, cy, angle)
            all_verts.append(verts)
            
            # Compute S_outer contribution
            # We need max over all vertices v and all outer normals n of |v dot n|
            # Since hexagon is convex and centered near origin, we can just check v dot n and -v dot n
            # Or simply max(v dot n) and min(v dot n) -> range? No, it's max |v dot n|.
            # Actually, for a set of points, max |v dot n| = max(max(v dot n), -min(v dot n))
            # Let's just compute projections.
            projs = verts @ outer_normals.T
            # projs shape (6, 6)
            # We want max absolute value across all vertices and normals
            S_outer = max(S_outer, np.max(np.abs(projs)))
            
        # Convert max projection to side length S
        # Condition: v dot n <= S * SQRT3 / 2
        # So S >= 2 / SQRT3 * max |v dot n|
        S_outer = (2.0 / SQRT3) * S_outer
        
        # Compute Overlap Penalty
        # For each pair (i, j), check separation.
        # Separating axes are normals of hex i and normals of hex j.
        for i in range(n_hex):
            v_i = all_verts[i]
            # Get normals for hex i
            # Edge normals of a hexagon rotated by angle 'angle'
            # Original normals (flat-top) are at 30, 90, ...
            # Rotated by angle: 30+angle, 90+angle, ...
            n_i_thetas = np.pi/6 + params[3*i+2] + np.arange(6) * np.pi/3
            n_i = np.column_stack([np.cos(n_i_thetas), np.sin(n_i_thetas)])
            
            for j in range(i+1, n_hex):
                v_j = all_verts[j]
                n_j_thetas = np.pi/6 + params[3*j+2] + np.arange(6) * np.pi/3
                n_j = np.column_stack([np.cos(n_j_thetas), np.sin(n_j_thetas)])
                
                # All candidate axes
                # We check h_i(n) - h_j(-n) > 0 for all n in N_i U N_j
                # If they overlap, for all n, h_i(n) + h_j(n) >= 2 * (some bound)? 
                # Actually, disjoint iff exists n such that h_i(n) + h_j(-n) < 0? No.
                # Disjoint iff exists n such that h_i(n) < h_j(-n) is false? 
                # Let's use the distance: d = min_n (h_j(-n) - h_i(n)). If d < 0, overlap.
                # Penalty = sum of (max(0, h_i(n) - h_j(-n)))^2 for n in N_i U N_j.
                
                # h_i(n) for n in N_i:
                h_i_n_i = compute_supports(v_i, n_i)
                # h_j(-n) for n in N_i:
                h_j_neg_n_i = compute_supports(v_j, -n_i)
                
                # h_i(n) for n in N_j:
                h_i_n_j = compute_supports(v_i, n_j)
                # h_j(-n) for n in N_j:
                h_j_neg_n_j = compute_supports(v_j, -n_j)
                
                # Overlap amounts
                # For axis n, if h_i(n) > h_j(-n), they overlap by h_i(n) - h_j(-n).
                # We sum squares of positive overlaps.
                overlap_i = h_i_n_i - h_j_neg_n_i
                overlap_j = h_i_n_j - h_j_neg_n_j
                
                # Use smooth max or just square positive parts
                # To make it differentiable, we can use 0.5 * (x + sqrt(x^2 + eps))^2 or just clip.
                # L-BFGS-B handles non-smoothness reasonably well if not too sharp.
                # Let's use a soft penalty: sum( (max(0, x))^2 )
                
                p1 = np.maximum(0, overlap_i)
                p2 = np.maximum(0, overlap_j)
                
                penalty += np.sum(p1**2) + np.sum(p2**2)
                
        return S_outer + 10.0 * penalty # Weight for penalty

    # Initial guess: Hexagonal lattice
    # Center at (0,0)
    params = np.zeros(3*N)
    params[0:3] = [0, 0, 0]
    
    idx = 1
    # Ring 1: 6 hexagons
    # Distance between centers of touching hexagons is 2 * apothem = SQRT3
    d1 = SQRT3
    for k in range(6):
        theta = k * np.pi/3
        cx = d1 * np.cos(theta)
        cy = d1 * np.sin(theta)
        params[3*idx] = cx
        params[3*idx+1] = cy
        params[3*idx+2] = 0
        idx += 1
        
    # Ring 2: 4 more to reach 11
    # In a perfect lattice, the next ring would be at distance 2*SQRT3.
    # We place 4 of them. To allow optimization, we can start them slightly closer or at the lattice position.
    # Starting at valid positions is safer.
    d2 = 2 * SQRT3
    # Angles for the 4 hexagons in the second ring.
    # In a hex lattice, the 2nd ring has 12 positions.
    # We pick 4. Let's pick angles 0, pi/2, pi, 3*pi/2? 
    # Or 0, pi/3, 2pi/3, pi?
    # Let's just pick 4 evenly spaced or aligned with lattice.
    # Lattice positions in ring 2 are at angles 0, 30, 60, 90... ? No.
    # Ring 1 angles: 0, 60, 120, 180, 240, 300.
    # Ring 2 "corners" are at 0, 60, ... same angles but further?
    # Actually, the "corners" of the hex cluster are at the same angles as ring 1.
    # The "sides" are at 30, 90, ...
    # Let's place the 4 at angles 0, pi/2, pi, 3*pi/2 to be symmetric?
    # Or 0, pi/3, 2*pi/3, pi?
    # Let's try 0, pi/2, pi, 3*pi/2.
    angles_r2 = [0, np.pi/2, np.pi, 3*np.pi/2]
    for k in range(4):
        theta = angles_r2[k]
        cx = d2 * np.cos(theta)
        cy = d2 * np.sin(theta)
        params[3*idx] = cx
        params[3*idx+1] = cy
        params[3*idx+2] = 0
        idx += 1

    # Optimize
    # Use L-BFGS-B for gradient-based optimization
    # We need to provide bounds? Angles can be unbounded, positions can be unbounded.
    # But practically, they won't go far.
    result = minimize(objective, params, method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
    
    opt_params = result.x
    
    inner = []
    for i in range(N):
        cx = opt_params[3*i]
        cy = opt_params[3*i+1]
        angle = np.degrees(opt_params[3*i+2])
        inner.append([cx, cy, angle])
    
    # Compute final S
    S = 0.0
    for i in range(N):
        cx = opt_params[3*i]
        cy = opt_params[3*i+1]
        angle = opt_params[3*i+2]
        verts = hex_vertices(cx, cy, angle)
        projs = verts @ outer_normals.T
        S = max(S, np.max(np.abs(projs)))
    S = (2.0 / SQRT3) * S
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
