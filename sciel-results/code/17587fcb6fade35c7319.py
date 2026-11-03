"""
Strategy: Topology-Driven Constraint Relaxation for Hexagon Packing.

Instead of generic random search, this approach:
1. Defines a specific contact topology for 11 hexagons (a central hexagon, 
   a ring of 6, and 4 outer ones in a specific arrangement).
2. Uses scipy.optimize.minimize to find the configuration that minimizes the 
   bounding hexagon size while maintaining the defined contacts (touching).
3. This is structurally different from random perturbation because it 
   explicitly enforces geometric constraints (distances between centers) 
   corresponding to a known dense packing motif.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _min_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
    """Compute minimum outer side length to contain all inner hexagons."""
    cx, cy = outer_center
    pos = np.asarray(inner_positions, dtype=np.float64)
    ang = np.radians(np.asarray(inner_angles, dtype=np.float64))
    
    # Generate all vertices for all hexagons
    # For N hexagons, each has 6 vertices.
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_angles = ang[:, None] + base_angles[None, :]
    
    offsets_x = np.cos(total_angles)
    offsets_y = np.sin(total_angles)
    
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    
    all_verts_x = verts_x.ravel()
    all_verts_y = verts_y.ravel()
    
    # Transform to outer hexagon coordinate system
    dx = all_verts_x - cx
    dy = all_verts_y - cy
    
    a_outer = np.radians(outer_angle)
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    # Support function for regular hexagon centered at origin
    # h(v) = max( |lx|, |lx + ly/sqrt3|, |lx - ly/sqrt3| ) is not quite right for side length S.
    # For a regular hexagon with side S centered at origin, aligned with axes:
    # Vertices are (S,0), (S/2, S*sqrt3/2), ...
    # The constraint for a point (lx, ly) to be inside is:
    # |lx| <= S
    # |lx + ly/sqrt3| <= S
    # |lx - ly/sqrt3| <= S
    
    c1 = np.abs(lx)
    c2 = np.abs(lx + ly / sqrt3)
    c3 = np.abs(lx - ly / sqrt3)
    
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def _check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-7):
    """Check if packing is valid: no overlaps, all inside outer hexagon."""
    cx, cy = outer_center
    verts_list = []
    for x, y, ang in inner:
        verts = _hex_vertices(x, y, 1.0, ang)
        verts_list.append(verts)
    
    # Check overlaps using SAT
    def sat_overlap(v1, v2):
        # Project onto all axes
        axes = []
        for v in [v1, v2]:
            for i in range(6):
                p1 = v[i]
                p2 = v[(i+1)%6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-9:
                    axes.append(normal / norm)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
                return False
        return True

    for i in range(N):
        for j in range(i+1, N):
            if sat_overlap(verts_list[i], verts_list[j]):
                return False
            
    # Check containment
    outer_verts = _hex_vertices(cx, cy, outer_side, outer_angle)
    # Simple point-in-polygon check for all vertices
    def point_in_poly(pt, poly):
        x, y = pt
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    for verts in verts_list:
        for v in verts:
            if not point_in_poly(v, outer_verts):
                return False
    return True

def _initial_configuration():
    """
    Generate a specific initial configuration for 11 hexagons.
    Topology:
    - Center: (0,0)
    - Ring 1: 6 hexagons at distance 2*sqrt(3) from center (touching).
      Angles: 0, 60, 120, 180, 240, 300 degrees.
    - Ring 2: 4 hexagons. To minimize bounding box, we place them in the "gaps" 
      of the ring 1, specifically at angles 30, 150, 210, 330 degrees.
      Distance for touching Ring 1 hexagons:
      Ring 1 hex at 0 deg is at (2*sqrt(3), 0).
      Ring 2 hex at 30 deg needs to touch Ring 1 hex at 0 and 60.
      Distance from center to Ring 2 hex:
      Let d2 be distance. Angle 30 deg.
      Distance to Ring 1 hex (0 deg) is 2*sqrt(3).
      d2^2 + (2*sqrt(3))^2 - 2*d2*(2*sqrt(3))*cos(30) = (2*sqrt(3))^2
      d2^2 - 2*d2*(2*sqrt(3))*(sqrt(3)/2) = 0
      d2^2 - 6*d2 = 0 => d2 = 6.
      
      So Ring 2 hexagons are at distance 6 from center, at angles 30, 150, 210, 330.
    """
    sqrt3 = np.sqrt(3)
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    
    # 0: Center
    pos[0] = [0, 0]
    
    # 1-6: Ring 1
    for i in range(6):
        a = i * np.pi / 3
        pos[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
        
    # 7-10: Ring 2 (4 hexagons)
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6] # 30, 150, 210, 330
    for i in range(4):
        a = ring2_angles[i]
        pos[i+7] = [6.0 * np.cos(a), 6.0 * np.sin(a)]
        
    return pos, ang

def _objective(params):
    """
    Objective: Minimize outer side length + penalty for breaking contacts.
    Params: [x_0...x_10, y_0...y_10, ang_0...ang_10]
    """
    n = N
    pos = params[:2*n].reshape(n, 2)
    ang = params[2*n:]
    
    side = _min_outer_side(pos, ang, (0.0, 0.0), 0.0)
    
    # Penalty for violating contact distances
    # We define specific pairs that should be touching (distance 2*sqrt(3))
    # In the initial config, all neighbors are touching.
    # To keep the structure rigid, we penalize deviations from 2*sqrt(3) for defined pairs.
    
    # Pairs (indices):
    # Center (0) to Ring 1 (1-6)
    # Ring 1 (i) to Ring 1 (i+1)
    # Ring 1 (i) to Ring 2 (touching)
    
    contacts = []
    # Center to Ring 1
    for i in range(1, 7):
        contacts.append((0, i))
    # Ring 1 adjacent
    for i in range(1, 7):
        contacts.append((i, i % 6 + 1))
    # Ring 1 to Ring 2
    # Ring 2 at 30 deg (idx 7) touches Ring 1 at 0 (idx 1) and 60 (idx 2)
    contacts.append((7, 1))
    contacts.append((7, 2))
    # Ring 2 at 150 deg (idx 8) touches Ring 1 at 120 (idx 4) and 180 (idx 5)
    contacts.append((8, 4))
    contacts.append((8, 5))
    # Ring 2 at 210 deg (idx 9) touches Ring 1 at 180 (idx 5) and 240 (idx 6)
    contacts.append((9, 5))
    contacts.append((9, 6))
    # Ring 2 at 330 deg (idx 10) touches Ring 1 at 300 (idx 3) and 0 (idx 1)
    contacts.append((10, 3))
    contacts.append((10, 1))
    
    penalty = 0.0
    target_dist = 2 * np.sqrt(3)
    for i, j in contacts:
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        d = np.sqrt(dx*dx + dy*dy)
        penalty += (d - target_dist)**2
        
    # Also penalize overlaps (hard constraint)
    # Simple distance check for centers
    for i in range(n):
        for j in range(i+1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < 2*sqrt3 - 0.1: # Allow some slack, but penalize deep overlap
                penalty += 100.0 * (2*sqrt3 - d)**2
                
    return side + 10.0 * penalty

def construct():
    global START_TIME, DEADLINE
    
    pos_init, ang_init = _initial_configuration()
    best_pos = pos_init.copy()
    best_ang = ang_init.copy()
    
    # Flatten for optimization
    x0 = np.concatenate([pos_init.ravel(), ang_init])
    
    # Use Nelder-Mead for local refinement of the topology
    # It handles non-smooth objectives reasonably well and doesn't require gradients
    best_side = _min_outer_side(best_pos, best_ang, (0.0, 0.0), 0.0)
    
    if time.time() < DEADLINE - 10.0:
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.success or res.fun < _objective(x0):
                x_opt = res.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                ang_opt = x_opt[2*N:]
                
                # Verify validity
                inner_list = [[float(pos_opt[i,0]), float(pos_opt[i,1]), float(ang_opt[i])] for i in range(N)]
                side_opt = _min_outer_side(pos_opt, ang_opt, (0.0, 0.0), 0.0)
                
                if _check_validity(inner_list, (0.0, 0.0), side_opt, 0.0):
                    if side_opt < best_side:
                        best_pos = pos_opt
                        best_ang = ang_opt
                        best_side = side_opt
        except Exception:
            pass
            
    # If optimization failed or didn't improve, try a few random restarts with the topology
    # to escape local minima
    if time.time() < DEADLINE - 5.0:
        for _ in range(10):
            if time.time() > DEADLINE - 5.0:
                break
            # Jitter the initial positions slightly
            pos_jit = pos_init + np.random.normal(0, 0.1, (N, 2))
            ang_jit = ang_init + np.random.normal(0, 0.1, N)
            
            x_jit = np.concatenate([pos_jit.ravel(), ang_jit])
            try:
                res = minimize(_objective, x_jit, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
                x_opt = res.x
                pos_opt = x_opt[:2*N].reshape(N, 2)
                ang_opt = x_opt[2*N:]
                
                inner_list = [[float(pos_opt[i,0]), float(pos_opt[i,1]), float(ang_opt[i])] for i in range(N)]
                side_opt = _min_outer_side(pos_opt, ang_opt, (0.0, 0.0), 0.0)
                
                if _check_validity(inner_list, (0.0, 0.0), side_opt, 0.0):
                    if side_opt < best_side:
                        best_pos = pos_opt
                        best_ang = ang_opt
                        best_side = side_opt
            except Exception:
                pass

    # Finalize
    inner_list = [[float(best_pos[i,0]), float(best_pos[i,1]), float(best_ang[i])] for i in range(N)]
    final_side = _min_outer_side(best_pos, best_ang, (0.0, 0.0), 0.0)
    
    # Final validity check
    if not _check_validity(inner_list, (0.0, 0.0), final_side, 0.0):
        # Fallback to initial configuration
        pos_f, ang_f = _initial_configuration()
        inner_list = [[float(pos_f[i,0]), float(pos_f[i,1]), float(ang_f[i])] for i in range(N)]
        final_side = _min_outer_side(pos_f, ang_f, (0.0, 0.0), 0.0)
        
    return inner_list, (0.0, 0.0), float(final_side), 0.0

if __name__ == "__main__":
    construct()
