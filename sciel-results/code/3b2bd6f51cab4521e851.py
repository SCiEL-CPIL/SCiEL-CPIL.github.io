import math
import time
import random
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

N = 12
DEADLINE = 110.0  # Safety margin

# Precompute unit hexagon vertices (side length 1)
# Angle 0: Flat top? Or pointy top?
# Standard: Vertices at angles 0, 60, ... 300.
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))

def construct():
    t0 = time.time()
    
    # Helper to rotate and translate hex vertices
    def get_hex_verts(pos, angle_deg):
        c = math.cos(math.radians(angle_deg))
        s = math.sin(math.radians(angle_deg))
        R = np.array([[c, -s], [s, c]])
        return _UNIT_HEX_VERTS @ R.T + pos

    # Check if two hexagons overlap using Separating Axis Theorem (SAT)
    # Returns 0 if disjoint, >0 if overlapping (approximate penetration)
    def hex_overlap_penalty(pos1, ang1, pos2, ang2):
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dist_sq = dx*dx + dy*dy
        # Max distance between centers for non-overlap is roughly 2 * circumradius = 2.
        # If dist > 2.1, definitely no overlap.
        if dist_sq > 4.5:
            return 0.0
            
        v1 = get_hex_verts(pos1, ang1)
        v2 = get_hex_verts(pos2, ang2)
        
        # Get edges of both hexagons to use as normals
        edges1 = np.roll(v1, -1, axis=0) - v1
        edges2 = np.roll(v2, -1, axis=0) - v2
        
        max_pen = 0.0
        
        for edges in [edges1, edges2]:
            for i in range(6):
                e = edges[i]
                # Normal vector
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-9: continue
                n = n / norm
                
                # Project both hexagons onto this normal
                proj1 = np.dot(v1, n)
                proj2 = np.dot(v2, n)
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                # Check for separation
                if max1 < min2 or max2 < min1:
                    # Separated on this axis, no overlap
                    return 0.0
                else:
                    # Overlap on this axis
                    # Penetration depth is the overlap of the intervals
                    pen = min(max1, max2) - max(min1, min2)
                    if pen > max_pen:
                        max_pen = pen
                    
        # If we didn't return 0, they overlap.
        # The max_pen is an overestimate of the true penetration depth,
        # but it's a valid positive penalty.
        return max_pen

    # Calculate the side length of the outer hexagon containing all inner hexagons
    def compute_outer_side(positions, angles):
        # Collect all vertices
        all_verts = []
        for i in range(N):
            v = get_hex_verts(positions[i], angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        # Center of the bounding box (mean of vertices is a good approximation for centroid of union)
        center = np.mean(all_verts, axis=0)
        offsets = all_verts - center
        
        # The outer hexagon can be rotated. We need to find the rotation phi
        # that minimizes the max projection.
        # The side length s of a regular hexagon is related to its width w by s = w / sqrt(3).
        # Width in direction theta is max(proj) - min(proj) = 2 * max(|proj|) if centered.
        # We want to minimize max_k (max_proj_k - min_proj_k) / sqrt(3).
        
        def eval_phi(phi_deg):
            phi = math.radians(phi_deg)
            # Normals of the outer hexagon sides
            # If outer hex is rotated by phi, its side normals are at phi + 0, 60, ... 300
            # Wait, standard hex normals are at 0, 60... if flat top?
            # Let's just check 6 directions spaced by 60 degrees.
            normal_angles = phi + np.arange(6) * (math.pi / 3)
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            
            max_width = 0
            for k in range(6):
                proj = offsets[:, 0] * nx[k] + offsets[:, 1] * ny[k]
                width = np.max(proj) - np.min(proj)
                if width > max_width:
                    max_width = width
            
            return max_width / math.sqrt(3)
            
        # Scan phi from 0 to 60
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 2):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        # Fine tune
        if best_side < 100: # Only if we have a valid packing
             try:
                 res = minimize_scalar_local(eval_phi, best_phi - 2, best_phi + 2)
                 if res.fun < best_side:
                     best_side = res.fun
                     best_phi = res.x
             except:
                 pass
                 
        return best_side, best_phi, center

    # Simple local minimization for scalar (avoid scipy dependency for tiny loops if needed, but scipy is available)
    from scipy.optimize import minimize_scalar
    def minimize_scalar_local(func, a, b):
        return minimize_scalar(func, bounds=(a, b), method='bounded', options={'xatol': 1e-6})

    # Objective function for optimizer
    def objective(params):
        # params: [x0, y0, a0, x1, y1, a1, ...]
        pos = np.array(params[0:3*N:3]).reshape(-1, 2) # This slicing is wrong
        # Correct unpacking
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
            
        # Penalty for overlap
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                p = hex_overlap_penalty(pos[i], ang[i], pos[j], ang[j])
                if p > 0:
                    overlap_penalty += p * p # Squared penalty
            
        if overlap_penalty > 1e-3:
            # Strong penalty to force disjointness
            return 100.0 + overlap_penalty
        
        side, phi, center = compute_outer_side(pos, ang)
        return side

    # Spectral Initialization
    def spectral_init():
        # Create a graph based on a honeycomb-like structure
        # 12 nodes. Let's use a 3x4 grid or a specific hexagonal cluster.
        # Coordinates for a hexagonal cluster of 12?
        # Center (0,0), 6 neighbors, 5 more?
        # Let's just use a random graph or a path?
        # Better: Use the "ideal" distances.
        # Construct a complete graph with weights w_ij = 1 / d_ij_ideal
        # d_ij_ideal = 1.8 (approx min distance)
        
        # To make it "spectral", we need a non-trivial graph structure.
        # Let's assume a 2D grid topology for the 12 nodes.
        # 3 rows of 4?
        # Indices:
        # 0 1 2 3
        # 4 5 6 7
        # 8 9 10 11
        
        adj = np.zeros((N, N))
        # Connect horizontal
        for r in range(3):
            for c in range(3):
                adj[r*4+c, r*4+c+1] = 1
        # Connect vertical
        for r in range(2):
            for c in range(4):
                adj[r*4+c, (r+1)*4+c] = 1
                
        # Compute Laplacian
        L = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i == j:
                    L[i, j] = adj[i, j].sum()
                else:
                    L[i, j] = -adj[i, j]
                    
        # Eigen decomposition
        try:
            vals, vecs = np.linalg.eigh(L)
            # Fiedler vector is vecs[:, 1] (2nd smallest eigenvalue)
            # We need 2D embedding. Use vecs[:, 1] and vecs[:, 2]
            v1 = vecs[:, 1]
            v2 = vecs[:, 2]
            
            # Normalize
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            
            # Scale to expected size
            # Ideal packing diameter ~ 4-5 units
            scale = 2.0
            pos = np.column_stack((v1 * scale, v2 * scale))
            
            # Add random perturbation
            pos += np.random.normal(0, 0.5, (N, 2))
            
            ang = np.zeros(N)
            
            # Unpack to params
            params = np.zeros(3*N)
            for i in range(N):
                params[3*i] = pos[i, 0]
                params[3*i+1] = pos[i, 1]
                params[3*i+2] = ang[i]
                
            return params
            
        except Exception as e:
            # Fallback to random
            return np.random.normal(0, 2.0, 3*N)

    best_side = float('inf')
    best_params = None
    best_phi = 0.0
    best_center = (0,0)
    
    # Multi-start optimization
    num_starts = 5
    time_per_start = (DEADLINE - 5.0) / num_starts
    
    for start in range(num_starts):
        t_start = time.time()
        
        # Generate initial params
        if start == 0:
            # Spectral
            init_params = spectral_init()
        elif start == 1:
            # Honeycomb
            a1 = np.array([math.sqrt(3), 0.0])
            a2 = np.array([math.sqrt(3)/2.0, 1.5])
            coords = [
                (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, -1), (-1, 1), (2, -1), (-2, 1),
                (1, 1), (-1, -1), (2, 0)
            ]
            init_params = np.zeros(3*N)
            for i, (u, v) in enumerate(coords):
                p = u * a1 + v * a2
                init_params[3*i] = p[0]
                init_params[3*i+1] = p[1]
                init_params[3*i+2] = 0
        else:
            # Random
            init_params = np.random.normal(0, 3.0, 3*N)
            
        # Optimize
        try:
            res = minimize(objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6})
            
            if res.fun < best_side and res.fun < 50: # Sanity check
                # Verify disjointness strictly
                pos = np.zeros((N, 2))
                ang = np.zeros(N)
                for i in range(N):
                    pos[i, 0] = res.x[3*i]
                    pos[i, 1] = res.x[3*i+1]
                    ang[i] = res.x[3*i+2]
                
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if hex_overlap_penalty(pos[i], ang[i], pos[j], ang[j]) > 1e-6:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    side, phi, center = compute_outer_side(pos, ang)
                    if side < best_side:
                        best_side = side
                        best_params = res.x
                        best_phi = phi
                        best_center = center
        except Exception as e:
            pass
            
        # Check time
        if time.time() - t_start > time_per_start:
            break

    # Final Polish with SLSQP or Powell if time remains
    if best_params is not None and time.time() - t0 < DEADLINE - 5.0:
        try:
            res = minimize(objective, best_params, method='Powell', 
                           options={'maxiter': 1000, 'xtol': 1e-6, 'ftol': 1e-8})
            if res.fun < best_side and res.fun < 50:
                pos = np.zeros((N, 2))
                ang = np.zeros(N)
                for i in range(N):
                    pos[i, 0] = res.x[3*i]
                    pos[i, 1] = res.x[3*i+1]
                    ang[i] = res.x[3*i+2]
                
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if hex_overlap_penalty(pos[i], ang[i], pos[j], ang[j]) > 1e-6:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    side, phi, center = compute_outer_side(pos, ang)
                    if side < best_side:
                        best_side = side
                        best_params = res.x
                        best_phi = phi
                        best_center = center
        except:
            pass

    # Fallback if no valid solution found
    if best_params is None:
        # Use honeycomb
        a1 = np.array([math.sqrt(3), 0.0])
        a2 = np.array([math.sqrt(3)/2.0, 1.5])
        coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0)
        ]
        best_params = np.zeros(3*N)
        for i, (u, v) in enumerate(coords):
            p = u * a1 + v * a2
            best_params[3*i] = p[0]
            best_params[3*i+1] = p[1]
            best_params[3*i+2] = 0
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = best_params[3*i]
            pos[i, 1] = best_params[3*i+1]
            ang[i] = best_params[3*i+2]
        best_side, best_phi, best_center = compute_outer_side(pos, ang)

    # Extract results
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
        
    center = (float(best_center[0]), float(best_center[1]))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
