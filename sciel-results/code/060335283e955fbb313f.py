import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0  # Safety margin below 150s

def _hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def _hex_edge_normals(rot_deg):
    base = np.radians(np.array([30, 90, 150, 210, 270, 330])) + np.radians(rot_deg)
    return np.column_stack((np.cos(base), np.sin(base)))

def _outer_hex_side_fast(points, outer_rot_deg):
    """
    Calculate the minimum side length of a regular hexagon with given rotation
    that contains all points.
    """
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    # The distance from center to the side (apothem) must be at least the max projection
    max_proj = np.max(np.abs(proj))
    # Side length S = 2 / sqrt(3) * apothem
    return (2.0 / np.sqrt(3.0)) * max_proj

def _check_disjoint(inner, tol=1e-9):
    """
    Check if inner hexagons are disjoint.
    inner: list of [x, y, angle_degrees]
    """
    n = len(inner)
    verts = []
    norms = []
    for i in range(n):
        x, y, rot = inner[i]
        v = _hex_vertices(x, y, 1.0, rot)
        nm = _hex_edge_normals(rot)
        verts.append(v)
        norms.append(nm)
        
    for i in range(n):
        for j in range(i + 1, n):
            # SAT check
            # We can simplify since they are regular hexagons of same size
            # Just check distance between centers? No, rotation matters.
            # Use SAT with all 12 axes (6 from each)
            min_overlap = float('inf')
            for k in range(6):
                n1 = norms[i][k]
                proj1 = verts[i] @ n1
                proj2 = verts[j] @ n1
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2 or max2 < min1:
                    # Separated
                    return True
                overlap = min(max1, max2) - max(min1, min2)
                if overlap < min_overlap:
                    min_overlap = overlap
            
            # If we get here, no separation axis found in first 6.
            # Check remaining 6
            for k in range(6):
                n2 = norms[j][k]
                proj1 = verts[i] @ n2
                proj2 = verts[j] @ n2
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2 or max2 < min1:
                    # Separated
                    return True
                overlap = min(max1, max2) - max(min1, min2)
                if overlap < min_overlap:
                    min_overlap = overlap
            
            if min_overlap > -tol:
                # Overlapping (or touching)
                return False
                
    return True

def _get_candidate_centers():
    """
    Generate centers for various compact topologies.
    Returns a list of arrays of shape (12, 2).
    """
    candidates = []
    sqrt3 = np.sqrt(3)
    
    # Topology 1: 1-6-5 Flower
    # Center
    centers = [[0, 0]]
    # Ring 1
    for k in range(6):
        ang = np.radians(60 * k)
        centers.append([sqrt3 * np.cos(ang), sqrt3 * np.sin(ang)])
    # Ring 2 (5 of 6)
    for k in range(5):
        ang = np.radians(60 * k)
        centers.append([2 * sqrt3 * np.cos(ang), 2 * sqrt3 * np.sin(ang)])
    centers = np.array(centers)
    # Center the cluster
    centers -= np.mean(centers, axis=0)
    candidates.append(centers)
    
    # Topology 2: 3x4 Grid (Staggered)
    # Rows of 4
    # Spacing x = sqrt3, y = 1.5
    # Row 0: 0, 1, 2, 3
    # Row 1: 0.5, 1.5, 2.5, 3.5 (offset)
    # Row 2: 0, 1, 2, 3
    centers = []
    for r in range(3):
        for c in range(4):
            x = c * sqrt3
            y = r * 1.5
            if r % 2 == 1:
                x += sqrt3 / 2.0
            centers.append([x, y])
    centers = np.array(centers)
    centers -= np.mean(centers, axis=0)
    candidates.append(centers)
    
    # Topology 3: 2x6 Grid (Staggered)
    centers = []
    for r in range(2):
        for c in range(6):
            x = c * sqrt3
            y = r * 1.5
            if r % 2 == 1:
                x += sqrt3 / 2.0
            centers.append([x, y])
    centers = np.array(centers)
    centers -= np.mean(centers, axis=0)
    candidates.append(centers)
    
    # Topology 4: 4x3 Grid (Staggered) - same as 3x4 but rotated 90 deg effectively?
    # Hex grid is not square. 3 rows of 4 is different from 4 rows of 3.
    centers = []
    for r in range(4):
        for c in range(3):
            x = c * sqrt3
            y = r * 1.5
            if r % 2 == 1:
                x += sqrt3 / 2.0
            centers.append([x, y])
    centers = np.array(centers)
    centers -= np.mean(centers, axis=0)
    candidates.append(centers)
    
    return candidates

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)
        
    best_L = float('inf')
    best_inner = None
    best_outer_rot = 0.0
    
    candidates = _get_candidate_centers()
    
    # For each candidate topology, find the optimal rotation
    # The relative rotation between the inner cluster and the outer hexagon
    # determines the bounding box size.
    # We can fix the outer hexagon at rot=0 and rotate the inner cluster.
    # Or fix inner at 0 and rotate outer.
    # Due to symmetry, we only need to search 0 to 30 degrees.
    
    for centers in candidates:
        if time_left() < 5:
            break
            
        # Precompute vertices for the cluster at rotation 0
        # We will rotate the whole cluster
        verts_list = []
        for c in centers:
            v = _hex_vertices(c[0], c[1], 1.0, 0.0)
            verts_list.append(v)
        all_verts = np.vstack(verts_list)
        
        # Search over rotation angle theta
        # The objective is L(theta) = _outer_hex_side_fast(all_verts_rotated, 0)
        # Actually, _outer_hex_side_fast computes the side length for a fixed outer rotation.
        # If we rotate the inner cluster by theta, it's equivalent to rotating the outer normals by -theta.
        # So we can just pass outer_rot_deg = -theta to the function with unrotated points.
        
        # Golden section search or grid search
        # Let's do a fine grid first, then refine
        thetas = np.linspace(0, 30, 301) # 0.1 deg steps
        
        min_L = float('inf')
        best_theta = 0
        
        for theta in thetas:
            L = _outer_hex_side_fast(all_verts, -theta)
            if L < min_L:
                min_L = L
                best_theta = theta
        
        # Refine around best_theta
        if time_left() > 2:
            # Narrow search
            thetas_fine = np.linspace(best_theta - 0.5, best_theta + 0.5, 101)
            for theta in thetas_fine:
                L = _outer_hex_side_fast(all_verts, -theta)
                if L < min_L:
                    min_L = L
                    best_theta = theta
        
        if min_L < best_L:
            best_L = min_L
            best_outer_rot = 0.0 # We fixed outer at 0, so inner is rotated by best_theta
            # Construct the inner hexagons with rotation best_theta
            # Wait, if we rotated the cluster by best_theta, the individual hexagons
            # are also rotated by best_theta.
            # However, hexagons have 6-fold symmetry. Rotation by best_theta is fine.
            
            # Verify disjointness
            # In the rigid body assumption, the relative positions are fixed.
            # Since the initial topology was disjoint (touching), and we just rotated the whole thing,
            # it remains disjoint.
            
            # We should still check to be safe.
            inner = []
            for c in centers:
                # Rotate center? No, we rotated the vertices in the objective.
                # To create the output, we need to rotate the centers and the hexagon angles.
                # Center rotation:
                rad = np.radians(best_theta)
                cx = c[0] * np.cos(rad) - c[1] * np.sin(rad)
                cy = c[0] * np.sin(rad) + c[1] * np.cos(rad)
                inner.append([cx, cy, best_theta])
            
            if _check_disjoint(inner, tol=1e-8):
                # Valid
                pass
            else:
                # Should not happen, but if it does, skip
                continue
                
    if best_inner is None:
        # Fallback: Use the best found L, construct inner
        # We need to reconstruct the best candidate's inner list
        # This logic is slightly messy. Let's restructure to keep the best inner.
        pass

    # Let's re-implement the loop to keep track of the best inner list directly.
    
    best_L = float('inf')
    best_inner = None
    best_outer_rot = 0.0
    
    for centers in candidates:
        if time_left() < 5:
            break
            
        verts_list = []
        for c in centers:
            v = _hex_vertices(c[0], c[1], 1.0, 0.0)
            verts_list.append(v)
        all_verts = np.vstack(verts_list)
        
        thetas = np.linspace(0, 30, 301)
        min_L = float('inf')
        best_theta = 0
        
        for theta in thetas:
            L = _outer_hex_side_fast(all_verts, -theta)
            if L < min_L:
                min_L = L
                best_theta = theta
        
        if time_left() > 2:
            thetas_fine = np.linspace(best_theta - 0.5, best_theta + 0.5, 101)
            for theta in thetas_fine:
                L = _outer_hex_side_fast(all_verts, -theta)
                if L < min_L:
                    min_L = L
                    best_theta = theta
        
        if min_L < best_L:
            best_L = min_L
            best_outer_rot = 0.0
            rad = np.radians(best_theta)
            inner = []
            for c in centers:
                cx = c[0] * np.cos(rad) - c[1] * np.sin(rad)
                cy = c[0] * np.sin(rad) + c[1] * np.cos(rad)
                inner.append([cx, cy, best_theta])
            
            if _check_disjoint(inner, tol=1e-8):
                best_inner = inner
                
    if best_inner is None:
        # Emergency fallback
        centers = candidates[0]
        inner = []
        for c in centers:
            inner.append([c[0], c[1], 0.0])
        best_L = _outer_hex_side_fast(np.vstack([_hex_vertices(c[0], c[1], 1.0, 0.0) for c in centers]), 0.0)
        best_outer_rot = 0.0
        best_inner = inner

    # Optional: Local optimization to squeeze out more
    # The rigid body assumption might be slightly suboptimal if "breathing" helps.
    # But given the time, and that hexagons in a honeycomb are locked, 
    # individual movement is very constrained.
    # We can try a quick L-BFGS on the full 39D problem starting from this best_inner.
    
    if time_left() > 20:
        def _objective(p, t0, deadline):
            if time.time() - t0 > deadline:
                return 1e6
            inner = p[:36].reshape(12, 3)
            ox = p[36]
            oy = p[37]
            oang = p[38]
            
            # Check disjointness
            if not _check_disjoint(inner, tol=1e-8):
                return 1e6
                
            all_pts = np.vstack([_hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2]) for i in range(12)])
            all_pts[:, 0] -= ox
            all_pts[:, 1] -= oy
            L = _outer_hex_side_fast(all_pts, oang)
            return L

        # Flatten best_inner
        init_params = np.concatenate((np.array(best_inner).flatten(), [0.0, 0.0, best_outer_rot]))
        
        try:
            res = minimize(
                _objective,
                init_params,
                args=(t0, DEADLINE - 5),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 1000}
            )
            if res.fun < best_L:
                best_L = res.fun
                inner_opt = res.x[:36].reshape(12, 3)
                if _check_disjoint(inner_opt, tol=1e-8):
                    best_inner = [[float(p[0]), float(p[1]), float(p[2])] for p in inner_opt]
                    best_outer_rot = float(res.x[38])
        except Exception:
            pass

    outer_center = (0.0, 0.0)
    
    # Final calculation of side length to return
    all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in best_inner])
    all_pts[:, 0] -= outer_center[0]
    all_pts[:, 1] -= outer_center[1]
    outer_side = _outer_hex_side_fast(all_pts, best_outer_rot)
    
    # Add tiny epsilon
    outer_side *= 1.00001

    return best_inner, outer_center, outer_side, best_outer_rot

