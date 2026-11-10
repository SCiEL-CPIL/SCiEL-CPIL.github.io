# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def hexagon_vertices(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon with center (cx,cy), side s, angle angle_deg."""
    angles = np.radians(angle_deg + 60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def get_edge_normals(verts):
    """Get outward-pointing edge normals for a hexagon given its vertices."""
    normals = []
    center = np.mean(verts, axis=0)
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        edge = p2 - p1
        n = np.array([-edge[1], edge[0]])
        mid = (p1 + p2) / 2
        if np.dot(n, mid - center) < 0:
            n = -n
        n = n / np.linalg.norm(n)
        normals.append(n)
    return np.array(normals)


def min_gap_between_hexes(v1, v2):
    """
    Compute the minimum gap between two hexagons using their 12 edge normals.
    Returns the maximum over all normals of the gap (negative means overlap).
    """
    n1 = get_edge_normals(v1)
    n2 = get_edge_normals(v2)
    all_normals = np.vstack([n1, n2])
    
    max_gap = -np.inf
    for n in all_normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        gap = max(min2 - max1, min1 - max2)
        max_gap = max(max_gap, gap)
    
    return max_gap


def check_separation(hex1, hex2, tol=1e-6):
    """
    Check if two hexagons are separated: on at least one of their 12 edge normals,
    the projected intervals are apart by more than tol.
    """
    x1, y1, a1 = hex1
    x2, y2, a2 = hex2
    
    v1 = hexagon_vertices(x1, y1, 1.0, a1)
    v2 = hexagon_vertices(x2, y2, 1.0, a2)
    
    gap = min_gap_between_hexes(v1, v2)
    return gap > tol


def calculate_min_side(inner_hex_data, outer_angle_deg):
    """
    Calculate the minimum side length S of an outer regular hexagon centered at origin
    with given angle that contains all inner hexagons.
    
    For a regular hexagon with side S and angle a, the 6 edge outward normals point at
    angles a + 30 + 60*k for k=0..5. The support function in direction n is S*sqrt(3)/2
    (the inradius). So we need: max over all vertices of dot(v, n_k) <= S*sqrt(3)/2
    for each normal direction n_k.
    Thus S = max_k (max_v dot(v, n_k)) * 2/sqrt(3)
    """
    normal_angles = [outer_angle_deg + 30 + 60 * k for k in range(6)]
    normals = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in normal_angles])
    
    all_vertices = []
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        verts = hexagon_vertices(x, y, 1.0, angle)
        all_vertices.append(verts)
    
    all_vertices = np.vstack(all_vertices)
    
    # For each normal direction, find the max projection
    max_projs = np.max(all_vertices @ normals.T, axis=0)
    max_proj = np.max(max_projs)
    
    # S * sqrt(3)/2 = max_proj => S = max_proj * 2/sqrt(3)
    side = max_proj * 2.0 / math.sqrt(3)
    
    return side


def is_valid_configuration(inner_hex_data, tol=1e-6):
    """Check if all pairs of inner hexagons are separated."""
    for i in range(len(inner_hex_data)):
        for j in range(i + 1, len(inner_hex_data)):
            if not check_separation(inner_hex_data[i], inner_hex_data[j], tol):
                return False
    return True


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    
    Strategy: Start with a 4-3-4 honeycomb cluster, then optimize positions and outer angle.
    """
    start_time = time.time()
    deadline = 620.0  # Safety margin before 630s kill
    
    h = math.sqrt(3) / 2  # height of unit hexagon = sqrt(3)/2
    
    # 4-3-4 honeycomb arrangement
    # Row 0 (bottom): 4 hexagons at y = 0
    # Row 1 (middle): 3 hexagons at y = 2*h (offset by 1.5 in x)
    # Row 2 (top): 4 hexagons at y = 4*h
    
    coords = []
    # Bottom row: 4 hexagons
    for i in range(4):
        coords.append([i * 3.0, 0.0])
    # Middle row: 3 hexagons, offset
    for i in range(3):
        coords.append([1.5 + i * 3.0, 2 * h])
    # Top row: 4 hexagons
    for i in range(4):
        coords.append([i * 3.0, 4 * h])
    
    # Center the cluster
    cx = np.mean([c[0] for c in coords])
    cy = np.mean([c[1] for c in coords])
    for c in coords:
        c[0] -= cx
        c[1] -= cy
    
    # Scale up slightly to ensure separation
    scale = 1.001
    for c in coords:
        c[0] *= scale
        c[1] *= scale
    
    inner_hex_data = np.zeros((11, 3))
    for i, c in enumerate(coords):
        inner_hex_data[i, 0] = c[0]
        inner_hex_data[i, 1] = c[1]
        inner_hex_data[i, 2] = 0.0
    
    # First, find the best outer angle for this configuration
    best_side = float('inf')
    best_outer_angle = 0.0
    best_positions = inner_hex_data.copy()
    
    # Coarse scan of outer angles
    for outer_angle_deg in np.arange(0, 30.01, 1.0):
        if time.time() - start_time > deadline:
            break
        side = calculate_min_side(inner_hex_data, outer_angle_deg)
        if side < best_side:
            best_side = side
            best_outer_angle = outer_angle_deg
    
    # Fine scan around best angle
    if time.time() - start_time < deadline:
        for outer_angle_deg in np.arange(max(0, best_outer_angle - 1.0), min(30, best_outer_angle + 1.0) + 0.01, 0.05):
            if time.time() - start_time > deadline:
                break
            side = calculate_min_side(inner_hex_data, outer_angle_deg)
            if side < best_side:
                best_side = side
                best_outer_angle = outer_angle_deg
    
    # Now try to optimize positions using a simple gradient-free approach
    # We'll use a stochastic local search
    if time.time() - start_time < deadline:
        best_positions = inner_hex_data.copy()
        best_side_local = best_side
        best_angle_local = best_outer_angle
        
        # Try small perturbations
        np.random.seed(42)
        for iteration in range(200):
            if time.time() - start_time > deadline:
                break
            
            # Create a perturbed version
            perturbed = best_positions.copy()
            noise_scale = 0.01 * (1 - iteration / 300)
            for i in range(11):
                perturbed[i, 0] += np.random.randn() * noise_scale
                perturbed[i, 1] += np.random.randn() * noise_scale
            
            # Check validity
            valid = True
            for i in range(11):
                for j in range(i + 1, 11):
                    if not check_separation(perturbed[i], perturbed[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Try a few outer angles
                for oa in np.arange(max(0, best_angle_local - 2.0), min(30, best_angle_local + 2.0) + 0.01, 0.5):
                    if time.time() - start_time > deadline:
                        break
                    s = calculate_min_side(perturbed, oa)
                    if s < best_side_local:
                        best_side_local = s
                        best_angle_local = oa
                        best_positions = perturbed.copy()
        
        best_side = best_side_local
        best_outer_angle = best_angle_local
        inner_hex_data = best_positions
    
    # Final fine-tuning of outer angle
    if time.time() - start_time < deadline:
        for outer_angle_deg in np.arange(max(0, best_outer_angle - 0.5), min(30, best_outer_angle + 0.5) + 0.001, 0.01):
            if time.time() - start_time > deadline:
                break
            s = calculate_min_side(inner_hex_data, outer_angle_deg)
            if s < best_side:
                best_side = s
                best_outer_angle = outer_angle_deg
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    outer_hex_side_length = best_side
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
