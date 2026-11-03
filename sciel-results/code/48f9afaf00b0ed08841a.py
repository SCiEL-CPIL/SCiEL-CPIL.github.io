import math
import time
import random
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0

    # Lattice vectors for hexagonal packing
    # Distance between centers of touching unit hexagons is sqrt(3)
    # We add a small epsilon to ensure strict disjointness
    eps = 0.01
    d = math.sqrt(3) + eps
    
    u = np.array([d, 0.0])
    v = np.array([d/2.0, d*math.sqrt(3)/2.0])
    
    # Ring 0
    centers = [np.array([0.0, 0.0])]
    
    # Ring 1
    # Neighbors of center
    ring1_vectors = [
        u,
        v,
        v - u,
        -u,
        -v,
        u - v
    ]
    for vec in ring1_vectors:
        centers.append(vec.copy())
        
    # Ring 2 Candidates
    # Neighbors of Ring 1 that are not in Ring 0 or 1
    ring2_candidates = []
    # For each ring1 center, find its 6 neighbors
    # A neighbor of P is P +/- u, P +/- v, P +/- (v-u)
    basis = [u, v, v-u, -u, -v, u-v]
    
    for i in range(6):
        p1 = ring1_vectors[i]
        for b in basis:
            p2 = p1 + b
            # Check if p2 is in Ring 0 or Ring 1
            is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
            is_r1 = False
            for p1j in ring1_vectors:
                if np.allclose(p2, p1j):
                    is_r1 = True
                    break
            if not is_r0 and not is_r1:
                # Avoid duplicates in candidates list
                duplicate = False
                for pc in ring2_candidates:
                    if np.allclose(p2, pc):
                        duplicate = True
                        break
                if not duplicate:
                    ring2_candidates.append(p2)
                    
    # Sort Ring 2 candidates by distance to origin
    ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
    
    # Pick 5 closest
    for i in range(5):
        centers.append(ring2_candidates[i].copy())
        
    # Now we have 12 centers
    # Verify count
    assert len(centers) == 12
    
    # Convert to list of [x, y, angle]
    # Rotate hexagons by 30 degrees so they are 'flat-top' relative to the x-axis,
    # allowing them to touch at distance sqrt(3) along the x-axis.
    inner = [[c[0], c[1], 30.0] for c in centers]
    
    # Function to get vertices of a hexagon
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    # Collect all vertices
    all_verts = []
    for x, y, a in inner:
        verts = get_hex_vertices(x, y, 1.0, a)
        all_verts.extend(verts)
        
    # Find minimum outer hexagon side
    # The outer hexagon is centered at the centroid of the inner cluster?
    # Or we can optimize the center too.
    # For symmetric clusters, the centroid is a good guess.
    cx = sum(p[0] for p in all_verts) / len(all_verts)
    cy = sum(p[1] for p in all_verts) / len(all_verts)
    
    best_side = float('inf')
    best_angle = 0.0
    
    # Scan outer hexagon angle
    # The hexagon is defined by 6 half planes.
    # Normal angles: theta + 30 + 60*k
    # Condition: dot(p - c, n) <= S * sqrt(3) / 2
    # S >= 2/sqrt(3) * max(dot(p-c, n))
    
    for theta_deg in np.linspace(0, 60, 61):
        theta = math.radians(theta_deg)
        max_proj = 0.0
        for k in range(6):
            normal_angle = theta + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for px, py in all_verts:
                proj = (px - cx) * nx + (py - cy) * ny
                if proj > max_proj:
                    max_proj = proj
        s = 2.0 * max_proj / math.sqrt(3)
        if s < best_side:
            best_side = s
            best_angle = theta_deg
            
    outer_center = (cx, cy)
    outer_side = best_side
    outer_angle = best_angle
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
