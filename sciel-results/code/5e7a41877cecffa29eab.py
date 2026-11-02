"""Packing 11 unit hexagons in minimal enclosing regular hexagon."""
import numpy as np
import time
from itertools import combinations

N = 11

def _min_enclosing_hexagon_optimal_center(points, angle_deg=0.0):
    """Find min side length hexagon with optimal center for given rotation angle."""
    a = np.radians(angle_deg)
    side_angles = (np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180) + a
    normals = np.column_stack([np.cos(side_angles), np.sin(side_angles)])
    
    best_S = float('inf')
    best_center = (0.0, 0.0)
    
    # Grid search for center
    xs = np.linspace(-2, 2, 30)
    ys = np.linspace(-2, 2, 30)
    
    for cx in xs:
        for cy in ys:
            shifted = points - np.array([cx, cy])
            projs = shifted @ normals.T
            max_abs = np.max(np.abs(projs), axis=0)
            S_candidate = np.max(max_abs) / (np.sqrt(3) / 2)
            if S_candidate < best_S:
                best_S = S_candidate
                best_center = (cx, cy)
    
    return best_S, best_center

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    sqrt3 = np.sqrt(3)
    neighbor_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180
    neighbors = np.column_stack([sqrt3 * np.cos(neighbor_angles), sqrt3 * np.sin(neighbor_angles)])
    
    base_positions = np.vstack([np.array([[0.0, 0.0]]), neighbors])
    
    second_ring = []
    for i in range(6):
        second_ring.append(2 * neighbors[i])
        j = (i + 1) % 6
        second_ring.append(neighbors[i] + neighbors[j])
    second_ring = np.array(second_ring)
    
    best_S = float('inf')
    best_combo = None
    best_angle = 0.0
    best_center = (0.0, 0.0)
    
    outer_angles = [0, 30, 60, 90, 120, 150]
    
    for outer_angle in outer_angles:
        if time.time() > deadline:
            break
        for combo in combinations(range(12), 4):
            if time.time() > deadline:
                break
            extra = second_ring[list(combo)]
            all_positions = np.vstack([base_positions, extra])
            
            S, center = _min_enclosing_hexagon_optimal_center(all_positions, outer_angle)
            if S < best_S:
                best_S = S
                best_combo = combo
                best_angle = outer_angle
                best_center = center
    
    if best_combo is None:
        # Fallback: just use base + first 4 second ring
        best_combo = (0, 1, 2, 3)
        best_angle = 0
        extra = second_ring[list(best_combo)]
        all_positions = np.vstack([base_positions, extra])
        best_S, best_center = _min_enclosing_hexagon_optimal_center(all_positions, best_angle)
    
    extra = second_ring[list(best_combo)]
    all_positions = np.vstack([base_positions, extra])
    
    inner = []
    for pos in all_positions:
        inner.append([pos[0], pos[1], 0.0])
    
    return inner, best_center, best_S, best_angle

