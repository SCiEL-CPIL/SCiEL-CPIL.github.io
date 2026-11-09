"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import permutations

def _rotations(w, h, d):
    """Return unique rotations of a box as (w, h, d) tuples."""
    seen = set()
    result = []
    for perm in permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits_at(x, y, z, bw, bh, bd, placed, W, H, D):
    """Check if a box of size (bw,bh,bd) fits at (x,y,z) without overlap."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + bw > W or y + bh > H or z + bd > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if not (x + bw <= px or px + pw <= x):
            if not (y + bh <= py or py + ph <= y):
                if not (z + bd <= pz or pz + pd <= z):
                    return False
    return True

def _face_points(placed, W, H, D):
    """Generate candidate positions from faces of placed boxes and container walls."""
    points = set()
    # Add origin
    points.add((0, 0, 0))
    
    # Add points based on existing placements
    for (px, py, pz, pw, ph, pd) in placed:
        # Points along x-axis
        points.add((px + pw, 0, 0))
        points.add((px + pw, py, 0))
        points.add((px + pw, py, pz))
        points.add((px, py + ph, 0))
        points.add((px, py + ph, pz))
        points.add((px, 0, pz + pd))
        points.add((px, py, pz + pd))
        points.add((px + pw, py + ph, 0))
        points.add((px + pw, py + ph, pz))
        points.add((px + pw, 0, pz + pd))
        points.add((px, py + ph, pz + pd))
        
        # Also add combinations
        for dx in [0, pw]:
            for dy in [0, ph]:
                for dz in [0, pd]:
                    points.add((px + dx, py + dy, pz + dz))
    
    # Filter to valid positions within container
    valid_points = []
    for (x, y, z) in points:
        if 0 <= x <= W and 0 <= y <= H and 0 <= z <= D:
            valid_points.append((x, y, z))
    
    return valid_points

def _score_position(x, y, z, bw, bh, bd, placed, W, H, D):
    """Score a position based on how tightly it packs (prefer lower z, then y, then x)."""
    # Prefer positions that are as low as possible, then as far left/back as possible
    score = z * 1000000 + y * 1000 + x
    return score

def load(boxes, container):
    W, H, D = container
    container_volume = W * H * D
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute rotations for each box
    box_rotations = [_rotations(w, h, d) for w, h, d in boxes]
    
    # Sort boxes by volume descending, then by max dimension
    box_indices = list(range(len(boxes)))
    box_indices.sort(key=lambda i: (-boxes[i][0] * boxes[i][1] * boxes[i][2], 
                                      -max(boxes[i])))
    
    best_placements = None
    best_volume = 0
    
    # Try multiple random orderings and greedy placements
    max_iterations = 200
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Create a shuffled order with bias towards larger boxes first
        if iteration == 0:
            order = box_indices[:]
        else:
            # Mix of sorted and random
            if iteration % 3 == 0:
                order = box_indices[:]
            elif iteration % 3 == 1:
                order = box_indices[:]
                random.shuffle(order)
            else:
                # Partially shuffled: keep top half sorted, shuffle bottom half
                order = box_indices[:]
                mid = len(order) // 2
                random.shuffle(order[mid:])
        
        placed = []
        placements = [None] * len(boxes)
        current_volume = 0
        
        for idx in order:
            if time.time() > deadline:
                break
            
            w, h, d = boxes[idx]
            rotations = box_rotations[idx]
            
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            # Get candidate positions
            candidates = _face_points(placed, W, H, D)
            
            for (rw, rh, rd) in rotations:
                for (cx, cy, cz) in candidates:
                    if _fits_at(cx, cy, cz, rw, rh, rd, placed, W, H, D):
                        score = _score_position(cx, cy, cz, rw, rh, rd, placed, W, H, D)
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_rot
                placed.append((bx, by, bz, bw, bh, bd))
                placements[idx] = (0, bx, by, bz, bw, bh, bd)
                current_volume += bw * bh * bd
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
        
        # Check if we have a complete solution
        if all(p is not None for p in placements):
            if current_volume > best_volume:
                best_volume = current_volume
                best_placements = placements[:]
        
        # Early termination if we've filled the container completely
        if current_volume >= container_volume:
            break
    
    # If no complete solution found, use the best partial or a simple fallback
    if best_placements is None:
        # Fallback: simple row-based packing
        placements = []
        x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
        for i in range(len(boxes)):
            w, h, d = boxes[i]
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + d > D:
                placements.append((-1, 0, 0, 0, w, h, d))
                continue
            placements.append((0, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return placements
    
    return best_placements
# EVOLVE-BLOCK-END
