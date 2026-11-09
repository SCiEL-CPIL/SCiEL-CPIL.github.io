"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import permutations
from collections import defaultdict

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
    """Generate candidate positions from face points of placed boxes."""
    points = set()
    # Origin
    points.add((0, 0, 0))
    for (px, py, pz, pw, ph, pd) in placed:
        # Faces extending from existing boxes
        points.add((px + pw, py, pz))
        points.add((px, py + ph, pz))
        points.add((px, py, pz + pd))
        points.add((px + pw, py + ph, pz))
        points.add((px + pw, py, pz + pd))
        points.add((px, py + ph, pz + pd))
        points.add((px + pw, py + ph, pz + pd))
    # Filter out-of-bounds
    valid_points = []
    for (x, y, z) in points:
        if 0 <= x < W and 0 <= y < H and 0 <= z < D:
            valid_points.append((x, y, z))
    return valid_points

def _score_position(x, y, z, bw, bh, bd, W, H, D):
    """Score a position: prefer lower z, then lower y, then lower x (gravity + packing)."""
    # Lower positions are better (gravity)
    return (z, y, x)

def load(boxes, container):
    W, H, D = container
    container_volume = W * H * D
    
    # Precompute rotations for each box
    box_rots = [_rotations(w, h, d) for (w, h, d) in boxes]
    
    # Sort boxes by volume descending, then by max dimension descending
    box_indices = list(range(len(boxes)))
    box_indices.sort(key=lambda i: (-(boxes[i][0] * boxes[i][1] * boxes[i][2]),
                                     -max(boxes[i])))
    
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    best_volume = 0
    best_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(len(boxes))]
    
    # Try multiple random orderings and greedy placements
    n_boxes = len(boxes)
    
    # First pass: deterministic greedy by volume descending
    placed = []  # list of (x, y, z, w, h, d)
    placement_map = {}  # box_index -> (x, y, z, w, h, d)
    
    for bi in box_indices:
        w, h, d = boxes[bi]
        best_pos = None
        best_rot = None
        best_score = None
        
        for (bw, bh, bd) in box_rots[bi]:
            # Try face points
            candidate_points = _face_points(placed, W, H, D)
            for (x, y, z) in candidate_points:
                if _fits_at(x, y, z, bw, bh, bd, placed, W, H, D):
                    score = _score_position(x, y, z, bw, bh, bd, W, H, D)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (x, y, z)
                        best_rot = (bw, bh, bd)
        
        if best_pos is not None:
            placed.append((best_pos[0], best_pos[1], best_pos[2], best_rot[0], best_rot[1], best_rot[2]))
            placement_map[bi] = (best_pos[0], best_pos[1], best_pos[2], best_rot[0], best_rot[1], best_rot[2])
    
    # Compute volume
    vol = sum(p[3] * p[4] * p[5] for p in placed)
    if vol > best_volume:
        best_volume = vol
        best_placements = []
        for i in range(n_boxes):
            if i in placement_map:
                x, y, z, bw, bh, bd = placement_map[i]
                best_placements.append((0, x, y, z, bw, bh, bd))
            else:
                w, h, d = boxes[i]
                best_placements.append((-1, 0, 0, 0, w, h, d))
    
    # Try several randomized orderings if time permits
    rng = random.Random(42)
    n_trials = 0
    while time.time() < deadline - 5:
        n_trials += 1
        # Random shuffle with bias toward larger boxes
        trial_order = list(box_indices)
        rng.shuffle(trial_order)
        
        placed = []
        placement_map = {}
        
        for bi in trial_order:
            if time.time() > deadline - 2:
                break
            w, h, d = boxes[bi]
            best_pos = None
            best_rot = None
            best_score = None
            
            for (bw, bh, bd) in box_rots[bi]:
                candidate_points = _face_points(placed, W, H, D)
                for (x, y, z) in candidate_points:
                    if _fits_at(x, y, z, bw, bh, bd, placed, W, H, D):
                        score = _score_position(x, y, z, bw, bh, bd, W, H, D)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
                            best_rot = (bw, bh, bd)
            
            if best_pos is not None:
                placed.append((best_pos[0], best_pos[1], best_pos[2], best_rot[0], best_rot[1], best_rot[2]))
                placement_map[bi] = (best_pos[0], best_pos[1], best_pos[2], best_rot[0], best_rot[1], best_rot[2])
        
        vol = sum(p[3] * p[4] * p[5] for p in placed)
        if vol > best_volume:
            best_volume = vol
            best_placements = []
            for i in range(n_boxes):
                if i in placement_map:
                    x, y, z, bw, bh, bd = placement_map[i]
                    best_placements.append((0, x, y, z, bw, bh, bd))
                else:
                    w, h, d = boxes[i]
                    best_placements.append((-1, 0, 0, 0, w, h, d))
        
        # If we've found a very good solution, we can stop early
        if best_volume / container_volume > 0.95:
            break
    
    return best_placements
# EVOLVE-BLOCK-END
