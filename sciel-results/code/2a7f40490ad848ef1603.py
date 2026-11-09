"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import permutations

# EVOLVE-BLOCK-START

def _unique_rotations(w, h, d):
    """Return unique rotations of a box as (w, h, d) tuples."""
    seen = set()
    result = []
    for perm in permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result


def _can_place(x, y, z, bw, bh, bd, placed, W, H, D):
    """Check if a box of size (bw,bh,bd) fits at (x,y,z) without overlap."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + bw > W or y + bh > H or z + bd > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        # Check separation in at least one axis
        if not (x + bw <= px or px + pw <= x):
            if not (y + bh <= py or py + ph <= y):
                if not (z + bd <= pz or pz + pd <= z):
                    return False
    return True


def _get_candidate_positions(placed, W, H, D):
    """Generate candidate bottom-left-front corners from existing placements."""
    candidates = set()
    candidates.add((0, 0, 0))
    for (px, py, pz, pw, ph, pd) in placed:
        # Positions just beyond each face
        candidates.add((px + pw, py, pz))
        candidates.add((px, py + ph, pz))
        candidates.add((px, py, pz + pd))
        # Also try combinations of offsets
        for (px2, py2, pz2, pw2, ph2, pd2) in placed:
            if px2 >= px:
                candidates.add((px2, py, pz))
            if py2 >= py:
                candidates.add((px, py2, pz))
            if pz2 >= pz:
                candidates.add((px, py, pz2))
    # Filter valid positions
    valid = []
    for (x, y, z) in candidates:
        if 0 <= x < W and 0 <= y < H and 0 <= z < D:
            valid.append((x, y, z))
    return valid


def _greedy_place(boxes, container, deadline=None, rng=None):
    """Greedy placement with rotation and position search."""
    if rng is None:
        rng = random.Random(42)
    
    W, H, D = container
    container_vol = W * H * D
    
    # Sort boxes by volume descending for better packing
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: -x[1][0] * x[1][1] * x[1][2])
    
    placed = []  # (x, y, z, w, h, d)
    results = [(-1, 0, 0, 0, 0, 0, 0)] * len(boxes)  # (bin, x, y, z, w, h, d)
    
    start_time = time.time() if deadline else 0
    
    for idx, (orig_idx, (w, h, d)) in enumerate(indexed_boxes):
        if deadline and (time.time() - start_time) > deadline:
            break
        
        rotations = _unique_rotations(w, h, d)
        best_vol = 0
        best_placement = None
        
        # Try each rotation
        for (rw, rh, rd) in rotations:
            # Get candidate positions
            if not placed:
                candidates = [(0, 0, 0)]
            else:
                candidates = _get_candidate_positions(placed, W, H, D)
            
            # Also add some random positions for diversity
            if len(candidates) < 50 and placed:
                for _ in range(10):
                    rx = rng.randint(0, max(0, W - rw))
                    ry = rng.randint(0, max(0, H - rh))
                    rz = rng.randint(0, max(0, D - rd))
                    candidates.append((rx, ry, rz))
            
            for (cx, cy, cz) in candidates:
                if _can_place(cx, cy, cz, rw, rh, rd, placed, W, H, D):
                    vol = rw * rh * rd
                    if vol > best_vol:
                        best_vol = vol
                        best_placement = (cx, cy, cz, rw, rh, rd)
            
            if best_placement:
                break
        
        if best_placement:
            cx, cy, cz, rw, rh, rd = best_placement
            placed.append((cx, cy, cz, rw, rh, rd))
            results[orig_idx] = (0, cx, cy, cz, rw, rh, rd)
    
    return results


def _improve_placement(results, boxes, container, deadline=None, rng=None):
    """Try to improve by moving some boxes."""
    if rng is None:
        rng = random.Random(42)
    
    W, H, D = container
    placed = []
    for i, (bin_id, x, y, z, w, h, d) in enumerate(results):
        if bin_id == 0:
            placed.append((i, x, y, z, w, h, d))
    
    if not placed:
        return results
    
    # Try to remove and re-place some boxes
    improved = False
    for _ in range(50):
        if deadline and time.time() > deadline:
            break
        
        # Pick a random placed box to potentially move
        if not placed:
            break
        idx = rng.randint(0, len(placed) - 1)
        box_idx, bx, by, bz, bw, bh, bd = placed[idx]
        
        # Try removing it and see if we can place it better
        placed.remove(placed[idx])
        
        # Rebuild placed list without this box
        new_placed = [(x, y, z, w, h, d) for (_, x, y, z, w, h, d) in placed]
        
        rotations = _unique_rotations(bw, bh, bd)
        best_placement = None
        best_score = -1
        
        for (rw, rh, rd) in rotations:
            if not new_placed:
                candidates = [(0, 0, 0)]
            else:
                candidates = _get_candidate_positions(new_placed, W, H, D)
            
            for (cx, cy, cz) in candidates:
                if _can_place(cx, cy, cz, rw, rh, rd, new_placed, W, H, D):
                    # Score: prefer lower positions
                    score = -(cx * 0.1 + cy * 0.1 + cz * 0.2)
                    if score > best_score:
                        best_score = score
                        best_placement = (cx, cy, cz, rw, rh, rd)
        
        if best_placement:
            cx, cy, cz, rw, rh, rd = best_placement
            placed.append((box_idx, cx, cy, cz, rw, rh, rd))
            results[box_idx] = (0, cx, cy, cz, rw, rh, rd)
            improved = True
        else:
            # Put it back
            placed.append((box_idx, bx, by, bz, bw, bh, bd))
    
    return results


def load(boxes, container):
    W, H, D = container
    container_vol = W * H * D
    
    # Deadline guard: use 110 seconds out of 120
    deadline = 110.0
    
    # Try multiple strategies and keep the best
    best_results = None
    best_score = -1
    
    rng = random.Random(42)
    
    # Strategy 1: Simple greedy with sorting
    results1 = _greedy_place(boxes, container, deadline=deadline * 0.3, rng=rng)
    score1 = sum(w * h * d for (b, x, y, z, w, h, d) in results1 if b == 0) / container_vol
    best_results = results1
    best_score = score1
    
    # Strategy 2: Greedy with different sort order
    rng2 = random.Random(123)
    results2 = _greedy_place(boxes, container, deadline=deadline * 0.3, rng=rng2)
    score2 = sum(w * h * d for (b, x, y, z, w, h, d) in results2 if b == 0) / container_vol
    if score2 > best_score:
        best_results = results2
        best_score = score2
    
    # Strategy 3: Improve the best result
    remaining_time = deadline * 0.4
    if remaining_time > 5:
        improved = _improve_placement(best_results[:], boxes, container, 
                                       deadline=time.time() + remaining_time, rng=rng)
        score3 = sum(w * h * d for (b, x, y, z, w, h, d) in improved if b == 0) / container_vol
        if score3 > best_score:
            best_results = improved
            best_score = score3
    
    return best_results

# EVOLVE-BLOCK-END
