"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second margin from 120s limit
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []
    
    # Precompute all valid orientations for each box
    # An orientation is (w, h, d) where w,h,d are a permutation of the box dims
    # We only need orientations that fit in the container
    box_orientations = []
    for i, (bw, bh, bd) in enumerate(boxes):
        orients = set()
        for perm in itertools.permutations((bw, bh, bd)):
            if perm[0] <= W and perm[1] <= H and perm[2] <= D:
                orients.add(perm)
        box_orientations.append(list(orients))
    
    # Sort boxes by volume descending (Best Fit Decreasing heuristic)
    box_indices = list(range(n))
    box_indices.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    
    def check_overlap(placed, x, y, z, w, h, d):
        """Check if a box at (x,y,z) with dims (w,h,d) overlaps with any placed box."""
        for (px, py, pz, pw, ph, pd) in placed:
            # Check overlap on all three axes
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return True
        return False
    
    def find_positions(placed, w, h, d):
        """Find candidate positions for a box with dims (w,h,d).
        Returns list of (x, y, z) positions where the box could fit."""
        candidates = set()
        
        # Candidate 1: origin
        candidates.add((0, 0, 0))
        
        # Candidate 2: positions derived from existing boxes
        # For each placed box, try placing adjacent to it
        for (px, py, pz, pw, ph, pd) in placed:
            # Try placing to the right (x + pw)
            nx = px + pw
            if nx + w <= W:
                # Try various y, z positions
                for cy in [0, py, py + ph]:
                    for cz in [0, pz, pz + pd]:
                        if cy + h <= H and cz + d <= D:
                            candidates.add((nx, cy, cz))
            
            # Try placing above (y + ph)
            ny = py + ph
            if ny + h <= H:
                for cx in [0, px, px + pw]:
                    for cz in [0, pz, pz + pd]:
                        if cx + w <= W and cz + d <= D:
                            candidates.add((cx, ny, cz))
            
            # Try placing in front (z + pd)
            nz = pz + pd
            if nz + d <= D:
                for cx in [0, px, px + pw]:
                    for cy in [0, py, py + ph]:
                        if cx + w <= W and cy + h <= H:
                            candidates.add((cx, cy, nz))
        
        # Filter candidates: must fit in container and not overlap
        valid = []
        for (cx, cy, cz) in candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            if not check_overlap(placed, cx, cy, cz, w, h, d):
                valid.append((cx, cy, cz))
        
        return valid
    
    def try_place(boxes_sorted, container_dims, seed=0):
        """Try to place boxes using the given sorted order and seed for tie-breaking."""
        W2, H2, D2 = container_dims
        rng = random.Random(seed)
        placed = []  # list of (x, y, z, w, h, d, box_index)
        result = [-1] * len(boxes_sorted)
        
        for idx, box_idx in enumerate(boxes_sorted):
            if time.time() > deadline:
                break
            
            bw, bh, bd = boxes[box_idx]
            orients = box_orientations[box_idx]
            
            if not orients:
                continue
            
            # Shuffle orientations for variety
            orients_shuffled = orients[:]
            rng.shuffle(orients_shuffled)
            
            best_pos = None
            best_orient = None
            best_score = None
            
            for (ow, oh, od) in orients_shuffled:
                positions = find_positions(placed, ow, oh, od)
                if not positions:
                    continue
                
                # Score each position: prefer lower y, then lower z, then lower x
                # (pack towards bottom, back, left)
                for (px, py, pz) in positions:
                    # Simple scoring: minimize y, then z, then x
                    score = (py, pz, px)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (px, py, pz)
                        best_orient = (ow, oh, od)
            
            if best_pos is not None:
                px, py, pz = best_pos
                ow, oh, od = best_orient
                placed.append((px, py, pz, ow, oh, od, box_idx))
                result[box_idx] = (0, px, py, pz, ow, oh, od)
        
        return result
    
    # First, do a quick greedy pass to get a baseline
    best_result = try_place(box_indices, (W, H, D), seed=42)
    best_volume = sum(
        boxes[i][0] * boxes[i][1] * boxes[i][2] 
        for i in range(n) 
        if best_result[i][0] == 0
    )
    
    # Try multiple seeds to find a better packing
    seed = 0
    while time.time() < deadline:
        seed += 1
        result = try_place(box_indices, (W, H, D), seed=seed)
        volume = sum(
            boxes[i][0] * boxes[i][1] * boxes[i][2] 
            for i in range(n) 
            if result[i][0] == 0
        )
        if volume > best_volume:
            best_volume = volume
            best_result = result
            # If we've filled a lot, we can be more patient
            if volume > 0.8 * W * H * D:
                # Continue searching a bit more
                pass
            elif volume > 0.95 * W * H * D:
                # Very good, break early
                break
    
    # Final pass: try to fit remaining boxes with different strategies
    if time.time() < deadline - 5:
        # Identify boxes that were left out
        left_out = [i for i in range(n) if best_result[i][0] == -1]
        
        if left_out and time.time() < deadline - 2:
            # Try to re-pack with different sorting
            for sort_key in [
                lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2],  # volume
                lambda i: boxes[i][0] + boxes[i][1] + boxes[i][2],  # sum of dims
                lambda i: max(boxes[i][0], boxes[i][1], boxes[i][2]),  # max dim
            ]:
                if time.time() > deadline - 1:
                    break
                
                sorted_indices = list(range(n))
                sorted_indices.sort(key=sort_key, reverse=True)
                
                result = try_place(sorted_indices, (W, H, D), seed=seed + 1000)
                volume = sum(
                    boxes[i][0] * boxes[i][1] * boxes[i][2] 
                    for i in range(n) 
                    if result[i][0] == 0
                )
                if volume > best_volume:
                    best_volume = volume
                    best_result = result
    
    # Convert result to the required format
    placements = []
    for i in range(n):
        if best_result[i][0] == 0:
            placements.append(best_result[i])
        else:
            w, h, d = boxes[i]
            placements.append((-1, 0, 0, 0, w, h, d))
    
    return placements

# EVOLVE-BLOCK-END
