"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import math

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []
    
    container_volume = W * H * D
    
    # Generate all 6 rotations for each box (unique ones)
    def get_rotations(w, h, d):
        rots = set()
        for perm in itertools.permutations((w, h, d)):
            rots.add(tuple(sorted(perm)))
        # Actually we want all 6 permutations, but unique dimensions matter for placement
        # Let's just use all 6 unique permutations
        rots = []
        seen = set()
        for perm in itertools.permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                rots.append(perm)
        return rots
    
    # Precompute rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        box_rotations.append(get_rotations(w, h, d))
    
    # Sort boxes by volume descending for better packing
    box_indices = list(range(n))
    box_volumes = [w * h * d for w, h, d in boxes]
    box_indices.sort(key=lambda i: -box_volumes[i])
    
    def can_place(placed, x, y, z, w, h, d):
        """Check if a box of size (w,h,d) can be placed at (x,y,z) without overlapping."""
        for (px, py, pz, pw, ph, pd) in placed:
            if not (x + w <= px or px + pw <= x):
                if not (y + h <= py or py + ph <= y):
                    if not (z + d <= pz or pz + pd <= z):
                        return False
        return True
    
    def find_position_for_box(placed, W, H, D, w, h, d):
        """Find the best position for a box using corner points and extremal points."""
        # Generate candidate positions from corners of existing boxes
        candidates = [(0, 0, 0)]
        for (px, py, pz, pw, ph, pd) in placed:
            # Corners that extend from existing boxes
            candidates.append((px + pw, py, pz))
            candidates.append((px, py + ph, pz))
            candidates.append((px, py, pz + pd))
            candidates.append((px + pw, py + ph, pz))
            candidates.append((px + pw, py, pz + pd))
            candidates.append((px, py + ph, pz + pd))
            candidates.append((px, py, pz + pd))
            candidates.append((px + pw, py + ph, pz + pd))
        
        # Also add some systematic positions
        # Try to pack along axes
        for x in [0]:
            for y in [0]:
                for z in [0]:
                    candidates.append((x, y, z))
        
        # Find the "lowest" valid position (minimize z, then y, then x for compactness)
        best_pos = None
        best_key = None
        for (cx, cy, cz) in candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            if can_place(placed, cx, cy, cz, w, h, d):
                # Prefer positions that are lower (smaller z), then more forward (smaller y), then left (smaller x)
                key = (cz, cy, cx)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pos = (cx, cy, cz)
        
        # If no candidate position works, try a grid search
        if best_pos is None:
            # Try a coarse grid
            step = max(1, min(w, h, d) // 2) if min(w, h, d) > 2 else 1
            found = False
            for cz in range(0, D - d + 1, step):
                if found:
                    break
                for cy in range(0, H - h + 1, step):
                    if found:
                        break
                    for cx in range(0, W - w + 1, step):
                        if can_place(placed, cx, cy, cz, w, h, d):
                            best_pos = (cx, cy, cz)
                            found = True
                            break
        
        return best_pos
    
    # Try multiple sorting strategies and keep the best result
    best_placements = None
    best_volume = 0
    
    sorting_strategies = [
        lambda i: -box_volumes[i],  # Volume descending
        lambda i: -max(boxes[i]),   # Max dimension descending
        lambda i: min(boxes[i]),    # Min dimension ascending
        lambda i: -sum(boxes[i]),   # Sum of dimensions descending
    ]
    
    for sort_key in sorting_strategies:
        if time.time() > deadline - 5:
            break
        
        sorted_indices = list(range(n))
        sorted_indices.sort(key=sort_key)
        
        placed = []  # List of (x, y, z, w, h, d)
        box_placement = [-1] * n  # Index into placed, or -1 if not placed
        
        for idx in sorted_indices:
            if time.time() > deadline - 5:
                break
            
            w, h, d = boxes[idx]
            rots = box_rotations[idx]
            
            best_rot = None
            best_pos = None
            best_key = None
            
            for (rw, rh, rd) in rots:
                pos = find_position_for_box(placed, W, H, D, rw, rh, rd)
                if pos is not None:
                    key = (pos[2], pos[1], pos[0])  # Prefer lower, more forward, more left
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = pos
                        best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                x, y, z = best_pos
                rw, rh, rd = best_rot
                placed.append((x, y, z, rw, rh, rd))
                box_placement[idx] = len(placed) - 1
        
        # Calculate filled volume
        filled_volume = sum(w * h * d for (_, _, _, w, h, d) in placed)
        
        if filled_volume > best_volume:
            best_volume = filled_volume
            # Build placements array
            tmp_placements = [(-1, 0, 0, 0, 0, 0, 0)] * n
            for idx in range(n):
                if box_placement[idx] != -1:
                    x, y, z, w, h, d = placed[box_placement[idx]]
                    tmp_placements[idx] = (0, x, y, z, w, h, d)
                else:
                    w, h, d = boxes[idx]
                    tmp_placements[idx] = (-1, 0, 0, 0, w, h, d)
            best_placements = tmp_placements
    
    if best_placements is None:
        # Fallback: all boxes left out
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placements
# EVOLVE-BLOCK-END
