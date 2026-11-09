"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    
    # Precompute rotations for each box (unique permutations)
    def get_rotations(w, h, d):
        return list(set(permutations((w, h, d))))
    
    rotations = [get_rotations(w, h, d) for w, h, d in boxes]
    
    # Deadline guard
    deadline = time.time() + 110  # 110s to be safe before 120s limit
    
    def try_place_box(placed, box_idx, rot_idx, x, y, z):
        """Check if box can be placed at (x,y,z) with given rotation."""
        rw, rh, rd = rotations[box_idx][rot_idx]
        # Check bounds
        if x + rw > W or y + rh > H or z + rd > D:
            return False
        # Check overlap with all placed boxes
        for (px, py, pz, pw, ph, pd) in placed:
            # Check if boxes overlap (they don't overlap if separated on any axis)
            if not (x + rw <= px or px + pw <= x or
                    y + rh <= py or py + ph <= y or
                    z + rd <= pz or pz + pd <= z):
                return False
        return True
    
    def place_box(placed, box_idx, rot_idx, x, y, z):
        rw, rh, rd = rotations[box_idx][rot_idx]
        placed.append((x, y, z, rw, rh, rd))
        return rw, rh, rd
    
    def compute_score(placed):
        """Compute volume filled."""
        vol = 0
        for (px, py, pz, pw, ph, pd) in placed:
            vol += pw * ph * pd
        return vol / container_vol if container_vol > 0 else 0
    
    def greedy_pack(order, placed):
        """Place boxes in given order using greedy bottom-left-front strategy."""
        placed = list(placed)
        for box_idx in order:
            best_pos = None
            best_rot = None
            for rot_idx, (rw, rh, rd) in enumerate(rotations[box_idx]):
                # Try to find the best (most bottom-left-front) position
                # Scan from origin
                best_x, best_y, best_z = None, None, None
                # Try all possible positions - this is expensive, so use heuristic
                # Place at the "most bottom-left-front" position where it fits
                
                # Heuristic: try to place as close to origin as possible
                # First try (0,0,0), then scan
                found = False
                for z in range(0, D - rd + 1, max(1, rd // 10)):
                    for y in range(0, H - rh + 1, max(1, rh // 10)):
                        for x in range(0, W - rw + 1, max(1, rw // 10)):
                            if try_place_box(placed, box_idx, rot_idx, x, y, z):
                                if best_pos is None:
                                    best_pos = (x, y, z)
                                    best_rot = rot_idx
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            
            if best_pos is not None:
                place_box(placed, box_idx, best_rot, best_pos[0], best_pos[1], best_pos[2])
        return placed
    
    def greedy_pack_simple(order, placed):
        """Simpler greedy: place each box at the first valid position from bottom-left-front."""
        placed = list(placed)
        for box_idx in order:
            best_pos = None
            best_rot = None
            for rot_idx, (rw, rh, rd) in enumerate(rotations[box_idx]):
                # Try to place at the lowest, leftmost, frontmost position
                # Use a coarse scan then refine
                placed_here = False
                # Try different z levels
                for z in sorted(range(0, D - rd + 1, max(1, min(rd, D // 20)))):
                    for y in sorted(range(0, H - rh + 1, max(1, min(rh, H // 20)))):
                        for x in sorted(range(0, W - rw + 1, max(1, min(rw, W // 20)))):
                            if try_place_box(placed, box_idx, rot_idx, x, y, z):
                                best_pos = (x, y, z)
                                best_rot = rot_idx
                                placed_here = True
                                break
                        if placed_here:
                            break
                    if placed_here:
                        break
                if placed_here:
                    break
            
            if best_pos is not None:
                place_box(placed, box_idx, best_rot, best_pos[0], best_pos[1], best_pos[2])
        return placed
    
    def sort_by_volume_desc(boxes):
        return sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    def sort_by_max_dim_desc(boxes):
        return sorted(range(len(boxes)), key=lambda i: max(boxes[i]), reverse=True)
    
    def sort_by_min_dim_desc(boxes):
        return sorted(range(len(boxes)), key=lambda i: min(boxes[i]), reverse=True)
    
    def sort_by_compactness(boxes):
        # Compactness: how close to a cube (lower ratio of max/min = more compact)
        return sorted(range(len(boxes)), key=lambda i: max(boxes[i])/min(boxes[i]) if min(boxes[i]) > 0 else float('inf'))
    
    def random_order(boxes, rng):
        order = list(range(len(boxes)))
        rng.shuffle(order)
        return order
    
    # Strategy 1: Volume descending
    order1 = sort_by_volume_desc(boxes)
    # Strategy 2: Max dimension descending
    order2 = sort_by_max_dim_desc(boxes)
    # Strategy 3: Min dimension descending
    order3 = sort_by_min_dim_desc(boxes)
    # Strategy 4: Compactness
    order4 = sort_by_compactness(boxes)
    
    best_placed = []
    best_score = 0
    
    # Try deterministic strategies first
    strategies = [order1, order2, order3, order4]
    for order in strategies:
        if time.time() > deadline - 10:
            break
        placed = greedy_pack_simple(order, [])
        score = compute_score(placed)
        if score > best_score:
            best_score = score
            best_placed = placed
    
    # Random restarts
    rng = random.Random(42)
    n_restarts = 0
    while time.time() < deadline - 5:
        order = random_order(boxes, rng)
        # Sometimes bias towards larger boxes first
        if rng.random() < 0.3:
            # Mix: large boxes first, then random
            large = [i for i in range(n) if boxes[i][0]*boxes[i][1]*boxes[i][2] > sum(b[0]*b[1]*b[2] for b in boxes)/n]
            small = [i for i in range(n) if i not in large]
            rng.shuffle(small)
            order = large + small
        placed = greedy_pack_simple(order, [])
        score = compute_score(placed)
        if score > best_score:
            best_score = score
            best_placed = placed
        n_restarts += 1
        if n_restarts % 100 == 0 and time.time() > deadline - 20:
            break
    
    # Build result
    placed_set = set()
    for (px, py, pz, pw, ph, pd) in best_placed:
        placed_set.add((px, py, pz, pw, ph, pd))
    
    # Map placements back to box indices
    # We need to track which box went where
    # Re-run the best strategy to get the mapping
    # For now, just mark boxes that fit
    placements = []
    used_boxes = set()
    
    # Re-derive the placement mapping
    # This is tricky - we need to track box indices
    # Let's redo with tracking
    
    def greedy_pack_tracked(order, placed):
        """Place boxes in given order, tracking which box index was placed where."""
        placed = list(placed)
        box_map = {}  # (x,y,z,rw,rh,rd) -> box_idx
        for box_idx in order:
            best_pos = None
            best_rot = None
            for rot_idx, (rw, rh, rd) in enumerate(rotations[box_idx]):
                placed_here = False
                for z in sorted(range(0, D - rd + 1, max(1, min(rd, D // 20)))):
                    for y in sorted(range(0, H - rh + 1, max(1, min(rh, H // 20)))):
                        for x in sorted(range(0, W - rw + 1, max(1, min(rw, W // 20)))):
                            if try_place_box(placed, box_idx, rot_idx, x, y, z):
                                best_pos = (x, y, z)
                                best_rot = rot_idx
                                placed_here = True
                                break
                        if placed_here:
                            break
                    if placed_here:
                        break
                if placed_here:
                    break
            
            if best_pos is not None:
                place_box(placed, box_idx, best_rot, best_pos[0], best_pos[1], best_pos[2])
                box_map[(best_pos[0], best_pos[1], best_pos[2], rw, rh, rd)] = box_idx
        return placed, box_map
    
    # Re-run the best strategy with tracking
    # We don't know which strategy was best, so just use the first one that matches
    # Actually, let's just re-run all and pick the best again with tracking
    best_placed_tracked = []
    best_box_map = {}
    best_score2 = 0
    
    strategies = [order1, order2, order3, order4]
    for order in strategies:
        if time.time() > deadline - 5:
            break
        placed, box_map = greedy_pack_tracked(order, [])
        score = compute_score(placed)
        if score > best_score2:
            best_score2 = score
            best_placed_tracked = placed
            best_box_map = box_map
    
    # Use the tracked version
    if best_placed_tracked:
        placed = best_placed_tracked
        box_map = best_box_map
    else:
        placed = best_placed
        box_map = {}
    
    # Build final placements
    loaded_boxes = set(box_map.values())
    for i, (w, h, d) in enumerate(boxes):
        if i in loaded_boxes:
            # Find the placement
            for (px, py, pz, pw, ph, pd), bidx in box_map.items():
                if bidx == i:
                    placements.append((0, px, py, pz, pw, ph, pd))
                    break
            else:
                placements.append((-1, 0, 0, 0, w, h, d))
        else:
            placements.append((-1, 0, 0, 0, w, h, d))
    
    return placements
# EVOLVE-BLOCK-END
