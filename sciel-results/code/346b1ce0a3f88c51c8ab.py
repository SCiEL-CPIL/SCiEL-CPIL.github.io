"""3D Container Loading with Extreme Point Representation and multi-strategy search."""
import time
import itertools
import random

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    deadline = time.time() + 115.0  # 5s safety margin before 120s kill
    
    # Precompute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((w, h, d)):
            opts.add(perm)
        return list(opts)
    
    box_orientations = [get_orientations(w, h, d) for w, h, d in boxes]
    box_volumes = [w * h * d for w, h, d in boxes]
    
    def is_overlap(x, y, z, w, h, d, placed):
        """Check if a box at (x,y,z) with size (w,h,d) overlaps with any placed box."""
        for px, py, pz, pw, ph, pd in placed:
            if x < px + pw and x + w > px and \
               y < py + ph and y + h > py and \
               z < pz + pd and z + d > pz:
                return True
        return False

    def pack_with_grid(ordered_indices):
        """
        Pack boxes using a list-based overlap check.
        Returns (placements_list, total_volume)
        """
        placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        placed_boxes = []  # List of (x, y, z, w, h, d)
        total_volume = 0
        
        # Candidate positions: use box boundaries as candidate x, y, z positions
        candidate_x = set([0])
        candidate_y = set([0])
        candidate_z = set([0])
        
        for idx in ordered_indices:
            if time.time() > deadline:
                break
            
            w, h, d = boxes[idx]
            orients = box_orientations[idx]
            
            best_pos = None
            best_score = float('inf')
            
            # Try all orientations
            for ow, oh, od in orients:
                # Generate candidate positions from existing box boundaries
                cx_candidates = list(candidate_x)
                cy_candidates = list(candidate_y)
                cz_candidates = list(candidate_z)
                
                for cx in cx_candidates:
                    if cx + ow > W:
                        continue
                    for cy in cy_candidates:
                        if cy + oh > H:
                            continue
                        # Find lowest valid z
                        # We can iterate through candidate_z or binary search if we had a grid.
                        # Since we don't have a grid, we just try candidate_z + 0? 
                        # Actually, candidate_z might not contain the exact lowest z.
                        # A simpler approach: try all candidate_z values and pick the lowest valid one?
                        # Or just try 0 first? No, 0 might be occupied.
                        # Let's try all candidate_z values.
                        best_z_for_xy = None
                        for cz in cz_candidates:
                            if cz + od > D:
                                continue
                            if not is_overlap(cx, cy, cz, ow, oh, od, placed_boxes):
                                if best_z_for_xy is None or cz < best_z_for_xy:
                                    best_z_for_xy = cz
                        
                        if best_z_for_xy is not None:
                            # Score: Prefer lower Y (stacking), then lower Z, then lower X
                            score = cy * 1000000 + best_z_for_xy * 1000 + cx
                            if score < best_score:
                                best_score = score
                                best_pos = (cx, cy, best_z_for_xy, ow, oh, od)
            
            if best_pos:
                ex, ey, ez, ow, oh, od = best_pos
                placements[idx] = (0, ex, ey, ez, ow, oh, od)
                placed_boxes.append((ex, ey, ez, ow, oh, od))
                total_volume += ow * oh * od
                
                # Update candidate positions with new box boundaries
                candidate_x.add(ex + ow)
                candidate_y.add(ey + oh)
                candidate_z.add(ez + od)
        
        return placements, total_volume

    # Strategies for sorting
    indexed_boxes = list(range(n))
    
    strategies = [
        lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2],  # Volume
        lambda i: max(boxes[i][0], boxes[i][1], boxes[i][2]),  # Max dim
        lambda i: boxes[i][0] + boxes[i][1] + boxes[i][2],  # Sum of dims
        lambda i: boxes[i][1],  # Height
        lambda i: boxes[i][0],  # Width
        lambda i: boxes[i][2],  # Depth
        lambda i: -min(boxes[i][0], boxes[i][1], boxes[i][2]),  # Min dim (ascending)
        lambda i: boxes[i][0] * boxes[i][1],  # Base area
        lambda i: boxes[i][1] * boxes[i][2],  # Side area
        lambda i: boxes[i][0] * boxes[i][2],  # Top area
    ]
    
    best_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    best_volume = 0
    
    # Run each static sorting strategy
    for sort_key in strategies:
        if time.time() > deadline - 10:
            break
        ordered = sorted(indexed_boxes, key=sort_key, reverse=True)
        p, v = pack_with_grid(ordered)
        if v > best_volume:
            best_placements = p
            best_volume = v

    # If we have time, try some random restarts
    if time.time() < deadline - 5:
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            random.shuffle(indexed_boxes)
            ordered = indexed_boxes[:]
            p, v = pack_with_grid(ordered)
            if v > best_volume:
                best_placements = p
                best_volume = v

    return best_placements
# EVOLVE-BLOCK-END
