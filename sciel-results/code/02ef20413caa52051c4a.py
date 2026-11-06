"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    deadline = time.time() + 110.0  # 10s safety margin
    n = len(boxes)
    
    # Initialize placements with all boxes left out
    # Format: (bin, x, y, z, w, h, d)
    # bin=0 for loaded, bin=-1 for left out
    initial_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    best_placements = initial_placements
    best_volume = 0
    
    def get_orientations(w, h, d):
        """Return unique orientations of a box (w, h, d)."""
        orients = set()
        for perm in itertools.permutations([w, h, d]):
            orients.add(perm)
        return list(orients)
    
    def blf_pack(ordered_indices):
        """
        Bottom-Left-Front packing algorithm.
        For each box, try to place it at the lowest, most-left, most-front position
        that doesn't overlap with already placed boxes.
        """
        placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        total_volume = 0
        placed_boxes = []  # List of (x, y, z, w, h, d) for placed boxes
        
        def can_place(x, y, z, w, h, d):
            """Check if a box at (x,y,z) with dims (w,h,d) fits in container and doesn't overlap."""
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            # Check overlap with all placed boxes
            for px, py, pz, pw, ph, pd in placed_boxes:
                if not (x + w <= px or px + pw <= x or
                        y + h <= py or py + ph <= y or
                        z + d <= pz or pz + pd <= z):
                    return False
            return True
        
        for idx in ordered_indices:
            if time.time() > deadline:
                break
            
            w, h, d = boxes[idx]
            orients = get_orientations(w, h, d)
            
            best_pos = None
            best_score = None
            
            for ow, oh, od in orients:
                # Try to find the best position (lowest y, then leftmost x, then frontmost z)
                # We'll scan through possible positions
                found = False
                for y in range(0, H - oh + 1):
                    if found:
                        break
                    for x in range(0, W - ow + 1):
                        if found:
                            break
                        for z in range(0, D - od + 1):
                            if can_place(x, y, z, ow, oh, od):
                                score = (y, x, z)
                                if best_score is None or score < best_score:
                                    best_score = score
                                    best_pos = (x, y, z, ow, oh, od)
                                found = True
                                break
                
                if found:
                    break
            
            if best_pos is not None:
                x, y, z, ow, oh, od = best_pos
                placements[idx] = (0, x, y, z, ow, oh, od)
                placed_boxes.append((x, y, z, ow, oh, od))
                total_volume += ow * oh * od
        
        return placements, total_volume
    
    # Generate different sorting strategies
    indexed_boxes = list(range(n))
    
    strategies = [
        ("volume_desc", lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], True),
        ("max_dim_desc", lambda i: max(boxes[i][0], boxes[i][1], boxes[i][2]), True),
        ("sum_desc", lambda i: boxes[i][0] + boxes[i][1] + boxes[i][2], True),
        ("height_desc", lambda i: boxes[i][1], True),
        ("width_desc", lambda i: boxes[i][0], True),
        ("depth_desc", lambda i: boxes[i][2], True),
        ("volume_asc", lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], False),
        ("max_dim_asc", lambda i: max(boxes[i][0], boxes[i][1], boxes[i][2]), False),
    ]
    
    # Run BLF for all strategies
    for name, key_func, reverse in strategies:
        if time.time() > deadline - 5:
            break
        
        ordered = sorted(indexed_boxes, key=key_func, reverse=reverse)
        p, v = blf_pack(ordered)
        
        if v > best_volume:
            best_volume = v
            best_placements = p
    
    return best_placements
# EVOLVE-BLOCK-END
