"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from typing import List, Tuple

def pack(boxes, container):
    start_time = time.time()
    DEADLINE = 110.0  # safety margin before 120s limit
    
    W, H, D = container
    container_vol = W * H * D
    n = len(boxes)
    
    if n == 0:
        return []
    
    def try_place_box_in_bin(box, bin_boxes, W, H, D):
        """Try to place a box in a bin. Returns (x,y,z) if successful, None otherwise."""
        w, h, d = box
        
        # Generate candidate positions: bin origin and corners/edges of existing boxes
        candidates = set()
        candidates.add((0, 0, 0))
        
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            # Adjacent positions along each axis
            if bx + bw <= W:
                candidates.add((bx + bw, by, bz))
            if by + bh <= H:
                candidates.add((bx, by + bh, bz))
            if bz + bd <= D:
                candidates.add((bx, by, bz + bd))
            # Corner positions (2D and 3D)
            if bx + bw <= W and by + bh <= H:
                candidates.add((bx + bw, by + bh, bz))
            if bx + bw <= W and bz + bd <= D:
                candidates.add((bx + bw, by, bz + bd))
            if by + bh <= H and bz + bd <= D:
                candidates.add((bx, by + bh, bz + bd))
            if bx + bw <= W and by + bh <= H and bz + bd <= D:
                candidates.add((bx + bw, by + bh, bz + bd))
        
        best_pos = None
        best_score = None
        
        for (cx, cy, cz) in candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            
            # Check overlap
            overlap = False
            for (bx, by, bz, bw, bh, bd) in bin_boxes:
                if not (cx + w <= bx or bx + bw <= cx or
                        cy + h <= by or by + bh <= cy or
                        cz + d <= bz or bz + bd <= cz):
                    overlap = True
                    break
            
            if not overlap:
                # Score: prefer compact placement (lower coordinates)
                score = cx * 10000 + cy * 100 + cz
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
        
        return best_pos
    
    def pack_with_ordering(order):
        """Pack boxes in the given order. Returns (placements, num_bins) or None."""
        bins = []  # list of lists of (x,y,z,w,h,d)
        placements = []  # list of (bin_idx, x, y, z, w, h, d)
        
        for idx in order:
            w, h, d = boxes[idx]
            placed = False
            
            for bin_idx, bin_boxes in enumerate(bins):
                pos = try_place_box_in_bin((w, h, d), bin_boxes, W, H, D)
                if pos is not None:
                    x, y, z = pos
                    bin_boxes.append((x, y, z, w, h, d))
                    placements.append((bin_idx, x, y, z, w, h, d))
                    placed = True
                    break
            
            if not placed:
                # Open new bin
                bin_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placements.append((bin_idx, 0, 0, 0, w, h, d))
        
        return placements, len(bins)
    
    def compute_score(placements, num_bins):
        """Compute the score for a given placement."""
        if num_bins == 0:
            return 0.0
        
        last_bin_idx = max(p[0] for p in placements)
        
        last_bin_vol = 0
        for (b, x, y, z, w, h, d) in placements:
            if b == last_bin_idx:
                last_bin_vol += w * h * d
        
        last_frac = last_bin_vol / container_vol if container_vol > 0 else 0
        return num_bins + 0.1 * (1 - last_frac)
    
    # Generate different orderings
    orderings = []
    
    # 1. By volume descending
    orderings.append(sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True))
    
    # 2. By max dimension descending
    orderings.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))
    
    # 3. Original order
    orderings.append(list(range(n)))
    
    # 4. By w descending, then h, then d
    orderings.append(sorted(range(n), key=lambda i: (boxes[i][0], boxes[i][1], boxes[i][2]), reverse=True))
    
    # 5. By d descending, then h, then w
    orderings.append(sorted(range(n), key=lambda i: (boxes[i][2], boxes[i][1], boxes[i][0]), reverse=True))
    
    # 6. By h descending, then w, then d
    orderings.append(sorted(range(n), key=lambda i: (boxes[i][1], boxes[i][0], boxes[i][2]), reverse=True))
    
    # 7. By volume ascending (smaller first)
    orderings.append(sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2]))
    
    best_placements = None
    best_score = float('inf')
    
    for order in orderings:
        if time.time() - start_time > DEADLINE:
            break
        
        placements, num_bins = pack_with_ordering(order)
        score = compute_score(placements, num_bins)
        
        if score < best_score:
            best_score = score
            best_placements = placements
    
    if best_placements is None:
        # Fallback: simple sequential packing
        placements = []
        bin_idx = 0
        x, y, z = 0, 0, 0
        row_h, layer_d = 0, 0
        for i, (w, h, d) in enumerate(boxes):
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                bin_idx += 1
                x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
            placements.append((bin_idx, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        best_placements = placements
    
    return best_placements
# EVOLVE-BLOCK-END
