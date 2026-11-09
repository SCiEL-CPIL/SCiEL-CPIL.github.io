"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import defaultdict

def pack(boxes, container):
    W, H, D = container
    container_vol = W * H * D
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard: return best found before ~115s (safety margin under 120s/150s kill)
    t_start = time.time()
    DEADLINE = 115.0
    
    # Precompute box volumes
    box_vols = [w * h * d for w, h, d in boxes]
    total_vol = sum(box_vols)
    
    # Helper: check if a box at (x,y,z) with dims (w,h,d) overlaps any placed box in bin
    def overlaps(x, y, z, w, h, d, placed):
        for (px, py, pz, pw, ph, pd) in placed:
            # Check overlap on all three axes
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return True
        return False
    
    # Helper: get candidate positions for a box in a bin (BLF - Bottom-Left-Farthest)
    def get_candidate_positions(bin_placed, w, h, d):
        """
        Generate candidate positions for placing a box of size (w,h,d) in a bin.
        Uses corners of existing boxes plus the origin.
        Returns list of (x, y, z) sorted by BLF preference (min z, then min y, then min x).
        """
        if not bin_placed:
            return [(0, 0, 0)]
        
        candidates = set()
        candidates.add((0, 0, 0))
        
        # For each placed box, generate corner candidates
        for (px, py, pz, pw, ph, pd) in bin_placed:
            # Corners of the placed box (right, top, front faces)
            cx_options = [px + pw]
            cy_options = [py + ph]
            cz_options = [pz + pd]
            
            for cx in cx_options:
                for cy in cy_options:
                    for cz in cz_options:
                        if cx + w <= W and cy + h <= H and cz + d <= D:
                            candidates.add((cx, cy, cz))
            
            # Also try positions adjacent to each face
            # Right face
            if px + pw + w <= W:
                for cy in [py, py + ph]:
                    for cz in [pz, pz + pd]:
                        if cy + h <= H and cz + d <= D:
                            candidates.add((px + pw, cy, cz))
            # Top face
            if py + ph + h <= H:
                for cx in [px, px + pw]:
                    for cz in [pz, pz + pd]:
                        if cx + w <= W and cz + d <= D:
                            candidates.add((cx, py + ph, cz))
            # Front face
            if pz + pd + d <= D:
                for cx in [px, px + pw]:
                    for cy in [py, py + ph]:
                        if cx + w <= W and cy + h <= H:
                            candidates.add((cx, cy, pz + pd))
        
        # Filter valid candidates (no overlap)
        valid = []
        for (x, y, z) in candidates:
            if x + w <= W and y + h <= H and z + d <= D:
                if not overlaps(x, y, z, w, h, d, bin_placed):
                    valid.append((x, y, z))
        
        # Sort by BLF: minimize z (bottom), then y (left), then x (farthest)
        valid.sort(key=lambda pos: (pos[2], pos[1], pos[0]))
        return valid
    
    # Try multiple sorting strategies
    def try_packing(order, bin_limit=None):
        """
        Try packing boxes in the given order. Returns (placements, num_bins, last_bin_vol).
        placements is list of (bin, x, y, z, w, h, d) where placements[i] corresponds to boxes[i].
        """
        bins = []  # list of lists of (x, y, z, w, h, d)
        # Initialize result array to hold placement for each original box index
        result_placements = [None] * n 
        box_to_bin = {}
        
        for idx in order:
            w, h, d = boxes[idx]
            placed = False
            
            # Try to place in existing bins first
            for b_idx in range(len(bins)):
                candidates = get_candidate_positions(bins[b_idx], w, h, d)
                for (x, y, z) in candidates:
                    # Place it
                    bins[b_idx].append((x, y, z, w, h, d))
                    result_placements[idx] = (b_idx, x, y, z, w, h, d)
                    box_to_bin[idx] = b_idx
                    placed = True
                    break
                if placed:
                    break
            
            if not placed:
                # Open a new bin
                b_idx = len(bins)
                # Place at origin
                bins.append([(0, 0, 0, w, h, d)])
                result_placements[idx] = (b_idx, 0, 0, 0, w, h, d)
                box_to_bin[idx] = b_idx
            
            # Check deadline
            if time.time() - t_start > DEADLINE:
                break
        
        # Verify all boxes were placed
        if any(p is None for p in result_placements):
            return None, 0, 0
        
        num_bins = len(bins)
        # Compute last bin volume (highest bin index that has boxes)
        last_bin_idx = max(box_to_bin.values()) if box_to_bin else 0
        last_bin_vol = sum(w * h * d for (x, y, z, w, h, d) in bins[last_bin_idx])
        
        return result_placements, num_bins, last_bin_vol
    
    # Generate several sorting orders
    orders = []
    
    # 1. Volume descending
    orders.append(sorted(range(n), key=lambda i: box_vols[i], reverse=True))
    
    # 2. Max dimension descending (largest side first)
    orders.append(sorted(range(n), key=lambda i: max(boxes[i][0], boxes[i][1], boxes[i][2]), reverse=True))
    
    # 3. Sum of dimensions descending
    orders.append(sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True))
    
    # 4. Area of largest face descending
    def largest_face_area(i):
        w, h, d = boxes[i]
        return max(w*h, w*d, h*d)
    orders.append(sorted(range(n), key=largest_face_area, reverse=True))
    
    # 5. Volume ascending (small first)
    orders.append(sorted(range(n), key=lambda i: box_vols[i]))
    
    # 6. Randomized orders
    rng = random.Random(42)
    for _ in range(5):
        shuffled = list(range(n))
        rng.shuffle(shuffled)
        orders.append(shuffled)
    
    # 7. Hybrid: large first, then fill gaps
    # Sort by volume desc, but for each box, prefer bins where it fits tightly
    orders.append(sorted(range(n), key=lambda i: (-box_vols[i], boxes[i][0], boxes[i][1], boxes[i][2])))
    
    best_placements = None
    best_score = float('inf')
    
    for order in orders:
        if time.time() - t_start > DEADLINE:
            break
        
        placements, num_bins, last_bin_vol = try_packing(order)
        if placements is None:
            continue
        
        # Compute score
        if num_bins == 0:
            score = float('inf')
        else:
            last_bin_frac = last_bin_vol / container_vol if container_vol > 0 else 0
            score = num_bins + 0.1 * (1 - last_bin_frac)
        
        if score < best_score:
            best_score = score
            best_placements = placements
    
    # If no valid placement found (shouldn't happen), fall back to simple shelf
    if best_placements is None:
        # Fallback: simple shelf packing
        placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        best_placements = placements
    
    return best_placements
# EVOLVE-BLOCK-END
