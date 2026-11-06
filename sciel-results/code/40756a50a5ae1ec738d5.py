import time
import sys
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Pre-compute box dimensions and volumes
    box_dims = [boxes[i] for i in range(n)]
    box_vols = [w*h*d for (w,h,d) in box_dims]
    
    # Sort indices by volume descending
    sorted_indices = sorted(range(n), key=lambda i: box_vols[i], reverse=True)
    
    # We will maintain a list of bins. Each bin is a list of placed box indices.
    # We also maintain the position of each placed box.
    # positions[bin_idx] is a list of (x, y, z, w, h, d)
    bins = []
    bin_volumes = [0] * 100  # Pre-allocate some, expand if needed
    placements = [None] * n  # (bin_idx, x, y, z, w, h, d)
    
    # To speed up overlap checking, we can maintain a simple list of boxes per bin.
    # Since n=40, O(N^2) per placement is fine (40*40*40 = 64k ops per bin check, * 40 bins * 40 boxes = 10M ops, very fast).
    
    def can_place(bin_idx, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (ox, oy, oz, ow, oh, od) in bins[bin_idx]:
            # Check overlap
            if not (x + w <= ox or ox + ow <= x or
                    y + h <= oy or oy + oh <= y or
                    z + d <= oz or oz + od <= z):
                return False
        return True

    def find_best_position(bin_idx, w, h, d):
        """
        Find the best position for a box in a given bin.
        Uses a candidate generation strategy based on existing boxes' corners.
        Returns (x, y, z) or None.
        """
        candidates = set()
        candidates.add((0, 0, 0))
        
        # Generate candidates from existing boxes
        for (ox, oy, oz, ow, oh, od) in bins[bin_idx]:
            # Faces and corners of existing boxes
            candidates.add((ox + ow, oy, oz))
            candidates.add((ox, oy + oh, oz))
            candidates.add((ox, oy, oz + od))
            candidates.add((ox + ow, oy + oh, oz))
            candidates.add((ox + ow, oy, oz + od))
            candidates.add((ox, oy + oh, oz + od))
            candidates.add((ox + ow, oy + oh, oz + od))
            
            # Also try aligning with the "front" of the bin
            # (0, oy, oz), (ox, 0, oz), (ox, oy, 0) are often good too
            candidates.add((0, oy, oz))
            candidates.add((ox, 0, oz))
            candidates.add((ox, oy, 0))
            
            # And combinations
            candidates.add((ox + ow, 0, oz))
            candidates.add((ox, oy + oh, oz))
            candidates.add((0, oy, oz + od))
            candidates.add((ox + ow, oy, oz + od))
            candidates.add((ox, 0, oz + od))
            candidates.add((0, oy + oh, oz + od))
            
            candidates.add((ox + ow, oy + oh, 0))
            candidates.add((ox + ow, 0, oz + od))
            candidates.add((0, oy + oh, oz + od))
            candidates.add((ox, oy + oh, oz + od))
            candidates.add((ox + ow, oy, oz + od))
            candidates.add((ox, 0, oz + od))
            candidates.add((ox + ow, oy + oh, oz + od))
        
        # Sort candidates by a heuristic: prefer lower x, then lower y, then lower z
        # This encourages packing towards the origin (0,0,0)
        # A better heuristic might be minimizing the "wasted space" above, to the right, etc.
        # But for now, lexicographic order (x, y, z) is a standard "Bottom-Left-Back" heuristic.
        
        valid_candidates = []
        for cx, cy, cz in candidates:
            if 0 <= cx < W and 0 <= cy < H and 0 <= cz < D:
                if can_place(bin_idx, cx, cy, cz, w, h, d):
                    valid_candidates.append((cx, cy, cz))
        
        if not valid_candidates:
            return None
        
        # Sort by x, then y, then z
        valid_candidates.sort(key=lambda p: (p[0], p[1], p[2]))
        
        return valid_candidates[0]

    def evaluate_bin(bin_idx, w, h, d):
        """
        Returns (fits, score) for placing box in bin_idx.
        Lower score is better.
        """
        pos = find_best_position(bin_idx, w, h, d)
        if pos is None:
            return False, float('inf')
        x, y, z = pos
        # Heuristic: minimize the "distance" from origin, or maximize the "tightness"
        # Using x + y + z is a simple proxy for "how much space is used up".
        # A more sophisticated metric could be the volume of the bounding box of the bin contents after placement.
        # For now, let's stick with x+y+z to encourage corner packing.
        score = x + y + z
        return True, score

    # Main loop: Place boxes one by one
    for idx in sorted_indices:
        if time.time() > deadline:
            # Fallback: place remaining boxes in new bins to ensure validity
            for remaining_idx in sorted_indices:
                if placements[remaining_idx] is None:
                    w, h, d = box_dims[remaining_idx]
                    b = len(bins)
                    bins.append([(0, 0, 0, w, h, d)])
                    if b >= len(bin_volumes):
                        bin_volumes.append(0)
                    bin_volumes[b] = w * h * d
                    placements[remaining_idx] = (b, 0, 0, 0, w, h, d)
            break
        
        w, h, d = box_dims[idx]
        
        best_bin = -1
        best_score = float('inf')
        best_pos = None
        
        # Try existing bins
        for b in range(len(bins)):
            fits, score = evaluate_bin(b, w, h, d)
            if fits and score < best_score:
                best_score = score
                best_bin = b
                best_pos = find_best_position(b, w, h, d)
        
        if best_bin != -1:
            bins[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
            bin_volumes[best_bin] += w * h * d
            placements[idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d)
        else:
            # Open a new bin
            b = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            if b >= len(bin_volumes):
                bin_volumes.append(0)
            bin_volumes[b] = w * h * d
            placements[idx] = (b, 0, 0, 0, w, h, d)

    # If we broke out early due to time, ensure all boxes are placed
    for i in range(n):
        if placements[i] is None:
            w, h, d = box_dims[i]
            b = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            if b >= len(bin_volumes):
                bin_volumes.append(0)
            bin_volumes[b] = w * h * d
            placements[i] = (b, 0, 0, 0, w, h, d)

    return placements
