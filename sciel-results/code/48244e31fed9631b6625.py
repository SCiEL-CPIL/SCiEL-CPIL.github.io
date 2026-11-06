"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import product

def pack(boxes, container):
    """
    3D Bin Packing with Best-Fit Decreasing and multi-anchor point search.
    
    Returns list of (bin_id, x, y, z, w, h, d) for each box in original order.
    """
    W, H, D = container
    n = len(boxes)
    
    # Sort boxes by volume descending, keep original index
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[1])
    
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 115.0  # 5s safety margin before 120s kill
    
    # Each bin is a list of placed boxes: [(x, y, z, w, h, d), ...]
    bins = []
    placement = {}  # original index -> (bin_idx, x, y, z)
    
    def check_overlap(placed, x, y, z, w, h, d):
        """Check if a box at (x,y,z) with dims (w,h,d) overlaps with any in placed."""
        for (px, py, pz, pw, ph, pd) in placed:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return True
        return False
    
    def get_anchor_positions(placed, w, h, d):
        """
        Generate candidate (x, y, z) positions using anchor points.
        Anchors are the "front" faces of existing boxes plus the origin.
        """
        x_vals = {0}
        y_vals = {0}
        z_vals = {0}
        
        for (px, py, pz, pw, ph, pd) in placed:
            x_vals.add(px + pw)
            y_vals.add(py + ph)
            z_vals.add(pz + pd)
        
        # Filter to valid starting positions
        x_vals = [x for x in x_vals if x + w <= W]
        y_vals = [y for y in y_vals if y + h <= H]
        z_vals = [z for z in z_vals if z + d <= D]
        
        if not x_vals or not y_vals or not z_vals:
            return []
        
        x_vals.sort()
        y_vals.sort()
        z_vals.sort()
        
        candidates = []
        for x in x_vals:
            for y in y_vals:
                for z in z_vals:
                    if x + w > W or y + h > H or z + d > D:
                        continue
                    if not check_overlap(placed, x, y, z, w, h, d):
                        candidates.append((x, y, z))
        
        return candidates
    
    def compute_cost(x, y, z, w, h, d, placed):
        """
        Compute a cost for a placement. Lower is better.
        We want to pack tightly, so minimize the "bounding box" expansion.
        """
        if not placed:
            return x + y + z
        
        # Find the current bounding box of placed items
        max_x = max(px + pw for (px, py, pz, pw, ph, pd) in placed)
        max_y = max(py + ph for (px, py, pz, pw, ph, pd) in placed)
        max_z = max(pz + pd for (px, py, pz, pw, ph, pd) in placed)
        
        new_max_x = max(max_x, x + w)
        new_max_y = max(max_y, y + h)
        new_max_z = max(max_z, z + d)
        
        # Cost: increase in bounding box volume + distance from origin
        bbox_increase = (new_max_x * new_max_y * new_max_z) - (max_x * max_y * max_z)
        dist = x + y + z
        
        return bbox_increase + 0.1 * dist
    
    # Process boxes in decreasing volume order
    for idx, vol, w, h, d in indexed_boxes:
        # Time check
        if time.time() > deadline:
            # Fallback: place remaining boxes in new bins at origin
            for idx2, vol2, w2, h2, d2 in indexed_boxes:
                if idx2 in placement:
                    continue
                bins.append([(0, 0, 0, w2, h2, d2)])
                placement[idx2] = (len(bins) - 1, 0, 0, 0)
            
            result = []
            for i in range(n):
                b_idx, x, y, z = placement[i]
                w_i, h_i, d_i = boxes[i]
                result.append((b_idx, x, y, z, w_i, h_i, d_i))
            return result
        
        # Try to place in existing bins
        best_bin = -1
        best_x, best_y, best_z = 0, 0, 0
        best_cost = float('inf')
        
        for b_idx in range(len(bins)):
            placed = bins[b_idx]
            candidates = get_anchor_positions(placed, w, h, d)
            for (cx, cy, cz) in candidates:
                cost = compute_cost(cx, cy, cz, w, h, d, placed)
                if cost < best_cost:
                    best_cost = cost
                    best_bin = b_idx
                    best_x, best_y, best_z = cx, cy, cz
        
        if best_bin == -1:
            # Create new bin
            bins.append([])
            best_bin = len(bins) - 1
            best_x, best_y, best_z = 0, 0, 0
        
        # Place the box
        bins[best_bin].append((best_x, best_y, best_z, w, h, d))
        placement[idx] = (best_bin, best_x, best_y, best_z)
    
    # Build result in original order
    result = []
    for i in range(n):
        b_idx, x, y, z = placement[i]
        w_i, h_i, d_i = boxes[i]
        result.append((b_idx, x, y, z, w_i, h_i, d_i))
    
    return result
# EVOLVE-BLOCK-END
