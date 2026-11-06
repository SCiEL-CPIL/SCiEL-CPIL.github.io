"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import heapq
import random

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending (largest first) for better packing
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[4])
    
    # Bins: each bin is a list of placed boxes (x, y, z, w, h, d)
    bins = []
    # Track bounding box for each bin for faster rejection
    bin_bboxes = []  # (max_x, max_y, max_z)
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        # Check if box fits within container
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check overlap with existing boxes in this bin
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # No overlap if separated in any dimension
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    def find_position(bin_boxes, w, h, d):
        """Find the best position for a box in a bin using corner-point heuristic."""
        # Generate candidate positions from corners of existing boxes
        candidates = set()
        candidates.add((0, 0, 0))
        
        if bin_boxes:
            for (px, py, pz, pw, ph, pd) in bin_boxes:
                # Generate all 8 corners of each existing box
                candidates.add((px, py, pz))
                candidates.add((px + pw, py, pz))
                candidates.add((px, py + ph, pz))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px, py, pz + pd))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
        
        # Filter valid candidates
        valid_candidates = []
        for (x, y, z) in candidates:
            if x + w <= W and y + h <= H and z + d <= D:
                if can_place(bin_boxes, x, y, z, w, h, d):
                    # Score: prefer positions closer to origin (tighter packing)
                    # Use a weighted sum that prioritizes filling gaps
                    score = x * 1.0 + y * 1.0 + z * 1.0
                    valid_candidates.append((score, x, y, z))
        
        if not valid_candidates:
            return None
        
        # Sort by score (prefer tighter positions)
        valid_candidates.sort()
        return valid_candidates[0][1:]
    
    def evaluate_bin_fit(bin_boxes, w, h, d, bb):
        """Evaluate how well a box fits in a bin. Lower is better."""
        # Quick rejection: if box doesn't fit in remaining space
        # Based on bounding box
        if bb is not None:
            max_x, max_y, max_z = bb
            if max_x + w > W and max_y + h > H and max_z + d > D:
                # Check if it could fit anywhere in the bin
                pass  # Fall through to find_position
        
        pos = find_position(bin_boxes, w, h, d)
        if pos is None:
            return None, None
        x, y, z = pos
        # Score: prefer positions that minimize the "corner" distance
        # This encourages filling compactly
        return (x + y + z), pos
    
    def place_box(w, h, d):
        """Place a box in the best existing bin or create a new one."""
        best_bin = -1
        best_score = float('inf')
        best_pos = None
        
        for b_idx, bin_boxes in enumerate(bins):
            bb = bin_bboxes[b_idx] if b_idx < len(bin_bboxes) else None
            score, pos = evaluate_bin_fit(bin_boxes, w, h, d, bb)
            if score is not None and score < best_score:
                best_score = score
                best_bin = b_idx
                best_pos = pos
        
        if best_bin == -1:
            # Create a new bin
            bins.append([])
            bin_bboxes.append((w, h, d))
            best_bin = len(bins) - 1
            best_pos = (0, 0, 0)
        else:
            # Update bounding box
            x, y, z = best_pos
            old_bb = bin_bboxes[best_bin]
            new_bb = (max(old_bb[0], x + w), max(old_bb[1], y + h), max(old_bb[2], z + d))
            bin_bboxes[best_bin] = new_bb
        
        x, y, z = best_pos
        bins[best_bin].append((x, y, z, w, h, d))
        return best_bin, x, y, z
    
    # Place boxes in sorted order
    placements = [None] * n
    for idx, w, h, d, vol in indexed_boxes:
        if time.time() > deadline:
            # Emergency: place remaining boxes in a new bin each (valid but suboptimal)
            for i, (w2, h2, d2) in enumerate(boxes[idx:]):
                bin_idx = len(bins)
                bins.append([])
                bin_bboxes.append((w2, h2, d2))
                bins[bin_idx].append((0, 0, 0, w2, h2, d2))
                placements[idx + i] = (bin_idx, 0, 0, 0, w2, h2, d2)
            break
        
        bin_idx, x, y, z = place_box(w, h, d)
        placements[idx] = (bin_idx, x, y, z, w, h, d)
    
    # Fill any None placements (shouldn't happen if deadline guard works)
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            bin_idx = len(bins)
            bins.append([])
            bin_bboxes.append((w, h, d))
            bins[bin_idx].append((0, 0, 0, w, h, d))
            placements[i] = (bin_idx, 0, 0, 0, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
