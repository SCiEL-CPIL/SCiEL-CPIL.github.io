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
        """Find the best position for a box in a bin using maximal space partitioning."""
        # Generate candidate positions using MSP approach
        # Start with the entire container as one space
        spaces = [(0, 0, 0, W, H, D)]
        
        # For each existing box, split the spaces it occupies
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            new_spaces = []
            for (sx, sy, sz, sw, sh, sd) in spaces:
                # Check if this space overlaps with the box
                if sx + sw <= px or px + pw <= sx or \
                   sy + sh <= py or py + ph <= sy or \
                   sz + sd <= pz or pz + pd <= sz:
                    # No overlap, keep the space
                    new_spaces.append((sx, sy, sz, sw, sh, sd))
                else:
                    # Overlap exists, split the space
                    # Split in x direction if box is within x range
                    if sx < px and px + pw < sx + sw:
                        # Left part
                        left_w = px - sx
                        if left_w > 0:
                            new_spaces.append((sx, sy, sz, left_w, sh, sd))
                        # Right part
                        right_x = px + pw
                        right_w = sx + sw - right_x
                        if right_w > 0:
                            new_spaces.append((right_x, sy, sz, right_w, sh, sd))
                    elif sx <= px and px + pw <= sx + sw:
                        # Box is fully within x range of space
                        # Split into left, right, and middle (above/below)
                        left_w = px - sx
                        if left_w > 0:
                            new_spaces.append((sx, sy, sz, left_w, sh, sd))
                        right_x = px + pw
                        right_w = sx + sw - right_x
                        if right_w > 0:
                            new_spaces.append((right_x, sy, sz, right_w, sh, sd))
                        # Middle part: split in y direction
                        if sy < py and py + ph < sy + sh:
                            bottom_h = py - sy
                            if bottom_h > 0:
                                new_spaces.append((px, sy, sz, pw, bottom_h, sd))
                            top_y = py + ph
                            top_h = sy + sh - top_y
                            if top_h > 0:
                                new_spaces.append((px, top_y, sz, pw, top_h, sd))
                        elif sy <= py and py + ph <= sy + sh:
                            # Box fully within y range
                            bottom_h = py - sy
                            if bottom_h > 0:
                                new_spaces.append((px, sy, sz, pw, bottom_h, sd))
                            top_y = py + ph
                            top_h = sy + sh - top_y
                            if top_h > 0:
                                new_spaces.append((px, top_y, sz, pw, top_h, sd))
                            # Middle part: split in z direction
                            if sz < pz and pz + pd < sz + sd:
                                front_d = pz - sz
                                if front_d > 0:
                                    new_spaces.append((px, py, sz, pw, ph, front_d))
                                back_z = pz + pd
                                back_d = sz + sd - back_z
                                if back_d > 0:
                                    new_spaces.append((px, py, back_z, pw, ph, back_d))
                            elif sz <= pz and pz + pd <= sz + sd:
                                front_d = pz - sz
                                if front_d > 0:
                                    new_spaces.append((px, py, sz, pw, ph, front_d))
                                back_z = pz + pd
                                back_d = sz + sd - back_z
                                if back_d > 0:
                                    new_spaces.append((px, py, back_z, pw, ph, back_d))
                    else:
                        # Partial overlap in x, just keep as is (simplified)
                        new_spaces.append((sx, sy, sz, sw, sh, sd))
            spaces = new_spaces
        
        # Generate candidate positions from the maximal spaces
        candidates = set()
        candidates.add((0, 0, 0))
        
        for (sx, sy, sz, sw, sh, sd) in spaces:
            # Candidate at the corner of each maximal space
            if sx + w <= W and sy + h <= H and sz + d <= D:
                candidates.add((sx, sy, sz))
        
        # Also add face-based candidates for robustness
        if bin_boxes:
            for (px, py, pz, pw, ph, pd) in bin_boxes:
                candidates.add((px + pw, py, pz))
                candidates.add((px, py + ph, pz))
                candidates.add((px, py, pz + pd))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
        
        # Filter valid candidates
        valid_candidates = []
        for (x, y, z) in candidates:
            if x + w <= W and y + h <= H and z + d <= D:
                if can_place(bin_boxes, x, y, z, w, h, d):
                    # Score: prefer positions closer to origin (tighter packing)
                    score = x * 1.0 + y * 1.0 + z * 1.0
                    valid_candidates.append((score, x, y, z))
        
        if not valid_candidates:
            return None
        
        # Sort by score (prefer tighter positions)
        valid_candidates.sort()
        return valid_candidates[0][1:]
    
    def evaluate_bin_fit(bin_boxes, w, h, d, bb):
        """Evaluate how well a box fits in a bin. Lower is better."""
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
