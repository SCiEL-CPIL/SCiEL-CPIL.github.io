"""3D Bin Packing with Best-Fit Decreasing + local improvement."""
import time
import sys
import math
import random

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending (largest first)
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    def run_bfd(sorted_indices):
        """Run Best-Fit Decreasing heuristic. Returns list of (bin, x, y, z, w, h, d)."""
        # Bins: each bin is a list of placed boxes (x, y, z, w, h, d)
        bins = []
        placements = [None] * n  # indexed by original box index
        
        for idx in sorted_indices:
            orig_idx, w, h, d = indexed_boxes[idx]
            
            best_bin = -1
            best_score = float('inf')
            best_pos = None
            
            for b in range(len(bins)):
                # Try to find a position in this bin
                pos = find_position(bins[b], w, h, d, W, H, D)
                if pos is not None:
                    x, y, z = pos
                    # Score: prefer lower z, then lower y, then lower x (bottom-left-back)
                    score = z * 10000 + y * 100 + x
                    if score < best_score:
                        best_score = score
                        best_bin = b
                        best_pos = pos
            
            if best_bin == -1:
                # Open a new bin
                bins.append([(0, 0, 0, w, h, d)])
                best_bin = len(bins) - 1
                best_pos = (0, 0, 0)
            
            x, y, z = best_pos
            bins[best_bin].append((x, y, z, w, h, d))
            placements[orig_idx] = (best_bin, x, y, z, w, h, d)
        
        return placements, bins
    
    def find_position(placed, w, h, d, W, H, D):
        """Find a valid (x, y, z) position for a box of size (w, h, d) in a bin with 'placed' boxes.
        Uses bottom-left-back search: try to place as low and left as possible."""
        
        # Generate candidate positions based on existing boxes
        # Candidate x: 0 and right edge of each placed box
        # Candidate y: 0 and front edge of each placed box  
        # Candidate z: 0 and top edge of each placed box
        
        candidate_xs = [0]
        candidate_ys = [0]
        candidate_zs = [0]
        
        for (px, py, pz, pw, ph, pd) in placed:
            candidate_xs.append(px + pw)
            candidate_ys.append(py + pd)  # y is depth in our convention? Let me check...
            candidate_zs.append(pz + ph)
        
        # Wait, I need to be careful about axis conventions.
        # The container is (W, H, D). Box is (w, h, d).
        # Placement is (bin, x, y, z, w, h, d).
        # x is along W, y is along H, z is along D? Or x along W, y along D, z along H?
        # Looking at the seed: W,H,D = container; box is w,h,d.
        # The seed uses x for W dimension, y for H dimension, z for D dimension.
        # So: x ∈ [0, W], y ∈ [0, H], z ∈ [0, D].
        # Box occupies [x, x+w] in x, [y, y+h] in y, [z, z+d] in z.
        
        # Redo candidate positions:
        candidate_xs = [0]
        candidate_ys = [0]
        candidate_zs = [0]
        
        for (px, py, pz, pw, ph, pd) in placed:
            candidate_xs.append(px + pw)
            candidate_ys.append(py + ph)
            candidate_zs.append(pz + pd)
        
        # Remove candidates that exceed container
        candidate_xs = [x for x in candidate_xs if x + w <= W]
        candidate_ys = [y for y in candidate_ys if y + h <= H]
        candidate_zs = [z for z in candidate_zs if z + d <= D]
        
        # Sort candidates to prefer bottom-left-back
        candidate_xs.sort()
        candidate_ys.sort()
        candidate_zs.sort()
        
        best_pos = None
        best_score = float('inf')
        
        for z in candidate_zs:
            for y in candidate_ys:
                for x in candidate_xs:
                    if not is_valid_position(placed, x, y, z, w, h, d, W, H, D):
                        continue
                    # Bottom-left-back: minimize z, then y, then x
                    score = z * 10000 + y * 100 + x
                    if score < best_score:
                        best_score = score
                        best_pos = (x, y, z)
        
        return best_pos
    
    def is_valid_position(placed, x, y, z, w, h, d, W, H, D):
        """Check if a box of size (w,h,d) at (x,y,z) doesn't overlap any placed box and fits in container."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in placed:
            # Check overlap
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    # Run initial BFD
    initial_order = list(range(n))
    placements, bins = run_bfd(initial_order)
    
    best_placements = placements
    best_num_bins = len(bins)
    
    # Calculate initial score
    def compute_score(placements):
        if not placements:
            return 0.0
        bin_labels = set(p[0] for p in placements)
        B = len(bin_labels)
        max_bin = max(bin_labels)
        # Last bin occupied volume
        last_bin_vol = 0
        for p in placements:
            if p[0] == max_bin:
                last_bin_vol += p[4] * p[5] * p[6]
        container_vol = W * H * D
        if container_vol > 0:
            frac = last_bin_vol / container_vol
        else:
            frac = 0.0
        return B + 0.1 * (1 - frac)
    
    best_score = compute_score(placements)
    
    elapsed = time.time() - start_time
    remaining = deadline - start_time - 5.0  # 5s safety margin
    
    if remaining <= 0:
        return best_placements
    
    # Try local search: swap pairs of boxes in the ordering
    # We'll do a few iterations of pairwise swaps to try to improve
    
    max_iterations = 200
    iteration = 0
    while iteration < max_iterations:
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            break
        
        # Pick two random indices to swap
        i, j = random.sample(range(n), 2)
        
        new_order = initial_order[:]
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        # Re-run BFD with new order
        # For speed, we only re-run if it's likely to help
        new_placements, new_bins = run_bfd(new_order)
        new_score = compute_score(new_placements)
        
        if new_score < best_score:
            best_score = new_score
            best_placements = new_placements
            initial_order = new_order
            best_num_bins = len(new_bins)
        
        iteration += 1
    
    return best_placements
