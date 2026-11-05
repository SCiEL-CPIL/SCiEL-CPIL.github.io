"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    # 120s limit, keep 10s margin for safety/overhead
    deadline = start_time + 110.0
    
    # Preprocess: store box dimensions and original indices
    box_data = []
    for i, (w, h, d) in enumerate(boxes):
        box_data.append((i, w, h, d))
        
    # Initial result: all boxes left out
    final_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    def get_rotations(w, h, d):
        # Generate unique rotations
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        return list(rots)

    def can_place(placed, x, y, z, bw, bh, bd):
        """Check if a box of size (bw, bh, bd) at (x, y, z) overlaps with any placed box."""
        for (px, py, pz, pw, ph, pd) in placed:
            if not (x + bw <= px or px + pw <= x or 
                    y + bh <= py or py + ph <= y or 
                    z + bd <= pz or pz + pd <= z):
                return False
        return True

    def find_best_fit(placed, bw, bh, bd):
        """
        Find the 'best fit' position for a box.
        Best fit is defined as the position that minimizes the sum of distances
        to the closest placed box on each axis (encouraging tight packing).
        Returns (x, y, z) or None.
        """
        if bw > W or bh > H or bd > D:
            return None

        # Candidate coordinates are 0 and the max extent of placed boxes
        x_candidates = set([0])
        y_candidates = set([0])
        z_candidates = set([0])
        
        for (px, py, pz, pw, ph, pd) in placed:
            x_candidates.add(px + pw)
            y_candidates.add(py + ph)
            z_candidates.add(pz + pd)
            
        # Filter candidates that fit within container bounds
        x_candidates = [x for x in x_candidates if x + bw <= W]
        y_candidates = [y for y in y_candidates if y + bh <= H]
        z_candidates = [z for z in z_candidates if z + bd <= D]
        
        if not x_candidates or not y_candidates or not z_candidates:
            return None

        best_score = None
        best_coord = None
        
        # Heuristic: Minimize "wasted space" or "distance to neighbors"
        # A common effective heuristic is to minimize (x + y + z) but that's BLF.
        # For Best Fit, we want to minimize the gap.
        # Let's try a composite score: minimize (x + y + z) primarily, but also check for "holes".
        # Actually, a robust Best-Fit is: for each valid position, calculate the 
        # minimum distance to any existing box face. Lower is better.
        
        # Optimization: Instead of checking all N^3 positions against all M placed boxes (N^3*M),
        # we can precompute the "occupied" status or just iterate smartly.
        # Given N=60, N^3 is ~216,000. M is up to 60. 216,000 * 60 = 12.9M checks per box.
        # 60 boxes * 12.9M = 774M checks. This is too slow for Python in 110s.
        
        # We need a faster check or fewer candidates.
        # Strategy: Only consider positions adjacent to existing boxes.
        # The candidate sets are already small (at most M+1 per axis).
        # The bottleneck is the `can_place` check which is O(M).
        # Total ops: Sum over boxes of (Nx * Ny * Nz * M).
        # If M is small initially, it's fast. If M is large, Nx, Ny, Nz are also large.
        
        # To speed up, we can use a 3D grid or just accept the cost if we limit candidates.
        # Let's limit candidates to the "frontier".
        
        for z in z_candidates:
            for y in y_candidates:
                for x in x_candidates:
                    if can_place(placed, x, y, z, bw, bh, bd):
                        # Score: Prefer positions that are "lower" (z), "left" (x), "back" (y)
                        # But also prefer "tight" packing.
                        # Let's use a simple score: minimize (z * 1000 + x * 10 + y)
                        # This prioritizes filling the bottom layer first.
                        score = (z, x, y)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_coord = (x, y, z)
                            
        return best_coord

    def run_packing_pass(boxes_to_place, placed, final_placements, deadline):
        """
        Attempt to place boxes from boxes_to_place into the container.
        Returns the number of newly placed boxes.
        """
        placed_count = 0
        # Sort boxes by volume descending for this pass
        boxes_to_place = sorted(boxes_to_place, key=lambda b: b[1]*b[2]*b[3], reverse=True)
        
        for idx, w, h, d in boxes_to_place:
            if time.time() > deadline:
                break
                
            rots = get_rotations(w, h, d)
            placed_this = False
            
            # Try each rotation
            for rw, rh, rd in rots:
                pos = find_best_fit(placed, rw, rh, rd)
                if pos is not None:
                    x, y, z = pos
                    placed.append((x, y, z, rw, rh, rd))
                    final_placements[idx] = (0, x, y, z, rw, rh, rd)
                    placed_this = True
                    break
                    
            if not placed_this:
                # If we couldn't place it now, maybe we can later? 
                # In a simple pass, we skip it. 
                # To improve, we could re-insert it into a queue, but that complicates the loop.
                # Instead, we will do multiple passes over the *entire* set of unplaced boxes.
                pass

        return placed_count

    # --- Main Strategy: Multi-pass Best Fit ---
    
    # List of indices not yet placed
    unplaced_indices = list(range(n))
    # List of boxes currently placed
    placed = []
    
    # We will do multiple passes. In each pass, we try to place any box that fits.
    # We sort by volume descending initially, but in subsequent passes, 
    # we might try smaller boxes first to fill gaps, or stick to volume to keep big items prominent.
    # Let's stick to Volume Descending for stability, but the "Best Fit" position selection
    # is the key differentiator from "First Fit".
    
    max_passes = 5
    for p in range(max_passes):
        if time.time() > deadline:
            break
            
        # Create a list of (index, w, h, d) for unplaced boxes
        candidates = []
        for i in unplaced_indices:
            w, h, d = boxes[i]
            candidates.append((i, w, h, d))
            
        if not candidates:
            break
            
        # Sort by volume descending
        candidates.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
        
        newly_placed = 0
        still_unplaced = []
        
        for idx, w, h, d in candidates:
            if time.time() > deadline:
                # If time runs out, we must ensure all remaining are marked as unplaced (which they are by default)
                # But we need to stop the loop cleanly.
                break
                
            rots = get_rotations(w, h, d)
            placed_this = False
            
            for rw, rh, rd in rots:
                pos = find_best_fit(placed, rw, rh, rd)
                if pos is not None:
                    x, y, z = pos
                    placed.append((x, y, z, rw, rh, rd))
                    final_placements[idx] = (0, x, y, z, rw, rh, rd)
                    placed_this = True
                    newly_placed += 1
                    break
                    
            if not placed_this:
                still_unplaced.append(idx)
                
        # If no new boxes were placed in this pass, further passes won't help with this simple algorithm
        if newly_placed == 0:
            break
            
        # Update unplaced list for next pass
        unplaced_indices = still_unplaced
        
    return final_placements
# EVOLVE-BLOCK-END
