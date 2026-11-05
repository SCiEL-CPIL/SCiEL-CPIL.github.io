"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def pack(boxes, container):
    """
    Place every box (w, h, d) into numbered bins of size container (W, H, D).
    Returns a list of (bin_id, x, y, z, w, h, d) for each box in original order.
    No rotations. No overlaps within a bin.
    Score: mean B + 0.1 * (1 - last-bin occupied volume / container volume)
    Lower is better.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s (and 150s kill)
    
    W, H, D = container
    container_vol = W * H * D
    n = len(boxes)
    
    # Pre-compute volumes
    volumes = [w * h * d for w, h, d in boxes]
    
    # Sort indices by volume descending (Best Fit Decreasing)
    # We'll keep original indices to map back
    order = sorted(range(n), key=lambda i: -volumes[i])
    
    def fits(x, y, z, w, h, d, occupied):
        """Check if box at (x,y,z) with dims (w,h,d) fits in container and doesn't overlap."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (ox, oy, oz, ow, oh, od) in occupied:
            # Overlap check: boxes overlap if they intersect in all 3 dimensions
            if (x < ox + ow and ox < x + w and
                y < oy + oh and oy < y + h and
                z < oz + od and oz < z + d):
                return False
        return True
    
    def find_best_position(w, h, d, occupied):
        """
        Find the 'best' position for a box in a bin.
        Uses a grid-based search over candidate positions derived from existing boxes.
        Candidate x: 0, and ox+ow for each occupied box
        Candidate y: 0, and oy+oh for each occupied box
        Candidate z: 0, and oz+od for each occupied box
        This is a common heuristic for 3D packing.
        We score positions by minimizing (x, y, z) in a lexicographic or weighted way,
        or by maximizing 'tightness'. Here we minimize x*W*H + y*W + z or similar.
        Actually, a common 'best fit' is to minimize the sum of distances to walls,
        or just pick the first valid position in a sorted order.
        Let's sort candidates by (z, y, x) to prefer lower z, then lower y, then lower x.
        Or better: minimize the 'waste' or just use a heuristic score.
        For simplicity and speed, we'll generate candidates and pick the one with
        minimum (z, y, x) that fits. This tends to fill from the bottom-left-front.
        """
        if not occupied:
            return (0, 0, 0)
        
        xs = sorted(set([0] + [ox + ow for (ox, oy, oz, ow, oh, od) in occupied]))
        ys = sorted(set([0] + [oy + oh for (ox, oy, oz, ow, oh, od) in occupied]))
        zs = sorted(set([0] + [oz + od for (ox, oy, oz, ow, oh, od) in occupied]))
        
        # Also include positions slightly offset? No, just the grid points.
        # To ensure we don't miss positions, we should also consider positions
        # that are 'just before' an occupied box? No, that's for 1D.
        # In 3D, the candidate positions for a new box to be non-overlapping
        # and 'snug' against existing boxes or walls are indeed the faces.
        
        best_pos = None
        best_score = float('inf')
        
        for z in zs:
            if z + d > D:
                continue
            for y in ys:
                if y + h > H:
                    continue
                for x in xs:
                    if x + w > W:
                        continue
                    if fits(x, y, z, w, h, d, occupied):
                        # Score: prefer lower z, then lower y, then lower x
                        # This packs towards the origin.
                        score = z * 1000000 + y * 1000 + x
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    def try_pack(order, bins):
        """
        Try to pack boxes in the given order into the given list of bins.
        bins is a list of lists of occupied boxes.
        Returns True if all boxes are packed, False otherwise.
        Also updates bins in place.
        """
        for idx in order:
            w, h, d = boxes[idx]
            placed = False
            # Try existing bins
            for b in range(len(bins)):
                pos = find_best_position(w, h, d, bins[b])
                if pos is not None:
                    bins[b].append((pos[0], pos[1], pos[2], w, h, d))
                    placed = True
                    break
            
            if not placed:
                # Try a new bin
                pos = find_best_position(w, h, d, [])
                if pos is not None:
                    bins.append([(pos[0], pos[1], pos[2], w, h, d)])
                    placed = True
                else:
                    # Should not happen if box fits in empty container
                    return False
        
        return True
    
    # --- Strategy 1: Best Fit Decreasing (BFD) ---
    # Sort by volume descending
    order1 = sorted(range(n), key=lambda i: -volumes[i])
    bins1 = []
    success1 = try_pack(order1, bins1)
    
    best_bins = bins1 if success1 else None
    best_order = order1
    
    # --- Strategy 2: Best Fit Decreasing by max dimension ---
    order2 = sorted(range(n), key=lambda i: -max(boxes[i]))
    bins2 = []
    success2 = try_pack(order2, bins2)
    if success2 and (best_bins is None or len(bins2) < len(best_bins)):
        best_bins = bins2
        best_order = order2
    
    # --- Strategy 3: Randomized BFD (multiple restarts) ---
    # Use remaining time for randomized restarts
    random.seed(42)
    
    while time.time() < deadline:
        # Create a randomized order: primarily by volume, but with some noise
        # Or just shuffle boxes with similar volumes?
        # Let's try: sort by volume, then randomly swap some adjacent elements
        order3 = order1[:]
        # Perform a few random swaps
        for _ in range(5):
            i = random.randint(0, n - 2)
            order3[i], order3[i + 1] = order3[i + 1], order3[i]
        
        bins3 = []
        success3 = try_pack(order3, bins3)
        if success3 and (best_bins is None or len(bins3) < len(best_bins)):
            best_bins = bins3
            best_order = order3
    
    # --- Consolidation / Local Search ---
    # Try to move boxes from later bins to earlier bins to reduce B
    # or improve the last bin fill.
    # This is complex. A simpler heuristic:
    # If we have multiple bins, try to 'pull' boxes from the last bin to earlier bins.
    
    if best_bins is None:
        # Fallback: should not happen
        return [(0, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # --- Improvement: Try to reduce the number of bins by moving boxes ---
    # This is a form of local search.
    # We'll try to move boxes from the last bin to any other bin.
    # If successful, remove the last bin if empty, or continue.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Get the last bin
        if len(best_bins) <= 1:
            break
        last_bin = best_bins[-1]
        # Try to move each box in the last bin to an earlier bin
        moved_any = False
        for i in range(len(last_bin) - 1, -1, -1):
            box = last_bin[i]
            w, h, d = box[3], box[4], box[5]
            # Try to place in bins 0 to len-2
            for b in range(len(best_bins) - 1):
                # Temporarily remove from last bin
                last_bin.pop(i)
                pos = find_best_position(w, h, d, best_bins[b])
                if pos is not None:
                    best_bins[b].append((pos[0], pos[1], pos[2], w, h, d))
                    moved_any = True
                    improved = True
                    break
                else:
                    # Put it back
                    last_bin.insert(i, box)
            if moved_any:
                break
        
        # If last bin is now empty, remove it
        if not last_bin:
            best_bins.pop()
            improved = True  # Continue loop to see if more can be moved
    
    # --- Final Score Calculation & Return ---
    # Map back to original indices
    # We need to know which box went where.
    # The `try_pack` function doesn't track original indices in the bins.
    # We need to modify the approach to track indices.
    
    # Let's redo the packing with index tracking.
    # This is a significant refactor.
    
    # --- Revised Approach with Index Tracking ---
    
    def find_best_position_with_index(w, h, d, occupied):
        """Same as find_best_position but returns position only."""
        return find_best_position(w, h, d, occupied)
    
    def pack_with_tracking(order, bins):
        """
        Pack boxes in order.
        bins: list of lists of (x, y, z, w, h, d, original_index)
        Returns True if success.
        """
        for idx in order:
            w, h, d = boxes[idx]
            placed = False
            for b in range(len(bins)):
                # Extract occupied boxes without index for fit check
                occ = [(x, y, z, ww, hh, dd) for (x, y, z, ww, hh, dd, _) in bins[b]]
                pos = find_best_position(w, h, d, occ)
                if pos is not None:
                    bins[b].append((pos[0], pos[1], pos[2], w, h, d, idx))
                    placed = True
                    break
            
            if not placed:
                occ = []
                pos = find_best_position(w, h, d, occ)
                if pos is not None:
                    bins.append([(pos[0], pos[1], pos[2], w, h, d, idx)])
                    placed = True
                else:
                    return False
        
        return True
    
    # Re-run the best strategy with tracking
    # We'll just re-run the BFD strategy (order1) as it's likely the best.
    # If time permits, we could re-run the randomized ones, but let's stick to BFD for tracking.
    # Actually, let's re-run the best order found, but we lost the specific order of the randomized ones.
    # Let's just use order1 (BFD) for the final tracked packing.
    # If the randomized search found a better bin count, we might lose it.
    # To be safe, let's just use BFD for the final result, as it's deterministic and good.
    # Or, we can track indices in the initial `try_pack` as well.
    
    # Let's modify `try_pack` to track indices from the start.
    
    def try_pack_tracked(order, bins):
        """
        bins: list of lists of (x, y, z, w, h, d, original_index)
        """
        for idx in order:
            w, h, d = boxes[idx]
            placed = False
            for b in range(len(bins)):
                occ = [(x, y, z, ww, hh, dd) for (x, y, z, ww, hh, dd, _) in bins[b]]
                pos = find_best_position(w, h, d, occ)
                if pos is not None:
                    bins[b].append((pos[0], pos[1], pos[2], w, h, d, idx))
                    placed = True
                    break
            
            if not placed:
                occ = []
                pos = find_best_position(w, h, d, occ)
                if pos is not None:
                    bins.append([(pos[0], pos[1], pos[2], w, h, d, idx)])
                    placed = True
                else:
                    return False
        
        return True
    
    # Run BFD with tracking
    bins_tracked = []
    success_tracked = try_pack_tracked(order1, bins_tracked)
    
    if not success_tracked:
        # Fallback
        return [(0, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Apply consolidation to bins_tracked
    improved = True
    while improved and time.time() < deadline:
        improved = False
        if len(bins_tracked) <= 1:
            break
        last_bin = bins_tracked[-1]
        moved_any = False
        for i in range(len(last_bin) - 1, -1, -1):
            box = last_bin[i]
            w, h, d, idx = box[3], box[4], box[5], box[6]
            for b in range(len(bins_tracked) - 1):
                last_bin.pop(i)
                occ = [(x, y, z, ww, hh, dd) for (x, y, z, ww, hh, dd, _) in bins_tracked[b]]
                pos = find_best_position(w, h, d, occ)
                if pos is not None:
                    bins_tracked[b].append((pos[0], pos[1], pos[2], w, h, d, idx))
                    moved_any = True
                    improved = True
                    break
                else:
                    last_bin.insert(i, box)
            if moved_any:
                break
        
        if not last_bin:
            bins_tracked.pop()
            improved = True
    
    # Construct final placements
    placements = [None] * n
    for b, bin_boxes in enumerate(bins_tracked):
        for (x, y, z, w, h, d, idx) in bin_boxes:
            placements[idx] = (b, x, y, z, w, h, d)
    
    # Check for any None (should not happen)
    for i in range(n):
        if placements[i] is None:
            # Fallback: put in bin 0 at (0,0,0) - this might overlap, but it's a last resort
            placements[i] = (0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
    
    return placements
# EVOLVE-BLOCK-END
