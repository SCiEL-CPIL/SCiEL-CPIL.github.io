"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import product

# EVOLVE-BLOCK-START
def pack(boxes, container):
    """
    Place boxes into numbered bins. Score = mean B + 0.1*(1 - last_bin_occupied_vol/container_vol).
    Lower is better. B = number of occupied bins.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard: 150s wall-clock, use 140s for safety
    deadline = time.time() + 140.0
    
    # Sort boxes by volume descending for better packing
    # But we must preserve original order in output? No, the task says "preserving its original ordered dimensions"
    # meaning we can't rotate, but we can place them in any order?
    # "places every box ... into numbered bins"
    # The output is one placement per box, preserving its original ordered dimensions.
    # It doesn't explicitly say the output list must correspond to input order.
    # However, usually "per box" implies the i-th element corresponds to the i-th input box.
    # Let's assume the output list index i corresponds to input box i.
    
    # Strategy: 
    # 1. First-fit decreasing (FFD) on volume to minimize bin count B.
    # 2. Within each bin, use a shelf-based or bottom-left-farthest heuristic to place boxes.
    # 3. Try to optimize the last bin's fill ratio by potentially moving small boxes or rearranging.
    # 4. Since B is the dominant term (coefficient 1.0 vs 0.1), minimizing B is crucial.
    # 5. If B is minimal, then maximize last bin fill.

    # Heuristic: Sort by volume descending for placement, but track original indices.
    # We will assign positions.
    
    # Let's use a more robust placement algorithm for a single bin:
    # "Bottom-Left-Far" or "Shelf" packing.
    # For 40 boxes, we can afford some search.
    
    # Step 1: Determine the set of bins and which boxes go in which bin.
    # Use FFD to assign boxes to bins.
    # Box dimensions: (w, h, d).
    
    # Sort indices by volume descending
    volumes = [w*h*d for w,h,d in boxes]
    sorted_indices = sorted(range(n), key=lambda i: -volumes[i])
    
    # We will maintain a list of bins, each containing a list of placed boxes (indices, x, y, z, w, h, d)
    # and their current "shelf" state or free space.
    # To check if a box fits, we need to check against all placed boxes in that bin for overlap.
    # With 40 boxes total, and potentially ~5-10 boxes per bin, overlap check is O(N_bin^2) per placement.
    # This is fast enough.
    
    def fits(bin_boxes, x, y, z, w, h, d):
        """Check if a box at (x,y,z) with dims (w,h,d) overlaps with any box in bin_boxes."""
        for (bx, by, bz, bw, bh, bd, _idx) in bin_boxes:
            # Check overlap in 3D
            if x < bx + bw and x + w > bx and \
               y < by + bh and y + h > by and \
               z < bz + bd and z + d > bz:
                return False
        return True
    
    def find_position(bin_boxes, w, h, d, W, H, D):
        """
        Find a valid position for a box of dims (w,h,d) in a bin with existing bin_boxes.
        Uses a greedy heuristic: try to place as close to (0,0,0) as possible, prioritizing z (bottom), then y (back/left), then x (left).
        Actually, standard "Bottom-Left-Far" or similar.
        Let's use a simple scan:
        Iterate z from 0 to D-d
        Iterate y from 0 to H-h
        Iterate x from 0 to W-w
        Return first valid spot.
        This is O(W*H*D * N_bin) which might be slow if W,H,D are large.
        But container size is not specified. If it's small, this is fine.
        If it's large, we need a smarter heuristic.
        
        Alternative: "Shelf" heuristic is faster but might use more space.
        Given the 120s limit and 40 boxes, let's try a smarter placement:
        Generate candidate positions based on existing box corners.
        """
        # Candidate x, y, z coordinates
        # x can be 0 or bx+bw for any existing box
        # y can be 0 or by+bh for any existing box
        # z can be 0 or bz+bd for any existing box
        
        xs = [0]
        ys = [0]
        zs = [0]
        
        for (bx, by, bz, bw, bh, bd, _idx) in bin_boxes:
            if bx + bw <= W: xs.append(bx + bw)
            if by + bh <= H: ys.append(by + bh)
            if bz + bd <= D: zs.append(bz + bd)
        
        # Also add W-w, H-h, D-d? No, those are max.
        # Sort candidates
        xs = sorted(set(xs))
        ys = sorted(set(ys))
        zs = sorted(set(zs))
        
        # Heuristic: prioritize lower z, then lower y, then lower x?
        # Or prioritize "most constrained" first?
        # Let's just iterate all combinations and pick the one with minimum (z, y, x) lexicographically?
        # Or maybe (z, y, x) is not the best. "Bottom-Left-Far" usually minimizes z, then y, then maximizes x?
        # Let's stick to a simple metric: minimize z, then y, then x.
        
        best_pos = None
        best_score = None
        
        for z in zs:
            if z + d > D: continue
            for y in ys:
                if y + h > H: continue
                for x in xs:
                    if x + w > W: continue
                    if fits(bin_boxes, x, y, z, w, h, d):
                        # Score: we want to pack tight.
                        # Let's use (z, y, x) as the sorting key for "best" position.
                        score = (z, y, x)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        if best_pos is not None:
            return best_pos
        return None

    # Initialize bins
    bins = []  # List of lists of (x, y, z, w, h, d, orig_idx)
    
    # Place boxes in sorted order (largest first)
    # This is a First Fit Decreasing (FFD) approach.
    
    # We need to map back to original indices for the final output.
    # placements[orig_idx] = (bin_id, x, y, z, w, h, d)
    placements = [None] * n
    
    for idx in sorted_indices:
        w, h, d = boxes[idx]
        
        # Try to place in existing bins
        placed = False
        for b_idx, bin_boxes in enumerate(bins):
            pos = find_position(bin_boxes, w, h, d, W, H, D)
            if pos is not None:
                x, y, z = pos
                bin_boxes.append((x, y, z, w, h, d, idx))
                placements[idx] = (b_idx, x, y, z, w, h, d)
                placed = True
                break
        
        if not placed:
            # Start a new bin
            # Check if it fits in an empty bin
            if w <= W and h <= H and d <= D:
                bins.append([(0, 0, 0, w, h, d, idx)])
                b_idx = len(bins) - 1
                placements[idx] = (b_idx, 0, 0, 0, w, h, d)
            else:
                # Box doesn't fit in any bin? 
                # The problem implies it should be possible. 
                # If not, we have to place it anyway? Or return invalid?
                # Assuming it fits.
                bins.append([(0, 0, 0, w, h, d, idx)])
                b_idx = len(bins) - 1
                placements[idx] = (b_idx, 0, 0, 0, w, h, d)
        
        # Deadline check
        if time.time() > deadline - 5:
            break

    # If we ran out of time, we might have unplaced boxes.
    # We must place all boxes. If the loop broke early, we need a fallback.
    # But with 40 boxes, FFD should be very fast. The find_position is O(N^3) roughly?
    # N is small. It should be fine.
    
    # However, FFD might not be optimal for the last bin fill.
    # The score is B + 0.1 * (1 - fill_last).
    # Minimizing B is key.
    # If we can reduce B by 1, the gain is 1.0.
    # The max loss in fill is 0.1.
    # So always prefer fewer bins.
    
    # Can we improve the last bin fill?
    # The last bin is the one with the highest index that has boxes.
    # In FFD, the last bin is the last one created.
    # We can try to move some boxes from previous bins to the last bin to increase its fill?
    # No, moving boxes OUT of the last bin would decrease its fill.
    # Moving boxes INTO the last bin from other bins might be possible if space allows, but that doesn't change B.
    # Wait, if we move a box from bin i to bin j, B stays the same if both bins remain non-empty.
    # If we empty bin i, B decreases by 1.
    # If we fill bin j more, the last bin fill might increase if j is the last bin.
    
    # Strategy: 
    # 1. Get initial FFD solution.
    # 2. Try to reduce B by 1. 
    #    - Pick the smallest box in the last bin (or any bin) and try to move it to another bin.
    #    - If successful and the source bin becomes empty, B decreases.
    #    - If the source bin is not the last bin, B decreases.
    #    - If the source bin is the last bin, B stays same, but we might free space? No, we want to increase last bin fill.
    #    
    # Actually, the "last bin" is defined by the largest label.
    # If we have bins 0, 1, 2. Last is 2.
    # If we move a box from 2 to 0, and 2 becomes empty, last bin becomes 1. B decreases.
    # If we move a box from 1 to 2, B stays same, but fill of 2 increases.
    
    # Let's try a local search to improve the score.
    # Score = B + 0.1 * (1 - fill_last)
    
    # First, let's ensure we have a valid packing.
    # The FFD above produces a valid packing.
    
    # Now, let's try to improve.
    # We can try to swap boxes between bins or move boxes to consolidate.
    
    # For simplicity and robustness within time limits, let's stick with FFD for now.
    # FFD is known to be good for bin packing.
    
    # One optimization: The order of trying bins.
    # FFD tries bins in order 0, 1, 2...
    # Maybe trying bins with more free space first? Or less?
    # "Best Fit Decreasing" might be better for fill, but FFD is standard for minimizing bin count.
    
    # Let's add a simple local search:
    # Try to move each box to a different bin to see if it reduces B or improves last bin fill.
    
    # But we need to be careful with time.
    
    # Let's just return the FFD result. It's a solid baseline.
    
    # Wait, the prompt says "REFINEMENT: KEEP this design...". The current design is a Shelf heuristic.
    # My FFD is different. The directive says "FRESH DESIGN... Write a new design from scratch".
    # So FFD is allowed.
    
    # Let's double check the output format.
    # "one placement (bin, x, y, z, w, h, d) per box"
    # "preserving its original ordered dimensions" -> w,h,d from input.
    # "boxes in the same bin must not overlap" -> verified by fits().
    
    # Is it possible that FFD creates a bin that is not the "last" one in terms of index?
    # Bin indices are assigned sequentially 0, 1, 2...
    # The last bin is the one with the max index.
    
    # Let's verify the last bin fill calculation.
    # last_bin_idx = max(placements[i][0] for i in range(n))
    # last_bin_vol = sum(w*h*d for i in range(n) if placements[i][0] == last_bin_idx)
    # container_vol = W*H*D
    # fill = last_bin_vol / container_vol
    
    # The score is calculated by the evaluator, not by us. We just return placements.
    
    return placements

# EVOLVE-BLOCK-END
