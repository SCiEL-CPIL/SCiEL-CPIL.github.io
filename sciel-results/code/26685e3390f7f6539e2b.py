"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115  # 115s limit to be safe within 120s
    
    # Sort boxes by volume descending (Best-Fit-Decreasing strategy)
    # We keep original index to map back to placements
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1], reverse=True)
    
    bins = []  # List of bins, each bin is a list of (x, y, z, w, h, d)
    placements = [None] * n
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        # Check bounds
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check overlap with existing boxes in this bin
        for bx, by, bz, bw, bh, bd in bin_boxes:
            # Check if intervals overlap in all 3 dimensions
            if x < bx + bw and bx < x + w and \
               y < by + bh and by < y + h and \
               z < bz + bd and bz < z + d:
                return False
        return True

    def find_spot(bin_boxes, w, h, d):
        # Heuristic: Try to place at the "lowest" possible z, then lowest y, then lowest x.
        # To keep it fast, we don't do a full grid search, but check against existing box "footprints" and corners.
        # A simple effective heuristic is to try placing against existing boxes' surfaces and the floor/walls.
        
        # Generate candidate positions:
        # 1. Origin (0,0,0)
        # 2. On top of any existing box (x, y, z + bd)
        # 3. Beside any existing box in x (bx + bw, y, z)
        # 4. Beside any existing box in y (x, by + bh, z)
        # 5. Combinations of these offsets are complex, so we stick to simple offsets from existing boxes.
        
        candidates = [(0, 0, 0)]
        
        for bx, by, bz, bw, bh, bd in bin_boxes:
            # Try placing on top
            candidates.append((bx, by, bz + bd))
            # Try placing next to in X
            candidates.append((bx + bw, by, bz))
            # Try placing next to in Y
            candidates.append((bx, by + bh, bz))
            
            # Also try combinations of "corner" placements relative to existing boxes
            # This helps fill gaps
            candidates.append((bx + bw, by + bh, bz))
            candidates.append((bx + bw, by, bz + bd))
            candidates.append((bx, by + bh, bz + bd))
            
        # Sort candidates to prefer lower Z, then lower Y, then lower X (compactness)
        # We want to minimize (z, y, x)
        candidates.sort(key=lambda pos: (pos[2], pos[1], pos[0]))
        
        for cx, cy, cz in candidates:
            if can_place(bin_boxes, cx, cy, cz, w, h, d):
                return (cx, cy, cz)
        
        return None

    # Place each box
    for i, vol, w, h, d in indexed_boxes:
        if time.time() > deadline:
            # If we run out of time, place remaining boxes in a new bin at 0,0,0 if possible, 
            # or just break and return what we have? 
            # The task requires placing EVERY box. 
            # If we are about to timeout, we must ensure validity.
            # A safe fallback: force place in new bins if we can't find spots quickly.
            # But finding spots is the slow part. 
            # Let's just proceed with a simpler check if time is critical, 
            # or better: if we are close to deadline, use a very fast heuristic (just new bin if first fit fails quickly).
            pass

        best_bin_idx = -1
        best_pos = None
        best_metric = float('inf')
        
        # Try existing bins
        for b_idx, bin_boxes in enumerate(bins):
            # Quick check: if bin is "full" in terms of volume, maybe skip? 
            # No, volume doesn't guarantee fit.
            
            # Optimization: Check if we can place in this bin.
            # To speed up, we can check a few candidate spots.
            spot = find_spot(bin_boxes, w, h, d)
            if spot is not None:
                # Metric for Best-Fit: Prefer the spot that leaves the least space?
                # Or prefer the bin where the box fits "tightest"?
                # Common BFD metric: minimize the "gap" or simply pick the first valid bin in order of "fullness".
                # Let's use a simple metric: prefer bins with more volume already used (fill them up).
                # Or just pick the first valid bin (First-Fit-Decreasing).
                # BFD usually picks the bin where the box fits in the "smallest" remaining space.
                # Approximation: Pick the bin where the box is placed at the highest Z? No, lowest Z is better for stability/compaction.
                # Let's stick to First-Fit-Decreasing (FFD) which is simpler and often competitive, 
                # or Best-Fit by "minimum waste".
                
                # Let's use a heuristic score: prefer bins that are "more full" (higher total volume)
                # to minimize the number of bins opened.
                bin_vol = sum(bw*bh*bd for _,_,_,bw,bh,bd in bin_boxes)
                # Lower score is better? 
                # If we want to fill bins, we want to put items in bins that are already big.
                # So we want to MAXIMIZE bin_vol.
                
                # Actually, standard BFD: fit in the bin with the smallest remaining capacity that can still fit the item.
                # Remaining capacity is hard to compute exactly.
                # Let's just use FFD: first bin that fits.
                # This is fast and robust.
                
                best_bin_idx = b_idx
                best_pos = spot
                break # First Fit
        
        if best_bin_idx == -1:
            # No existing bin fits, open a new one
            best_bin_idx = len(bins)
            bins.append([])
            best_pos = (0, 0, 0)
            
        # Place the box
        bx, by, bz = best_pos
        bins[best_bin_idx].append((bx, by, bz, w, h, d))
        placements[i] = (best_bin_idx, bx, by, bz, w, h, d)
        
        # Time check for safety
        if time.time() > deadline:
             # If we are still in the loop and close to deadline, 
             # we must ensure the rest are placed. 
             # Since FFD is O(N^2 * K) where K is boxes per bin, it's usually fast enough for N=40.
             # 40 boxes is very small. This will finish in milliseconds.
             pass

    return placements
# EVOLVE-BLOCK-END
