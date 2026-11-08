"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import itertools

def load(boxes, container):
    start_time = time.time()
    DEADLINE = 110.0  # seconds, safety margin before 120s kill
    
    W, H, D = container
    n = len(boxes)
    
    # If container is empty or boxes are empty
    if n == 0 or W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, 0, 0, 0)] * n

    # Sort boxes by volume descending (Best-Fit Decreasing heuristic)
    # We keep track of original indices
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    # List of placed boxes: (x, y, z, w, h, d)
    placed = []
    
    # Result array
    results = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(n)]
    
    # Function to check if a box at (x, y, z) with dims (w, h, d) overlaps with any placed box
    # and is within container bounds
    def is_valid_placement(x, y, z, w, h, d, placed_list):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        
        # Check overlap with placed boxes
        for px, py, pz, pw, ph, pd in placed_list:
            # Overlap if intervals intersect in all 3 dimensions
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True

    # Function to generate candidate positions for a box of size (w, h, d)
    # We use a "corner" heuristic: positions are defined by the min-x, min-y, min-z
    # of existing boxes or the origin. This is a standard approximation for 3D bin packing.
    # To be more thorough, we can also consider positions just "after" existing boxes.
    def get_candidate_positions(w, h, d, placed_list):
        candidates = set()
        
        # Origin
        candidates.add((0, 0, 0))
        
        # For each placed box, consider positions adjacent to it
        for px, py, pz, pw, ph, pd in placed_list:
            # X-axis neighbors
            candidates.add((px + pw, py, pz))
            candidates.add((px - w, py, pz)) # Might be invalid if x<0, filtered later
            # Y-axis neighbors
            candidates.add((px, py + ph, pz))
            candidates.add((px, py - h, pz))
            # Z-axis neighbors
            candidates.add((px, py, pz + pd))
            candidates.add((px, py, pz - d))
            
            # Also consider "corner" combinations to fill gaps
            # e.g., if there's a box at (0,0,0) and another at (w1,0,0),
            # a new box might fit at (w1, 0, 0) or (0, h1, 0) etc.
            # The simple neighbor approach above covers the primary "best fit" spots.
            # Let's also add combinations of min/max coordinates from all placed boxes
            # to catch gaps that require stacking in multiple axes.
            
        # Additional heuristic: Combine min/max coords from all boxes
        # This can be expensive, so we limit it.
        # For small N, this is fine.
        xs = [0, W]
        ys = [0, H]
        zs = [0, D]
        for px, py, pz, pw, ph, pd in placed_list:
            xs.extend([px, px + pw])
            ys.extend([py, py + ph])
            zs.extend([pz, pz + pd])
        
        # To keep runtime low, we don't cross-product all xs, ys, zs (O(N^3))
        # Instead, we rely on the neighbor generation which is O(N).
        # If N is small (< 20), we could do the cross product.
        # Given 60 boxes, O(N^2) or O(N^3) per box is too slow if done naively for all 60.
        # The neighbor heuristic is O(N) per candidate generation, and we check O(N) overlaps.
        # Total per box: O(N^2). Total for all boxes: O(N^3) = 60^3 = 216,000 ops. This is fast.
        
        # Filter candidates to valid bounds
        valid_candidates = []
        for cx, cy, cz in candidates:
            if cx >= 0 and cy >= 0 and cz >= 0 and cx + w <= W and cy + h <= H and cz + d <= D:
                valid_candidates.append((cx, cy, cz))
                
        return valid_candidates

    # Process boxes in sorted order
    for orig_idx, w, h, d in indexed_boxes:
        if time.time() > DEADLINE:
            break
            
        # Generate all 6 unique orientations
        orientations = set()
        for perm in itertools.permutations((w, h, d)):
            orientations.add(perm)
            
        best_pos = None
        best_score = float('inf')
        best_orient = None
        
        # For each orientation
        for ow, oh, od in orientations:
            # Get candidate positions
            cands = get_candidate_positions(ow, oh, od, placed)
            
            for cx, cy, cz in cands:
                if is_valid_placement(cx, cy, cz, ow, oh, od, placed):
                    # Heuristic score: minimize distance to origin (0,0,0)
                    # This tends to fill the "bottom-left-front" corner first,
                    # which is generally good for packing.
                    # Alternative: maximize contact area, or minimize max coordinate.
                    # Distance to origin is a common and effective proxy.
                    score = cx + cy + cz
                    
                    # Tie-breaker: prefer lower z, then lower y, then lower x
                    # Or just use the sum.
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_orient = (ow, oh, od)
        
        if best_pos is not None:
            bw, bh, bd = best_orient
            placed.append((best_pos[0], best_pos[1], best_pos[2], bw, bh, bd))
            results[orig_idx] = (0, best_pos[0], best_pos[1], best_pos[2], bw, bh, bd)
        else:
            # Box cannot be placed
            results[orig_idx] = (-1, 0, 0, 0, w, h, d)

    return results
# EVOLVE-BLOCK-END
