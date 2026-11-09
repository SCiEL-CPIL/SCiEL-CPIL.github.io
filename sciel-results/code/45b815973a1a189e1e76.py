"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def pack(boxes, container):
    """
    Pack boxes into numbered bins using a Best-Fit Decreasing Height (BFDH) variant.
    Returns a list of (bin, x, y, z, w, h, d) tuples.
    """
    W, H, D = container
    if not boxes:
        return []
    
    # Sort boxes by height (h) descending, then by depth (d) descending, then width (w) descending.
    # This is a heuristic: taller boxes first often leads to better shelf packing.
    indexed_boxes = sorted(enumerate(boxes), key=lambda x: (-x[1][1], -x[1][2], -x[1][0]))
    
    # Bins will be a list of lists of placed boxes (x, y, z, w, h, d)
    bins = []
    
    def can_place(bin_placements, x, y, z, w, h, d):
        # Check if box at (x,y,z) with dims (w,h,d) fits in the bin without overlapping existing placements
        # and stays within container bounds
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_placements:
            # Check for overlap
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True

    def find_position(bin_placements, w, h, d):
        # Try to find the best (lowest y, then lowest z, then lowest x) position for the box
        # We scan through potential y (shelves) and z (layers) and x (columns)
        # To be efficient, we can track the "skyline" or just scan grid points.
        # Given small N (40 boxes), a simple scan is feasible.
        
        best_pos = None
        best_score = float('inf')
        
        # Generate candidate y positions: 0 and top of each existing box
        # Generate candidate z positions: 0 and top of each existing box
        # Generate candidate x positions: 0 and right of each existing box
        
        y_candidates = set([0])
        z_candidates = set([0])
        x_candidates = set([0])
        
        for (px, py, pz, pw, ph, pd) in bin_placements:
            y_candidates.add(py + ph)
            z_candidates.add(pz + pd)
            x_candidates.add(px + pw)
        
        # Sort candidates to prefer lower values
        y_candidates = sorted(y_candidates)
        z_candidates = sorted(z_candidates)
        x_candidates = sorted(x_candidates)
        
        for y in y_candidates:
            if y + h > H:
                break
            for z in z_candidates:
                if z + d > D:
                    break
                for x in x_candidates:
                    if x + w > W:
                        break
                    if can_place(bin_placements, x, y, z, w, h, d):
                        # Score: minimize y, then z, then x
                        score = (y, z, x)
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos

    # Process boxes in sorted order
    for idx, (w, h, d) in indexed_boxes:
        placed = False
        # Try to place in existing bins
        for i, bin_placements in enumerate(bins):
            pos = find_position(bin_placements, w, h, d)
            if pos is not None:
                x, y, z = pos
                bin_placements.append((x, y, z, w, h, d))
                placed = True
                break
        
        if not placed:
            # Start a new bin
            new_bin = [(0, 0, 0, w, h, d)]
            bins.append(new_bin)
    
    # Construct the result
    placements = []
    for bin_idx, bin_placements in enumerate(bins):
        for (x, y, z, w, h, d) in bin_placements:
            placements.append((bin_idx, x, y, z, w, h, d))
    
    # The problem requires preserving the original order of boxes in the output?
    # "returns one placement (bin, x, y, z, w, h, d) per box"
    # Usually, the i-th element in the returned list corresponds to the i-th box in the input.
    # Let's check the seed program: it appends in the order of the input loop.
    # So we need to map back to the original indices.
    
    # Re-initialize to store placements by original index
    final_placements = [None] * len(boxes)
    
    # We need to re-run the packing but keep track of which original index went where.
    # Let's refactor the packing logic to track indices.
    
    # Reset bins
    bins = []
    
    for idx, (w, h, d) in indexed_boxes:
        placed = False
        for i, bin_placements in enumerate(bins):
            pos = find_position(bin_placements, w, h, d)
            if pos is not None:
                x, y, z = pos
                bin_placements.append((x, y, z, w, h, d, idx))
                placed = True
                break
        
        if not placed:
            new_bin = [(0, 0, 0, w, h, d, idx)]
            bins.append(new_bin)
    
    for bin_idx, bin_placements in enumerate(bins):
        for (x, y, z, w, h, d, idx) in bin_placements:
            final_placements[idx] = (bin_idx, x, y, z, w, h, d)
            
    return final_placements

# EVOLVE-BLOCK-END
