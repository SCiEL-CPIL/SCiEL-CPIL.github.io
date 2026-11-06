"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    W, H, D = container
    n_boxes = len(boxes)
    
    # 1. Sort boxes by volume descending (largest first)
    # We keep the original index to restore the order in the final output if needed, 
    # but the problem asks for placements per box. 
    # "preserving its original ordered dimensions" means w,h,d cannot be rotated.
    # It does NOT mean the output list must be in original box order, 
    # but typically "placements" is a list corresponding to the input boxes.
    # Let's assume the output list index i corresponds to input box i.
    
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    # Bins state:
    # Each bin is a list of placed boxes: [(x, y, z, w, h, d), ...]
    # We also maintain a list of "corner points" for each bin to speed up placement.
    # A corner point is (x, y, z).
    # Initial corner point for an empty bin is (0, 0, 0).
    
    bins = []  # List of lists of placed boxes
    corners = [] # List of lists of corner points (x, y, z)
    
    def can_place(bin_idx, x, y, z, w, h, d):
        """Check if a box fits at (x,y,z) in bin_idx without overlapping."""
        if x + w > W or y + h > H or z + d > D:
            return False
        
        placed = bins[bin_idx]
        for (px, py, pz, pw, ph, pd) in placed:
            # Check overlap
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True

    def find_best_spot(bin_idx, w, h, d):
        """
        Find the best corner point in bin_idx to place the box.
        Returns (x, y, z) or None if no spot found.
        Heuristic: Minimize a score based on distance from (0,0,0) and 'tightness'.
        A common heuristic is to minimize x*W + y*H + z*D or similar weighted distance.
        Another is to minimize the 'wasted space' around the box, but distance is faster.
        Let's use a weighted distance score: lower is better.
        """
        best_score = float('inf')
        best_pos = None
        
        # We should check corners that are "reachable" and haven't been dominated.
        # For simplicity and speed with 40 boxes, we check all current corners.
        # To optimize, we can skip corners that are 'behind' already placed boxes, 
        # but with small N, checking all is fine.
        
        for (cx, cy, cz) in corners[bin_idx]:
            if can_place(bin_idx, cx, cy, cz, w, h, d):
                # Score: prefer positions closer to origin (0,0,0) to fill from bottom-left-front
                # Weights can be normalized or just raw distances.
                # Using raw coordinates:
                score = cx * 1.0 + cy * 1.0 + cz * 1.0 
                # Alternatively, a more sophisticated score might consider the 'gap'
                # but let's stick to simple first-fit decreasing on corners sorted by score.
                
                if score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
        
        return best_pos

    def add_corners_for(box, bin_idx, x, y, z):
        """
        After placing a box at (x,y,z) with dims (w,h,d), add new corner points.
        New corners are generated at the 'front' of the placed box.
        (x+w, y, z), (x, y+h, z), (x, y, z+d)
        """
        new_corners = [
            (x + box[1], y, z),
            (x, y + box[2], z),
            (x, y, z + box[3])
        ]
        
        # Add to corners list, avoiding duplicates if possible (or just add all, 
        # duplicates just mean redundant checks later which is okay for small N)
        # To keep list small, we can check if the point is already in the list.
        for nc in new_corners:
            if nc not in corners[bin_idx]:
                corners[bin_idx].append(nc)

    # Main Loop
    placements = [None] * n_boxes
    
    for orig_idx, w, h, d, vol in indexed_boxes:
        if time.time() > deadline:
            # Emergency fallback: place remaining boxes in a simple shelf manner 
            # in a new bin or just fail? 
            # Better: If we run out of time, we should try to finish quickly.
            # For 40 boxes, this shouldn't happen. If it does, we return what we have?
            # No, we must return a valid placement for ALL boxes.
            # So we must ensure the loop is fast enough. 
            # If we hit deadline, we can't return invalid. 
            # We will assume the algorithm is fast enough. 
            # If not, we could switch to a very fast heuristic for the rest.
            pass
            
        placed = False
        # Try to place in existing bins
        # Sort bins by current volume or just iterate? 
        # Iterating in order is fine.
        
        for b_idx in range(len(bins)):
            pos = find_best_spot(b_idx, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Place the box
                bins[b_idx].append((x, y, z, w, h, d))
                add_corners_for((w, h, d), b_idx, x, y, z)
                placements[orig_idx] = (b_idx, x, y, z, w, h, d)
                placed = True
                break
        
        if not placed:
            # Open a new bin
            b_idx = len(bins)
            bins.append([])
            corners.append([(0, 0, 0)])
            
            # Place at (0,0,0)
            x, y, z = 0, 0, 0
            bins[b_idx].append((x, y, z, w, h, d))
            add_corners_for((w, h, d), b_idx, x, y, z)
            placements[orig_idx] = (b_idx, x, y, z, w, h, d)
            
    return placements
# EVOLVE-BLOCK-END
