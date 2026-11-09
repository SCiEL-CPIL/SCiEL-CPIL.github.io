"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def pack(boxes, container):
    """
    Pack boxes into bins using a 3D grid-based occupancy map approach.
    For each box, we try to place it in the current bin at the position that
    minimizes the sum of distances to the origin (x, y, z), i.e., the "lowest"
    and "leftmost" and "deepest" position. We use a greedy best-fit strategy
    with a 3D boolean occupancy grid per bin.
    
    To keep runtime within limits, we limit the search space: for each candidate
    bin, we check all possible (x, y, z) positions where the box fits.
    We prefer earlier bins (lower bin index) to minimize the number of bins used.
    """
    W, H, D = container
    n_boxes = len(boxes)
    
    # Pre-sort boxes by volume descending? No, the task says "preserving its original ordered dimensions"
    # but it doesn't say we can't reorder which box goes where. Wait, "places every box... preserving its 
    # original ordered dimensions (no rotations)" means we cannot rotate a box, but we CAN choose the order
    # in which we place them? The interface says "one placement per box", implying we must assign a placement
    # to each box. The order of boxes in the input list is fixed, but we can place them in any bin.
    # Actually, standard bin packing allows choosing the order. However, the signature is pack(boxes, container).
    # Let's assume we can reorder the boxes for placement to improve packing.
    # But wait, the return is a list of placements. Does the order of the list matter?
    # "one placement (bin, x, y, z, w, h, d) per box" - usually this implies the i-th placement corresponds
    # to the i-th box in the input. Let's assume the output list index i corresponds to boxes[i].
    
    # If we must preserve the order of boxes in the output list, we still can choose which bin to put them in.
    # Let's try a First-Fit Decreasing Height (FFDH) or similar heuristic, but with a grid check.
    # Since we can't rotate, we just use the given w, h, d.
    
    # Strategy:
    # 1. Create a list of bins, each with a 3D boolean array of shape (W, H, D) or (W, D, H) depending on numpy indexing.
    #    Let's use (x, y, z) -> (W, H, D).
    # 2. For each box in the given order (to respect the interface if order matters), find the best position.
    #    "Best" defined as:
    #      a. Lowest bin index (to minimize B).
    #      b. Within that bin, the position (x, y, z) that minimizes x + y + z (or some other metric like max(x,y,z)).
    #         Minimizing x+y+z tends to pack towards the corner.
    #    
    #    To make it faster, we can iterate through bins. For the first bin that can fit the box, we find the best position in that bin.
    #    We don't need to check all bins if the first one fits, unless we want to optimize further.
    #    Let's try: For each box, check bin 0, 1, 2... in order. For the first bin where the box fits, find the best (x,y,z) and place it.
    
    # This is O(N * B * W * H * D) in the worst case if we scan all positions.
    # N=40, B~5, W,H,D maybe 100? 40*5*1e6 = 2e8 which might be too slow in Python loops, but vectorized with NumPy it's okay.
    # However, creating a W*H*D array for each bin might be memory heavy if dimensions are large.
    # Let's assume dimensions are reasonable (e.g., < 100).
    
    # Optimization: Instead of checking all (x,y,z) for every box in every bin, we can maintain a list of "free" spaces or use a simpler heuristic.
    # Given the 120s limit, a full grid scan for 40 boxes should be fine if vectorized.
    
    bins = []  # List of numpy arrays, each of shape (W, H, D)
    
    def get_bin(i):
        while len(bins) <= i:
            bins.append(np.zeros((W, H, D), dtype=bool))
        return bins[i]
    
    # For each box, find the best placement
    placements = [None] * n_boxes
    
    for i, (w, h, d) in enumerate(boxes):
        best_bin_idx = None
        best_pos = None
        best_score = float('inf')
        
        # Check existing bins and potentially a new bin
        max_bin_to_check = len(bins)
        
        for b_idx in range(max_bin_to_check):
            occ = bins[b_idx]
            # Find all positions (x, y, z) where the box fits
            # A box of size (w, h, d) fits at (x, y, z) if:
            # x + w <= W, y + h <= H, z + d <= D
            # and occ[x:x+w, y:y+h, z:z+d] is all False
            
            # We can vectorize this check.
            # Create a mask of valid starting positions?
            # Or simply iterate? No, vectorize.
            
            # The condition "occ[x:x+w, y:y+h, z:z+d] is all False" is equivalent to
            # the sum of that slice being 0.
            
            # We can compute a "valid" array of shape (W-w+1, H-h+1, D-d+1)
            # where valid[x, y, z] = True if the box fits at (x, y, z).
            
            if w > W or h > H or d > D:
                continue # Box doesn't fit in any bin, but problem guarantees it does?
                
            # Compute valid positions
            # We can use a sliding window sum or just simple slicing logic if dimensions are small.
            # A simpler vectorized way:
            # For each candidate x in [0, W-w], y in [0, H-h], z in [0, D-d]:
            # check if np.any(occ[x:x+w, y:y+h, z:z+d])
            
            # This is expensive to do in a pure Python loop.
            # Let's use a trick:
            # If we sum occ along the last axis, then the last axis, etc., we get a 3D array of sums.
            # Specifically, let S be the 3D sum of occ over windows of size (w, h, d).
            # We can compute this using convolution or cumulative sums.
            
            # Using cumulative sums (prefix sums) is efficient.
            # Let P be the prefix sum of occ.
            # P[x, y, z] = sum(occ[0:x, 0:y, 0:z])
            # Then sum(occ[x:x+w, y:y+h, z:z+d]) = P[x+w, y+h, z+d] - P[x, y+h, z+d] - P[x+w, y, z+d] + P[x, y, z+d] - ... (inclusion-exclusion)
            
            # Let's compute the prefix sum of occ.
            # occ is (W, H, D)
            # P is (W+1, H+1, D+1)
            P = np.zeros((W+1, H+1, D+1), dtype=np.int32)
            P[1:, 1:, 1:] = occ
            # Cumulative sum along axis 2 (z)
            P = P.cumsum(axis=2)
            # Cumulative sum along axis 1 (y)
            P = P.cumsum(axis=1)
            # Cumulative sum along axis 0 (x)
            P = P.cumsum(axis=0)
            
            # Now, for a box at (x, y, z) with size (w, h, d):
            # Sum = P[x+w, y+h, z+d] - P[x, y+h, z+d] - P[x+w, y, z+d] + P[x, y, z+d]
            #      - P[x+w, y+h, z] + P[x, y+h, z] + P[x+w, y, z] - P[x, y, z]
            
            # We want to find (x, y, z) such that Sum == 0.
            # x in [0, W-w], y in [0, H-h], z in [0, D-d]
            
            x_max = W - w
            y_max = H - h
            z_max = D - d
            
            if x_max < 0 or y_max < 0 or z_max < 0:
                continue
                
            # Create arrays of indices
            xs = np.arange(x_max + 1)
            ys = np.arange(y_max + 1)
            zs = np.arange(z_max + 1)
            
            # This might create large arrays. Let's try to be smart.
            # Instead of checking all, let's just check if there's ANY valid position.
            # We can compute the sum array for all valid positions.
            
            # Sums[x, y, z] = P[x+w, y+h, z+d] - P[x, y+h, z+d] - P[x+w, y, z+d] + P[x, y, z+d]
            #                - P[x+w, y+h, z] + P[x, y+h, z] + P[x+w, y, z] - P[x, y, z]
            
            # Let's extract the relevant slices of P.
            # P is (W+1, H+1, D+1)
            
            # Term 1: P[x+w, y+h, z+d] for x in [0, x_max], y in [0, y_max], z in [0, z_max]
            # This is P[w:w+x_max+1, h:h+y_max+1, d:d+z_max+1]
            # Note: x+w ranges from w to w+x_max = W. So P[w:W+1, ...]
            
            t1 = P[w:w+x_max+1, h:h+y_max+1, d:d+z_max+1]
            t2 = P[0:x_max+1, h:h+y_max+1, d:d+z_max+1]
            t3 = P[w:w+x_max+1, 0:y_max+1, d:d+z_max+1]
            t4 = P[0:x_max+1, 0:y_max+1, d:d+z_max+1]
            t5 = P[w:w+x_max+1, h:h+y_max+1, 0:z_max+1]
            t6 = P[0:x_max+1, h:h+y_max+1, 0:z_max+1]
            t7 = P[w:w+x_max+1, 0:y_max+1, 0:z_max+1]
            t8 = P[0:x_max+1, 0:y_max+1, 0:z_max+1]
            
            sums = t1 - t2 - t3 + t4 - t5 + t6 + t7 - t8
            
            valid_mask = (sums == 0)
            
            if np.any(valid_mask):
                # Find the position that minimizes x + y + z (or some other metric)
                # We want to pack towards (0,0,0).
                # Let's minimize x + y + z.
                # Create a score array for valid positions.
                # score = x + y + z
                # We need to map valid_mask indices to x, y, z.
                
                # Get coordinates of valid positions
                valid_coords = np.argwhere(valid_mask) # (x, y, z)
                
                if len(valid_coords) > 0:
                    # Calculate score for each valid position
                    # score = x + y + z
                    scores = valid_coords[:, 0] + valid_coords[:, 1] + valid_coords[:, 2]
                    min_score_idx = np.argmin(scores)
                    min_score = scores[min_score_idx]
                    
                    # Compare with best so far
                    # We prioritize lower bin index first.
                    # So if we are in bin b_idx, and we haven't found a better bin (lower index), this is a candidate.
                    # Since we iterate b_idx from 0 upwards, the first bin that has a valid position is the best bin index.
                    # So we can break after the first bin that has a valid position.
                    
                    best_bin_idx = b_idx
                    best_pos = valid_coords[min_score_idx]
                    best_score = min_score
                    break # Found a valid position in the lowest possible bin index
        
        # If no existing bin worked, create a new bin
        if best_bin_idx is None:
            best_bin_idx = len(bins)
            get_bin(best_bin_idx)
            # Place at (0, 0, 0)
            best_pos = np.array([0, 0, 0])
        
        x, y, z = best_pos
        b_idx = best_bin_idx
        
        # Update the occupancy grid
        bins[b_idx][x:x+w, y:y+h, z:z+d] = True
        
        placements[i] = (b_idx, x, y, z, w, h, d)
        
    return placements
# EVOLVE-BLOCK-END
