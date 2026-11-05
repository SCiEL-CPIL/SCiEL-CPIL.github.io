"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard: aim to finish well under 120s (evaluator kills at 150s)
    deadline = time.time() + 110.0
    
    # Pre-compute box properties
    box_vols = [w * h * d for w, h, d in boxes]
    total_vol = sum(box_vols)
    
    # Strategy 1: Best-Fit Decreasing by volume
    # Strategy 2: Sort by max dimension descending
    # Strategy 3: Sort by "aspect ratio" (largest face area)
    
    def score_placement(placement_dict, bin_count):
        """Compute the score: B + 0.1 * (1 - last_bin_occupied / container_vol)"""
        container_vol = W * H * D
        if bin_count == 0:
            return 0.0
        last_bin = max(placement_dict.keys())
        # Sum volumes of boxes in the last bin
        last_bin_vol = 0
        for b_idx in placement_dict.values():
            if b_idx == last_bin:
                # We need to know which box indices are in this bin
                pass
        # Actually, let's compute this more efficiently outside
        return bin_count  # placeholder, will compute properly below
    
    def greedy_pack(sort_key_func, boxes_list):
        """
        Greedy packing using a shelf-like approach but with best-fit placement.
        sort_key_func: function to sort boxes
        boxes_list: list of (original_index, w, h, d)
        Returns: dict mapping original_index -> (bin_id, x, y, z, w, h, d)
        """
        sorted_indices = sorted(range(len(boxes_list)), key=lambda i: sort_key_func(boxes_list[i]))
        
        # For each bin, maintain a list of placed boxes for overlap checking
        # To speed up, we'll use a simple approach: for each new box, try to place in existing bins
        # If it fits, place it; else open a new bin.
        
        bin_placements = {}  # bin_id -> list of (x, y, z, w, h, d)
        bin_volumes = {}  # bin_id -> total volume placed
        next_bin = 0
        result = {}
        
        for idx in sorted_indices:
            orig_idx, w, h, d = boxes_list[idx]
            
            best_bin = -1
            best_pos = None
            best_waste = float('inf')
            
            # Try existing bins first (best-fit: minimize wasted space or maximize fill)
            # For simplicity and speed, we'll try to place in the bin with the most volume already used
            # that can still fit the box, preferring the one where the box fits "best"
            
            candidate_bins = []
            for b_id in bin_placements:
                candidate_bins.append(b_id)
            
            # Sort candidate bins by current volume descending (prefer fuller bins)
            candidate_bins.sort(key=lambda b: bin_volumes.get(b, 0), reverse=True)
            
            for b_id in candidate_bins:
                pos = find_position(boxes_list[idx][1], boxes_list[idx][2], boxes_list[idx][3], bin_placements[b_id], W, H, D)
                if pos is not None:
                    # Calculate "waste" - how much empty space this placement leaves in the bin
                    # Simple heuristic: prefer positions closer to origin (more compact)
                    x, y, z = pos
                    waste = x + y + z  # simple proxy for compactness
                    if waste < best_waste:
                        best_waste = waste
                        best_bin = b_id
                        best_pos = pos
            
            if best_bin == -1:
                # Open a new bin
                best_bin = next_bin
                next_bin += 1
                bin_placements[best_bin] = []
                bin_volumes[best_bin] = 0
                best_pos = (0, 0, 0)
            
            x, y, z = best_pos
            bin_placements[best_bin].append((x, y, z, w, h, d))
            bin_volumes[best_bin] = bin_volumes.get(best_bin, 0) + w * h * d
            result[orig_idx] = (best_bin, x, y, z, w, h, d)
        
        return result, next_bin
    
    def find_position(w, h, d, placed_boxes, W, H, D):
        """
        Find a valid position (x, y, z) for a box of size (w, h, d) in a bin.
        Uses a simplified best-fit: try to find the position that minimizes the
        maximum coordinate (most compact).
        
        For speed, we'll use a grid-based approach or a few heuristic positions.
        """
        # Generate candidate positions
        # Simple approach: try to place at the "bottom-left-front" corner of available space
        # This is a simplification; a full search would be expensive.
        
        # For now, use a simple shelf-based placement within the bin
        # Track the "frontier" of the bin
        # This is a heuristic: find the lowest z where the box fits in a row
        
        # Simplified: try to place in a shelf structure
        # Shelves are along y, rows along x, layers along z
        
        # For speed, we'll just try a few candidate positions
        # 1. Origin
        # 2. Along x-axis
        # 3. Along y-axis
        # 4. Along z-axis
        # 5. Combinations
        
        # Actually, for correctness and reasonable performance, let's use a proper
        # but efficient placement check.
        
        # Check if a position (x, y, z) is valid
        def is_valid(x, y, z, w, h, d, placed):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            for (px, py, pz, pw, ph, pd) in placed:
                # Check overlap
                if not (x + w <= px or px + pw <= x):  # no overlap in x
                    if not (y + h <= py or py + ph <= y):  # no overlap in y
                        if not (z + d <= pz or pz + pd <= z):  # no overlap in z
                            return False
            return True
        
        # Try to find the best position using a few heuristics
        # 1. Try to place as low as possible (minimize z)
        # 2. Then as far left as possible (minimize x)
        # 3. Then as far front as possible (minimize y)
        
        best_pos = None
        best_key = None
        
        # Sample candidate positions
        # For efficiency, we'll check a limited set of positions
        # Based on the existing placed boxes, the interesting x, y, z coordinates are
        # 0 and the edges of existing boxes
        
        x_candidates = [0]
        y_candidates = [0]
        z_candidates = [0]
        
        for (px, py, pz, pw, ph, pd) in placed:
            x_candidates.append(px + pw)
            y_candidates.append(py + ph)
            z_candidates.append(pz + pd)
        
        # Deduplicate and sort
        x_candidates = sorted(set(x_candidates))
        y_candidates = sorted(set(y_candidates))
        z_candidates = sorted(set(z_candidates))
        
        # Limit the number of candidates for speed
        # If there are too many, sample
        if len(x_candidates) > 20:
            x_candidates = x_candidates[:10] + x_candidates[-10:]
        if len(y_candidates) > 20:
            y_candidates = y_candidates[:10] + y_candidates[-10:]
        if len(z_candidates) > 20:
            z_candidates = z_candidates[:10] + z_candidates[-10:]
        
        for z in z_candidates:
            if z + d > D:
                continue
            for y in y_candidates:
                if y + h > H:
                    continue
                for x in x_candidates:
                    if x + w > W:
                        continue
                    if is_valid(x, y, z, w, h, d, placed):
                        key = (z, x, y)  # prioritize low z, then low x, then low y
                        if best_key is None or key < best_key:
                            best_key = key
                            best_pos = (x, y, z)
        
        return best_pos
    
    # Prepare boxes_list for each strategy
    # boxes_list[i] = (original_index, w, h, d)
    boxes_list = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    
    # Strategy 1: Sort by volume descending
    def sort_by_volume(box):
        return -(box[1] * box[2] * box[3])
    
    # Strategy 2: Sort by max dimension descending
    def sort_by_max_dim(box):
        return -max(box[1], box[2], box[3])
    
    # Strategy 3: Sort by "surface area" of largest face
    def sort_by_face_area(box):
        w, h, d = box[1], box[2], box[3]
        return -(max(w*h, w*d, h*d))
    
    strategies = [sort_by_volume, sort_by_max_dim, sort_by_face_area]
    
    best_score = float('inf')
    best_result = None
    
    for sort_func in strategies:
        if time.time() > deadline - 5:
            break
        
        result, bin_count = greedy_pack(sort_func, boxes_list)
        
        # Compute score
        container_vol = W * H * D
        if bin_count == 0:
            score = 0.0
        else:
            last_bin = max(result.values(), key=lambda x: x[0])[0]
            last_bin_vol = 0
            for orig_idx, (b, x, y, z, w, h, d) in result.items():
                if b == last_bin:
                    last_bin_vol += w * h * d
            score = bin_count + 0.1 * (1 - last_bin_vol / container_vol)
        
        if score < best_score:
            best_score = score
            best_result = result
    
    # If no result, fall back to simple shelf
    if best_result is None:
        # Fallback: simple shelf packing
        placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for i, (w, h, d) in enumerate(boxes):
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return placements
    
    # Convert result dict to list in original order
    final_placements = []
    for i in range(n):
        b, x, y, z, w, h, d = best_result[i]
        final_placements.append((b, x, y, z, w, h, d))
    
    return final_placements
# EVOLVE-BLOCK-END
