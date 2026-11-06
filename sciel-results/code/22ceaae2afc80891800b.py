import time
import itertools

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []
    
    deadline = time.time() + 110.0  # safety margin
    
    # Generate all 6 unique rotations for each box (dedup if symmetric)
    def get_rotations(w, h, d):
        seen = set()
        rots = []
        for perm in itertools.permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                rots.append(perm)
        return rots
    
    box_rots = []
    for i, (w, h, d) in enumerate(boxes):
        box_rots.append(get_rotations(w, h, d))
    
    # Sorting strategies
    def sort_volume_desc(idx_list):
        return sorted(idx_list, key=lambda i: -(boxes[i][0]*boxes[i][1]*boxes[i][2]))
    
    def sort_max_dim_desc(idx_list):
        return sorted(idx_list, key=lambda i: -max(boxes[i]))
    
    def sort_min_dim_asc(idx_list):
        return sorted(idx_list, key=lambda i: min(boxes[i]))
    
    def sort_ratio_desc(idx_list):
        # volume / max_dim^2, prefer "cube-like" or flat efficient
        return sorted(idx_list, key=lambda i: -(boxes[i][0]*boxes[i][1]*boxes[i][2]) / max(1, max(boxes[i])**2))
    
    strategies = [sort_volume_desc, sort_max_dim_desc, sort_min_dim_asc, sort_ratio_desc]
    
    def try_place(box_idx, rots, container_dims, placed):
        """Try to place box with given rotations in container. Returns (x,y,z,w,h,d) or None."""
        cW, cH, cD = container_dims
        for rw, rh, rd in rots:
            # Try all positions: this is expensive, so we use a smarter approach
            # For now, try a grid of candidate positions based on existing placements
            # But for simplicity and speed, let's try the "skyline" or "shelf" approach in 3D
            pass
        return None
    
    # A more practical approach: use a 3D shelf/skyline algorithm
    # We'll maintain a list of "shelves" or use a point-based free space check
    
    def place_box_3d(box_dims, rots, container_dims, placed_rects):
        """
        Try to place a box in the container given existing placed rectangles.
        Uses a candidate position approach: try positions at (0,0,0), and at the
        edges of existing boxes.
        Returns (x, y, z, w, h, d) or None.
        """
        cW, cH, cD = container_dims
        
        # Generate candidate x, y, z positions
        x_candidates = [0]
        y_candidates = [0]
        z_candidates = [0]
        
        for (px, py, pz, pw, ph, pd) in placed_rects:
            if px + pw <= cW:
                x_candidates.append(px + pw)
            if py + ph <= cH:
                y_candidates.append(py + ph)
            if pz + pd <= cD:
                z_candidates.append(pz + pd)
        
        # Remove duplicates and sort
        x_candidates = sorted(set(x for x in x_candidates if x <= cW))
        y_candidates = sorted(set(y for y in y_candidates if y <= cH))
        z_candidates = sorted(set(z for z in z_candidates if z <= cD))
        
        # Limit candidates to keep it fast
        if len(x_candidates) > 30:
            x_candidates = x_candidates[:30] + [cW - max(boxes[0][0], 1)]
            x_candidates = sorted(set(x for x in x_candidates if 0 <= x <= cW))
        if len(y_candidates) > 30:
            y_candidates = y_candidates[:30] + [cH - max(boxes[0][1], 1)]
            y_candidates = sorted(set(y for y in y_candidates if 0 <= y <= cH))
        if len(z_candidates) > 30:
            z_candidates = z_candidates[:30] + [cD - max(boxes[0][2], 1)]
            z_candidates = sorted(set(z for z in z_candidates if 0 <= z <= cD))
        
        for rw, rh, rd in rots:
            for x in x_candidates:
                if x + rw > cW:
                    continue
                for y in y_candidates:
                    if y + rh > cH:
                        continue
                    for z in z_candidates:
                        if z + rd > cD:
                            continue
                        # Check overlap with all placed boxes
                        overlap = False
                        for (px, py, pz, pw, ph, pd) in placed_rects:
                            if (x < px + pw and x + rw > px and
                                y < py + ph and y + rh > py and
                                z < pz + pd and z + rd > pz):
                                overlap = True
                                break
                        if not overlap:
                            return (x, y, z, rw, rh, rd)
        return None
    
    def run_strategy(sort_func):
        """Run a packing strategy and return placements."""
        placed_rects = []
        result = [None] * n
        order = sort_func(list(range(n)))
        
        for i in order:
            if time.time() > deadline:
                break
            rots = box_rots[i]
            pos = place_box_3d(boxes[i], rots, (W, H, D), placed_rects)
            if pos is not None:
                x, y, z, rw, rh, rd = pos
                placed_rects.append((x, y, z, rw, rh, rd))
                result[i] = (0, x, y, z, rw, rh, rd)
            else:
                result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
        
        # Fill in any that weren't processed due to time
        for i in range(n):
            if result[i] is None:
                result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
        
        return result
    
    best_placements = None
    best_score = -1
    
    for sort_func in strategies:
        if time.time() > deadline:
            break
        
        placements = run_strategy(sort_func)
        
        # Score: fraction of container volume filled
        total_vol = sum(p[4]*p[5]*p[6] for p in placements if p[0] == 0)
        score = total_vol / (W * H * D) if W * H * D > 0 else 0
        
        if score > best_score:
            best_score = score
            best_placements = placements
    
    if best_placements is None:
        # Fallback: try the original simple approach
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placements
