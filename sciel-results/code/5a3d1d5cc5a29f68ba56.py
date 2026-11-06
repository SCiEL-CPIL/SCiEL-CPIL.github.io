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
        return sorted(idx_list, key=lambda i: -(boxes[i][0]*boxes[i][1]*boxes[i][2]) / max(1, max(boxes[i])**2))
    
    def sort_by_fit(idx_list):
        # Prefer boxes that fit well in the container's smallest dimension
        min_c = min(W, H, D)
        return sorted(idx_list, key=lambda i: min(boxes[i]) / max(1, min_c))
    
    strategies = [sort_volume_desc, sort_max_dim_desc, sort_min_dim_asc, sort_ratio_desc, sort_by_fit]
    
    def check_overlap(x, y, z, rw, rh, rd, placed_rects):
        """Check if a box at (x,y,z) with dims (rw,rh,rd) overlaps with any placed box."""
        for (px, py, pz, pw, ph, pd) in placed_rects:
            if (x < px + pw and x + rw > px and
                y < py + ph and y + rh > py and
                z < pz + pd and z + rd > pz):
                return True
        return False
    
    def place_box_3d(rw, rh, rd, placed_rects, W, H, D):
        """
        Try to place a box with given dimensions in the container.
        Uses candidate positions from existing box edges plus corners.
        Returns (x, y, z) or None.
        """
        # Generate candidate positions
        x_candidates = [0]
        y_candidates = [0]
        z_candidates = [0]
        
        for (px, py, pz, pw, ph, pd) in placed_rects:
            # Right face of existing boxes
            if px + pw + rw <= W:
                x_candidates.append(px + pw)
            # Top face
            if py + ph + rh <= H:
                y_candidates.append(py + ph)
            # Back face
            if pz + pd + rd <= D:
                z_candidates.append(pz + pd)
        
        # Add corner positions
        x_candidates = sorted(set(x for x in x_candidates if 0 <= x <= W - rw))
        y_candidates = sorted(set(y for y in y_candidates if 0 <= y <= H - rh))
        z_candidates = sorted(set(z for z in z_candidates if 0 <= z <= D - rd))
        
        # Limit candidates to keep it fast but comprehensive
        max_cands = 20
        if len(x_candidates) > max_cands:
            x_candidates = x_candidates[:max_cands]
        if len(y_candidates) > max_cands:
            y_candidates = y_candidates[:max_cands]
        if len(z_candidates) > max_cands:
            z_candidates = z_candidates[:max_cands]
        
        # Try all combinations
        for x in x_candidates:
            for y in y_candidates:
                for z in z_candidates:
                    if not check_overlap(x, y, z, rw, rh, rd, placed_rects):
                        return (x, y, z)
        return None
    
    def run_strategy(sort_func):
        """Run a packing strategy and return placements."""
        placed_rects = []
        result = [None] * n
        order = sort_func(list(range(n)))
        
        for i in order:
            if time.time() > deadline:
                break
            
            # Try each rotation
            best_pos = None
            for rw, rh, rd in box_rots[i]:
                if rw > W or rh > H or rd > D:
                    continue
                pos = place_box_3d(rw, rh, rd, placed_rects, W, H, D)
                if pos is not None:
                    best_pos = (pos, rw, rh, rd)
                    break  # Take first valid rotation
            
            if best_pos is not None:
                (x, y, z), rw, rh, rd = best_pos
                placed_rects.append((x, y, z, rw, rh, rd))
                result[i] = (0, x, y, z, rw, rh, rd)
            else:
                result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
        
        # Fill in any that weren't processed due to time
        for i in range(n):
            if result[i] is None:
                result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
        
        return result
    
    def run_layer_packing(sort_func, num_layers):
        """
        Divide container into horizontal layers and pack each layer independently.
        This can be more efficient for boxes with one small dimension.
        """
        result = [None] * n
        order = sort_func(list(range(n)))
        layer_height = H / num_layers if num_layers > 0 else H
        
        # For each layer, do 2D packing in W x D plane
        for layer_idx in range(num_layers):
            z_start = layer_idx * layer_height
            z_end = min(z_start + layer_height, H)
            layer_h = z_end - z_start
            
            if layer_h <= 0:
                continue
            
            # 2D shelf packing for this layer
            placed_2d = []  # list of (x, y, w, d)
            
            for i in order:
                if time.time() > deadline:
                    break
                
                bw, bh, bd = boxes[i]
                
                # Find rotation that fits in the layer height
                valid_rots = []
                for rw, rh, rd in box_rots[i]:
                    if rh <= layer_h + 1e-9:
                        valid_rots.append((rw, rd))  # (x_dim, z_dim) for 2D
                        break  # Take first valid rotation for speed
                
                if not valid_rots:
                    continue
                
                rw, rd = valid_rots[0]
                
                # Try to place in 2D using candidate positions
                x_candidates = [0]
                y_candidates = [0]
                
                for (px, py, pw, pd) in placed_2d:
                    if px + pw + rw <= W:
                        x_candidates.append(px + pw)
                    if py + pd + rd <= D:
                        y_candidates.append(py + pd)
                
                x_candidates = sorted(set(x for x in x_candidates if 0 <= x <= W - rw))
                y_candidates = sorted(set(y for y in y_candidates if 0 <= y <= D - rd))
                
                max_cands = 15
                if len(x_candidates) > max_cands:
                    x_candidates = x_candidates[:max_cands]
                if len(y_candidates) > max_cands:
                    y_candidates = y_candidates[:max_cands]
                
                placed = False
                for x in x_candidates:
                    for y in y_candidates:
                        # Check overlap in 2D
                        overlap = False
                        for (px, py, pw, pd) in placed_2d:
                            if (x < px + pw and x + rw > px and
                                y < py + pd and y + rd > py):
                                overlap = True
                                break
                        
                        if not overlap:
                            placed_2d.append((x, y, rw, rd))
                            # Use the actual box height, not layer height
                            result[i] = (0, x, y, z_start, rw, bh, rd)
                            placed = True
                            break
                    if placed:
                        break
        
        # Fill in any that weren't processed
        for i in range(n):
            if result[i] is None:
                result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
        
        return result
    
    best_placements = None
    best_score = -1
    
    # Try all strategies with full 3D packing
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
    
    # Try layer decomposition with different numbers of layers
    if time.time() < deadline:
        for num_layers in [2, 3, 4, 5]:
            if time.time() > deadline:
                break
            
            for sort_func in strategies:
                if time.time() > deadline:
                    break
                
                placements = run_layer_packing(sort_func, num_layers)
                
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
