"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import heapq

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Preprocess: compute volume and sort by volume descending
    indexed_boxes = [(w, h, d, i) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[0]*x[1]*x[2], reverse=True)
    
    # State: list of placed boxes as (x, y, z, w, h, d)
    placed = []
    # For each original index, store (bin, x, y, z, w, h, d)
    result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    def get_orientations(w, h, d):
        """Generate all 6 unique orientations as tuples."""
        orients = set()
        for perm in itertools.permutations([w, h, d]):
            orients.add(perm)
        return list(orients)
    
    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        """Check if two boxes overlap."""
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1)
    
    def fits(x, y, z, w, h, d):
        """Check if a box at (x,y,z) with dims (w,h,d) fits in container without overlap."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in placed:
            if overlaps(x, y, z, w, h, d, px, py, pz, pw, ph, pd):
                return False
        return True
    
    def candidate_positions():
        """Generate candidate positions by combining x/y/z coordinates from placed boxes."""
        xs = set([0])
        ys = set([0])
        zs = set([0])
        
        for (px, py, pz, pw, ph, pd) in placed:
            xs.add(px)
            xs.add(px + pw)
            ys.add(py)
            ys.add(py + ph)
            zs.add(pz)
            zs.add(pz + pd)
        
        # Filter coordinates to be within container bounds
        xs = [x for x in xs if 0 <= x <= W]
        ys = [y for y in ys if 0 <= y <= H]
        zs = [z for z in zs if 0 <= z <= D]
        
        # If too many combinations, limit to a subset for speed
        max_candidates = 5000
        if len(xs) * len(ys) * len(zs) > max_candidates:
            # Prioritize smaller coordinates (closer to origin)
            xs.sort()
            ys.sort()
            zs.sort()
            # Take first N from each dimension
            limit = max(1, int((max_candidates ** (1/3)) + 1))
            xs = xs[:limit]
            ys = ys[:limit]
            zs = zs[:limit]
        
        positions = []
        for x in xs:
            for y in ys:
                for z in zs:
                    positions.append((x, y, z))
        
        # Sort by proximity to origin for deterministic ordering
        positions.sort(key=lambda p: (p[0], p[1], p[2]))
        
        return positions
    
    def best_fit_position(w, h, d):
        """Find the best position for a box with given dimensions using best-fit decreasing."""
        best_pos = None
        best_score = float('inf')
        
        for (x, y, z) in candidate_positions():
            if not fits(x, y, z, w, h, d):
                continue
            # Multi-criteria score: prefer compact placement near origin
            # Primary: minimize x+y+z (compactness)
            # Secondary: minimize max extent to preserve space
            primary = x + y + z
            # Use a small secondary term to break ties
            secondary = max(x + w, y + h, z + d) * 0.01
            score = primary + secondary
            if score < best_score:
                best_score = score
                best_pos = (x, y, z)
        
        return best_pos
    
    def pack_bfd():
        """Best-Fit Decreasing packing."""
        placed_local = []
        result_local = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
        for (w, h, d, orig_idx) in indexed_boxes:
            if time.time() > deadline:
                break
            
            orients = get_orientations(w, h, d)
            
            best_placement = None
            best_score = float('inf')
            
            for (ow, oh, od) in orients:
                pos = _find_best_position(ow, oh, od, placed_local)
                if pos is not None:
                    x, y, z = pos
                    # Score: prefer placements that minimize the resulting bounding box
                    score = x + y + z
                    if score < best_score:
                        best_score = score
                        best_placement = (x, y, z, ow, oh, od)
            
            if best_placement is not None:
                x, y, z, ow, oh, od = best_placement
                placed_local.append((x, y, z, ow, oh, od))
                result_local[orig_idx] = (0, x, y, z, ow, oh, od)
        
        return result_local
    
    def _find_best_position(w, h, d, placed_boxes):
        """Find the best position for a box with given dimensions using extremal point search."""
        best_pos = None
        best_score = float('inf')
        
        # Generate candidate positions from extremal points
        xs = set([0])
        ys = set([0])
        zs = set([0])
        
        for (px, py, pz, pw, ph, pd) in placed_boxes:
            xs.add(px)
            xs.add(px + pw)
            ys.add(py)
            ys.add(py + ph)
            zs.add(pz)
            zs.add(pz + pd)
        
        xs = [x for x in xs if 0 <= x <= W]
        ys = [y for y in ys if 0 <= y <= H]
        zs = [z for z in zs if 0 <= z <= D]
        
        # Limit candidates if too many
        max_candidates = 5000
        if len(xs) * len(ys) * len(zs) > max_candidates:
            xs.sort()
            ys.sort()
            zs.sort()
            limit = max(1, int((max_candidates ** (1/3)) + 1))
            xs = xs[:limit]
            ys = ys[:limit]
            zs = zs[:limit]
        
        for x in xs:
            if x + w > W:
                continue
            for y in ys:
                if y + h > H:
                    continue
                for z in zs:
                    if z + d > D:
                        continue
                    if not _fits_inside(x, y, z, w, h, d, placed_boxes):
                        continue
                    # Score: prefer compact placement near origin
                    score = x + y + z + max(x + w, y + h, z + d) * 0.01
                    if score < best_score:
                        best_score = score
                        best_pos = (x, y, z)
        
        return best_pos
    
    def _fits_inside(x, y, z, w, h, d, placed_boxes):
        """Check if a box fits without overlap."""
        for (px, py, pz, pw, ph, pd) in placed_boxes:
            if overlaps(x, y, z, w, h, d, px, py, pz, pw, ph, pd):
                return False
        return True
    
    def pack_shelf():
        """Shelf-based packing: place boxes in horizontal shelves."""
        placed_local = []
        result_local = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
        # Sort boxes by height (z-dimension) for shelf packing
        shelf_boxes = sorted(indexed_boxes, key=lambda x: min(x[0], x[1], x[2]))
        
        current_z = 0
        shelf_boxes_list = []
        
        for (w, h, d, orig_idx) in shelf_boxes:
            if time.time() > deadline:
                break
            
            # Find the orientation that minimizes the height (z-dimension)
            orients = get_orientations(w, h, d)
            best_orient = None
            best_z = float('inf')
            
            for (ow, oh, od) in orients:
                if od < best_z:
                    best_z = od
                    best_orient = (ow, oh, od)
            
            if best_orient is None:
                continue
            
            ow, oh, od = best_orient
            
            # Find a position in the current shelf
            pos = None
            for x in range(0, W - ow + 1):
                if x + ow > W:
                    break
                for y in range(0, H - oh + 1):
                    if y + oh > H:
                        break
                    if not _fits_inside(x, y, current_z, ow, oh, od, placed_local):
                        continue
                    # Check if this creates a good fit (minimal wasted space)
                    pos = (x, y, current_z)
                    break
                if pos:
                    break
            
            if pos is None:
                # Try to start a new shelf
                new_z = current_z + best_z
                if new_z + od > D:
                    # Can't fit in container
                    continue
                current_z = new_z
                pos = (0, 0, current_z)
            
            x, y, z = pos
            placed_local.append((x, y, z, ow, oh, od))
            result_local[orig_idx] = (0, x, y, z, ow, oh, od)
        
        return result_local
    
    def volume_filled(result_list):
        """Calculate the volume filled by placed boxes."""
        total_vol = 0
        for (bin_id, x, y, z, w, h, d) in result_list:
            if bin_id == 0:
                total_vol += w * h * d
        return total_vol / (W * H * D)
    
    # Try multiple strategies and pick the best
    best_result = None
    best_vol = 0
    
    # Strategy 1: BFD
    result1 = pack_bfd()
    vol1 = volume_filled(result1)
    if vol1 > best_vol:
        best_vol = vol1
        best_result = result1
    
    # Strategy 2: Shelf packing
    result2 = pack_shelf()
    vol2 = volume_filled(result2)
    if vol2 > best_vol:
        best_vol = vol2
        best_result = result2
    
    # If we have time, try a hybrid approach
    if time.time() < deadline - 20:
        # Try BFD with different sorting orders
        for sort_key in [lambda x: x[0]*x[1]*x[2], lambda x: max(x[0], x[1], x[2]), lambda x: x[0]+x[1]+x[2]]:
            if time.time() > deadline:
                break
            
            indexed_boxes2 = [(w, h, d, i) for i, (w, h, d) in enumerate(boxes)]
            indexed_boxes2.sort(key=sort_key, reverse=True)
            
            placed_local = []
            result_local = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
            
            for (w, h, d, orig_idx) in indexed_boxes2:
                if time.time() > deadline:
                    break
                
                orients = get_orientations(w, h, d)
                
                best_placement = None
                best_score = float('inf')
                
                for (ow, oh, od) in orients:
                    pos = _find_best_position(ow, oh, od, placed_local)
                    if pos is not None:
                        x, y, z = pos
                        score = x + y + z
                        if score < best_score:
                            best_score = score
                            best_placement = (x, y, z, ow, oh, od)
                
                if best_placement is not None:
                    x, y, z, ow, oh, od = best_placement
                    placed_local.append((x, y, z, ow, oh, od))
                    result_local[orig_idx] = (0, x, y, z, ow, oh, od)
            
            vol3 = volume_filled(result_local)
            if vol3 > best_vol:
                best_vol = vol3
                best_result = result_local
    
    return best_result if best_result is not None else result
# EVOLVE-BLOCK-END
