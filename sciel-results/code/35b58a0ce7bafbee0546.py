"""3D Bin Packing using Skyline/Extreme Points method."""
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume in descending order
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1] * x[2] * x[3], reverse=True)
    
    # Each bin stores:
    # 1. list of placed boxes (orig_idx, x, y, z, w, h, d)
    # 2. set of extreme points (x, y, z)
    bins = []
    
    def is_valid_point(p, bin_boxes, w, h, d):
        """
        Check if a point p is a valid starting position for a box of dims w,h,d.
        1. Within container bounds.
        2. No overlap with existing boxes.
        3. Supported (z=0 or sufficient support from boxes below). 
        """
        x, y, z = p
        if x + w > W or y + h > H or z + d > D:
            return False
        
        # Check overlap
        for (_, px, py, pz, pw, ph, pd) in bin_boxes:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        
        # Check support: need at least some contact with boxes below or floor
        if z > 0:
            # Find all boxes that could provide support
            support_area = 0.0
            box_area = w * h
            for (_, px, py, pz, pw, ph, pd) in bin_boxes:
                # Box must be directly below (top face at z)
                if pz + pd != z:
                    continue
                # Check horizontal overlap
                overlap_x = max(0, min(x + w, px + pw) - max(x, px))
                overlap_y = max(0, min(y + h, py + ph) - max(y, py))
                support_area += overlap_x * overlap_y
            
            # Require at least 10% support (can be adjusted)
            if support_area < 0.1 * box_area:
                return False
        
        return True

    def update_points(points, x, y, z, w, h, d):
        """
        Update the set of extreme points after placing a box.
        Add all 8 corners of the new box and keep existing valid points.
        """
        new_points = set()
        
        # Filter existing points - remove those inside the new box
        for p in points:
            px, py, pz = p
            if x <= px < x + w and y <= py < y + h and z <= pz < z + d:
                continue
            new_points.add(p)
            
        # Add all 8 corners of the new box
        corners = [
            (x, y, z),
            (x + w, y, z),
            (x, y + h, z),
            (x + w, y + h, z),
            (x, y, z + d),
            (x + w, y, z + d),
            (x, y + h, z + d),
            (x + w, y + h, z + d)
        ]
        for c in corners:
            if 0 <= c[0] <= W and 0 <= c[1] <= H and 0 <= c[2] <= D:
                new_points.add(c)
                
        # Also add points on top face at various positions for better packing
        # Add midpoints on top face edges
        top_z = z + d
        mid_x = x + w // 2
        mid_y = y + h // 2
        extra_points = [
            (mid_x, y, top_z),
            (mid_x, y + h, top_z),
            (x, mid_y, top_z),
            (x + w, mid_y, top_z),
            (mid_x, mid_y, top_z),
        ]
        for c in extra_points:
            if 0 <= c[0] <= W and 0 <= c[1] <= H and 0 <= c[2] <= D:
                new_points.add(c)
                
        return new_points

    # Process boxes
    for (orig_idx, w, h, d) in indexed_boxes:
        placed = False
        
        # Try existing bins
        for bin_idx in range(len(bins)):
            bin_data = bins[bin_idx]
            bin_boxes = bin_data['boxes']
            points = bin_data['points']
            
            best_pos = None
            best_key = None
            
            # Check all extreme points
            for p in points:
                if is_valid_point(p, bin_boxes, w, h, d):
                    # Heuristic: Minimize z, then y, then x
                    key = (p[2], p[1], p[0])
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = p
                        
            if best_pos is not None:
                # Try to find a better position by checking nearby points
                bx, by, bz = best_pos
                # Check if we can shift slightly to pack tighter
                candidates = [(bx, by, bz)]
                # Try shifting x and y to align with existing box edges
                for (_, px, py, pz, pw, ph, pd) in bin_boxes:
                    # Try aligning right edge with left edge of existing box
                    if bx + w < px:
                        candidates.append((px, by, bz))
                    # Try aligning left edge with right edge of existing box
                    if bx > px + pw:
                        candidates.append((px + pw, by, bz))
                    # Try aligning back edge with front edge
                    if by + h < py:
                        candidates.append((bx, py, bz))
                    # Try aligning front edge with back edge
                    if by > py + ph:
                        candidates.append((bx, py + ph, bz))
                
                # Check all candidates and pick the best
                best_candidate = None
                best_candidate_key = None
                for cand in candidates:
                    cx, cy, cz = cand
                    if cx + w > W or cy + h > H or cz + d > D:
                        continue
                    if is_valid_point(cand, bin_boxes, w, h, d):
                        cand_key = (cz, cy, cx)
                        if best_candidate_key is None or cand_key < best_candidate_key:
                            best_candidate_key = cand_key
                            best_candidate = cand
                
                if best_candidate is not None:
                    best_pos = best_candidate
                x, y, z = best_pos
                bin_data['boxes'].append((orig_idx, x, y, z, w, h, d))
                bin_data['points'] = update_points(points, x, y, z, w, h, d)
                placed = True
                break
        
        if not placed:
            # Start new bin
            # Try to place at origin first
            if is_valid_point((0, 0, 0), [], w, h, d):
                bins.append({
                    'boxes': [(orig_idx, 0, 0, 0, w, h, d)],
                    'points': update_points({(0,0,0)}, 0, 0, 0, w, h, d)
                })
            else:
                # Shouldn't happen for first box in new bin, but handle gracefully
                bins.append({
                    'boxes': [(orig_idx, 0, 0, 0, w, h, d)],
                    'points': update_points({(0,0,0)}, 0, 0, 0, w, h, d)
                })
            
    # Build result in original order
    placements = [None] * n
    for bin_idx, bin_data in enumerate(bins):
        for (orig_idx, x, y, z, w, h, d) in bin_data['boxes']:
            placements[orig_idx] = (bin_idx, x, y, z, w, h, d)
            
    return placements
