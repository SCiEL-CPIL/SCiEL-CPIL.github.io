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
        3. Supported (z=0 or box below). 
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
        
        # Check support
        if z > 0:
            supported = False
            for (_, px, py, pz, pw, ph, pd) in bin_boxes:
                if (x < px + pw and px < x + w and
                    y < py + ph and py < y + h and
                    pz + pd == z):
                    supported = True
                    break
            if not supported:
                return False
        
        return True

    def update_points(points, x, y, z, w, h, d):
        """
        Update the set of extreme points after placing a box.
        """
        new_points = set()
        
        # Filter existing points
        for p in points:
            px, py, pz = p
            if x <= px < x + w and y <= py < y + h and z <= pz < z + d:
                continue
            new_points.add(p)
            
        # Add corners of the new box
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
            if c[0] <= W and c[1] <= H and c[2] <= D:
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
                x, y, z = best_pos
                bin_data['boxes'].append((orig_idx, x, y, z, w, h, d))
                bin_data['points'] = update_points(points, x, y, z, w, h, d)
                placed = True
                break
        
        if not placed:
            # Start new bin
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
