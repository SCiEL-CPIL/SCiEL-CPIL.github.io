import numpy as np

def pack(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    
    # Sort indices by descending volume
    indexed = list(enumerate(boxes))
    indexed.sort(key=lambda t: t[1][0] * t[1][1] * t[1][2], reverse=True)
    
    # bins: list of dicts or lists
    # Each bin has:
    #   'max_points': list of (x, y, z)
    #   'items': list of (x, y, z, w, h, d, orig_idx)
    bins = []
    
    def is_dominated(p, points):
        """Check if p is dominated by any point in points."""
        for q in points:
            if p == q:
                continue
            if q[0] <= p[0] and q[1] <= p[1] and q[2] <= p[2]:
                return True
        return False

    def is_inside_box(p, x, y, z, w, h, d):
        """Check if point p is strictly inside the box [x, x+w]x[y, y+h]x[z, z+d]."""
        # We consider points on the boundary as potentially valid, 
        # but points strictly inside are occupied.
        # Actually, maximal points are usually corners. 
        # If p is inside, it's not a corner of the free space.
        return (x < p[0] < x + w) and (y < p[1] < y + h) and (z < p[2] < z + d)

    def update_max_points(max_points, x, y, z, w, h, d):
        """Update the list of maximal points after placing a box at (x,y,z) with size (w,h,d)."""
        new_pts = [(x + w, y, z), (x, y + h, z), (x, y, z + d)]
        
        # Add new points
        for p in new_pts:
            max_points.append(p)
            
        # Remove points that are inside the placed box
        # A point is inside if x < px < x+w, etc.
        # Points on the boundary are kept as they might be valid for adjacent placements.
        # However, if a point is dominated by another, it's removed.
        
        # First pass: remove points strictly inside the box
        filtered = []
        for p in max_points:
            if not is_inside_box(p, x, y, z, w, h, d):
                filtered.append(p)
        
        # Second pass: remove dominated points
        # A point p is dominated if there exists q in filtered such that 
        # q.x <= p.x and q.y <= p.y and q.z <= p.z and q != p
        final_points = []
        for p in filtered:
            dominated = False
            for q in filtered:
                if p == q:
                    continue
                if q[0] <= p[0] and q[1] <= p[1] and q[2] <= p[2]:
                    dominated = True
                    break
            if not dominated:
                final_points.append(p)
                
        return final_points

    result = [None] * len(boxes)
    
    for orig_idx, (w, h, d) in indexed:
        best_bin_idx = -1
        best_pos = None
        best_score = float('inf')
        
        for bi in range(len(bins)):
            max_points = bins[bi]['max_points']
            
            # Find candidate positions
            candidates = []
            for p in max_points:
                x, y, z = p
                if x + w <= W and y + h <= H and z + d <= D:
                    # Valid position
                    # Heuristic: prefer bottom (low z), then left (low y), then back (low x)
                    # Or just min z, min y, min x
                    score = (z, y, x)
                    if score < best_score:
                        best_score = score
                        best_bin_idx = bi
                        best_pos = p
            
        if best_bin_idx != -1:
            x, y, z = best_pos
            bins[best_bin_idx]['max_points'] = update_max_points(bins[best_bin_idx]['max_points'], x, y, z, w, h, d)
            bins[best_bin_idx]['items'].append((x, y, z, w, h, d, orig_idx))
            result[orig_idx] = (best_bin_idx, x, y, z, w, h, d)
        else:
            # Open new bin
            new_bin = {
                'max_points': [(0, 0, 0)],
                'items': [(0, 0, 0, w, h, d, orig_idx)]
            }
            bins.append(new_bin)
            result[orig_idx] = (len(bins) - 1, 0, 0, 0, w, h, d)
            
    return result
