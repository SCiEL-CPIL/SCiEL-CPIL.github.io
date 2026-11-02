def min_separation(v1, v2):
    """
    Returns the minimum separation distance between two convex polygons.
    Positive if separated, negative if overlapping.
    """
    min_overlap = float('inf')
    
    # Get normals for both polygons
    # For a regular hexagon, normals are at angles 0, 60, ... or 30, 90... depending on vertex definition.
    # hex_vertices uses cos/sin, so vertices are at angles 0, 60, ...
    # Edge normals are at 30, 90, ...
    
    # We can just compute normals from the vertices provided.
    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            p1 = verts[i]
            p2 = verts[(i+1)%n]
            # Edge vector
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            # Normal (outward)
            # For CCW polygon, right normal is (ey, -ex)
            # Check orientation of hex_vertices:
            # theta = a + k*pi/3. k=0: (1,0). k=1: (0.5, 0.866).
            # This is CCW.
            # So right normal is (ey, -ex).
            nx = ey
            ny = -ex
            # Normalize
            norm = math.sqrt(nx*nx + ny*ny)
            if norm > 1e-10:
                normals.append((nx/norm, ny/norm))
        return normals

    n1 = get_normals(v1)
    n2 = get_normals(v2)
    
    all_normals = n1 + n2
    
    for nx, ny in all_normals:
        # Project v1
        proj1 = [v[0]*nx + v[1]*ny for v in v1]
        min1 = min(proj1)
        max1 = max(proj1)
        
        # Project v2
        proj2 = [v[0]*nx + v[1]*ny for v in v2]
        min2 = min(proj2)
        max2 = max(proj2)
        
        # Overlap on this axis
        # If max1 < min2 or max2 < min1, they are separated.
        # Separation distance = min(min2 - max1, min1 - max2) if positive.
        # Overlap amount = min(max1, max2) - max(min1, min2).
        # If overlap > 0, they overlap on this axis.
        # If overlap < 0, they are separated by -overlap.
        
        overlap = min(max1, max2) - max(min1, min2)
        
        # We want the minimum separation.
        # If separated (overlap < 0), separation = -overlap.
        # If overlapping (overlap > 0), this axis is not separating.
        # The minimum separation distance is the minimum over all separating axes of the separation.
        # If no axis separates them, they overlap.
        # The "depth" of overlap is the minimum over all axes of the overlap amount?
        # Actually, the minimum translation distance to separate is the minimum over all axes (separating or not) of the "gap".
        # If they overlap, the gap is negative.
        # So D = -max(0, min_overlap_amount)? No.
        
        # Let's define D as the minimum distance between the two sets.
        # If they are disjoint, D > 0.
        # If they intersect, D = 0.
        # But we want a signed distance for the penalty.
        
        # Standard SAT:
        # If for any axis, the projections do not overlap, the polygons are disjoint.
        # The distance is the minimum of the gaps on those axes.
        # If all axes have overlapping projections, the polygons intersect.
        
        if overlap < 0:
            # They are separated on this axis
            # The distance is -overlap
            # We track the minimum distance (closest approach)
            # Wait, if they are separated on one axis, they are disjoint.
            # The distance is the minimum gap over ALL axes? No.
            # The distance is the minimum gap over the axes that are separating?
            # Actually, the distance between two convex sets is the minimum over all separating axes of the gap.
            # If they are disjoint, there is at least one separating axis.
            # The distance is min(-overlap) over all axes where overlap < 0?
            # No, it's the minimum gap.
            
            # Let's just store the gap.
            gap = -overlap
            # We want the minimum gap?
            # If they are far apart, the gap on the normal axis is large.
            # The minimum gap is the actual distance.
            # So we want min(gap) over all axes?
            # No, if they are disjoint, the distance is the minimum over all axes of the gap?
            # Yes, for convex polygons, the distance is the minimum over all separating axes of the projection gap.
            
            if gap < min_overlap:
                min_overlap = gap
                
    if min_overlap == float('inf'):
        # No separating axis found -> they overlap
        # Return a negative value indicating overlap depth?
        # For penalty, we can return a small negative number or 0.
        # To make the penalty work, we need to know how much they overlap.
        # The overlap depth is the minimum over all axes of the overlap amount?
        # No, the minimum translation distance to separate is the minimum over all axes of the overlap amount (if we consider the axis that requires least movement).
        # Actually, if they overlap, the "separation" is negative.
        # Let's compute the maximum overlap? No.
        
        # If they overlap, let's return -1.0 as a default penalty trigger.
        # Or better, compute the minimum overlap depth.
        # The minimum translation distance to separate two overlapping convex polygons is the minimum over all axes of the overlap amount?
        # No, it's the minimum over all axes of the "penetration".
        
        # Let's just return -1.0 if they overlap. The penalty will be (1.0)^2 * weight.
        # This is a constant penalty, which is not great for guiding the optimizer.
        # But it will push them apart eventually.
        
        # To improve, we can compute the minimum overlap depth.
        min_overlap_depth = float('inf')
        for nx, ny in all_normals:
            proj1 = [v[0]*nx + v[1]*ny for v in v1]
            min1 = min(proj1)
            max1 = max(proj1)
            proj2 = [v[0]*nx + v[1]*ny for v in v2]
            min2 = min(proj2)
            max2 = max(proj2)
            overlap = min(max1, max2) - max(min1, min2)
            if overlap > 0:
                # They overlap on this axis by 'overlap'
                # The minimum translation to separate on this axis is 'overlap'
                # The actual minimum translation is the minimum of these 'overlap' values?
                # No, it's the minimum over all axes?
                # Yes, the minimum translation distance is the minimum over all axes of the overlap amount (if overlapping).
                if overlap < min_overlap_depth:
                    min_overlap_depth = overlap
                    
        if min_overlap_depth != float('inf'):
            return -min_overlap_depth
        else:
            return 0.0 # Should not happen if they overlap
            
    return min_overlap
