def has_overlap(inner):
    """Check if any two inner hexagons overlap."""
    from scipy.spatial import ConvexHull
    # Simple check: for each pair, check if distance < min_dist
    # Min dist for same angle is sqrt(3).
    # For different angles, it's more complex.
    # Let's use a robust geometric check.
    # If time permits, use shapely if available? No, only stdlib/np/scipy.
    # Let's implement a simple edge intersection check for convex polygons.
    
    def polygons_intersect(p1, p2):
        # p1, p2 are arrays of vertices (6, 2)
        # Check if any vertex of p1 is inside p2 or vice versa
        # And if any edges intersect.
        
        def point_in_polygon(pt, poly):
            # Ray casting
            x, y = pt
            n = len(poly)
            inside = False
            j = n - 1
            for i in range(n):
                xi, yi = poly[i]
                xj, yj = poly[j]
                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                    inside = not inside
                j = i
            return inside

        # Check vertices
        for v in p1:
            if point_in_polygon(v, p2):
                return True
        for v in p2:
            if point_in_polygon(v, p1):
                return True
        
        # Check edge intersections
        def ccw(A, B, C):
            return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

        def on_segment(A, B, C):
            # Check if C is on segment AB
            return min(A[0], B[0]) <= C[0] <= max(A[0], B[0]) and \
                   min(A[1], B[1]) <= C[1] <= max(A[1], B[1])

        def segments_intersect(A, B, C, D):
            if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):
                return True
            if on_segment(A, B, C) or on_segment(A, B, D) or on_segment(C, D, A) or on_segment(C, D, B):
                return True
            return False

        for i in range(6):
            A = p1[i]
            B = p1[(i+1)%6]
            for j in range(6):
                C = p2[j]
                D = p2[(j+1)%6]
                if segments_intersect(A, B, C, D):
                    return True
        return False

    n = len(inner)
    for i in range(n):
        for j in range(i+1, n):
            v1 = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            v2 = hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            if polygons_intersect(v1, v2):
                return True
    return False
