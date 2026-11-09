<<<<<<< SEARCH
    def min_dist_between_hexes(hex1, hex2):
        """Minimum distance between two convex polygons (hexagons).
        Returns 0 if they intersect or touch, > 0 if separated."""
        # Check if any edge of hex1 intersects any edge of hex2
        # Also check if any vertex of one is inside the other
        from scipy.spatial import ConvexHull
        try:
            # Check edge intersections
            def segments_intersect(p1, p2, p3, p4):
                def cross(o, a, b):
                    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
                d1 = cross(p3, p4, p1)
                d2 = cross(p3, p4, p2)
                d3 = cross(p1, p2, p3)
                d4 = cross(p1, p2, p4)
                if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
                   ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                    return True
                return False
            
            for i in range(6):
                for j in range(6):
                    p1 = hex1[i]
                    p2 = hex1[(i+1) % 6]
                    p3 = hex2[j]
                    p4 = hex2[(j+1) % 6]
                    if segments_intersect(p1, p2, p3, p4):
                        return 0.0
            
            # Check if any vertex of hex1 is inside hex2
            def point_in_polygon(point, polygon):
                x, y = point
                n = len(polygon)
                inside = False
                j = n - 1
                for i in range(n):
                    xi, yi = polygon[i]
                    xj, yj = polygon[j]
                    if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                        inside = not inside
                    j = i
                return inside
            
            for i in range(6):
                if point_in_polygon(hex1[i], hex2):
                    return 0.0
            for i in range(6):
                if point_in_polygon(hex2[i], hex1):
                    return 0.0
            
            # Minimum distance between edges
            min_d = float('inf')
            def point_segment_dist(p, a, b):
                ab = b - a
                ap = p - a
                ab2 = np.dot(ab, ab)
                if ab2 == 0:
                    return np.linalg.norm(p - a)
                t = np.dot(ap, ab) / ab2
                t = max(0, min(1, t))
                proj = a + t * ab
                return np.linalg.norm(p - proj)
            
            for i in range(6):
                for j in range(6):
                    p1 = hex1[i]
                    p2 = hex1[(i+1) % 6]
                    p3 = hex2[j]
                    p4 = hex2[(j+1) % 6]
                    d = min(
                        point_segment_dist(p1, p3, p4),
                        point_segment_dist(p2, p3, p4),
                        point_segment_dist(p3, p1, p2),
                        point_segment_dist(p4, p1, p2)
                    )
                    if d < min_d:
                        min_d = d
            return min_d
        except:
            return 0.0
=======
    def min_dist_between_hexes(hex1, hex2):
        """Minimum distance between two convex polygons (hexagons).
        Returns 0 if they intersect or touch, > 0 if separated."""
        try:
            # Ensure shapes are correct
            if hex1.shape != (6, 2) or hex2.shape != (6, 2):
                return 0.0
                
            # Check edge intersections
            def segments_intersect(p1, p2, p3, p4):
                def cross(o, a, b):
                    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
                d1 = cross(p3, p4, p1)
                d2 = cross(p3, p4, p2)
                d3 = cross(p1, p2, p3)
                d4 = cross(p1, p2, p4)
                if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
                   ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                    return True
                return False
            
            for i in range(6):
                for j in range(6):
                    p1 = hex1[i]
                    p2 = hex1[(i+1) % 6]
                    p3 = hex2[j]
                    p4 = hex2[(j+1) % 6]
                    if segments_intersect(p1, p2, p3, p4):
                        return 0.0
            
            # Check if any vertex of hex1 is inside hex2
            def point_in_polygon(point, polygon):
                x, y = point
                n = len(polygon)
                inside = False
                j = n - 1
                for i in range(n):
                    xi, yi = polygon[i]
                    xj, yj = polygon[j]
                    if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                        inside = not inside
                    j = i
                return inside
            
            for i in range(6):
                if point_in_polygon(hex1[i], hex2):
                    return 0.0
            for i in range(6):
                if point_in_polygon(hex2[i], hex1):
                    return 0.0
            
            # Minimum distance between edges
            min_d = float('inf')
            def point_segment_dist(p, a, b):
                ab = b - a
                ap = p - a
                ab2 = np.dot(ab, ab)
                if ab2 == 0:
                    return np.linalg.norm(p - a)
                t = np.dot(ap, ab) / ab2
                t = max(0, min(1, t))
                proj = a + t * ab
                return np.linalg.norm(p - proj)
            
            for i in range(6):
                for j in range(6):
                    p1 = hex1[i]
                    p2 = hex1[(i+1) % 6]
                    p3 = hex2[j]
                    p4 = hex2[(j+1) % 6]
                    d = min(
                        point_segment_dist(p1, p3, p4),
                        point_segment_dist(p2, p3, p4),
                        point_segment_dist(p3, p1, p2),
                        point_segment_dist(p4, p1, p2)
                    )
                    if d < min_d:
                        min_d = d
            return min_d
        except Exception:
            return 0.0
>>>>>>> REPLACE

Additionally, the optimization might be too slow. I will reduce the number of iterations and popsize to ensure it finishes in time, and add a fallback to a simple grid search or initial guess if optimization fails or times out.

