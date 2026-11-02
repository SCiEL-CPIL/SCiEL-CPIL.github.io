def points_in_hex(pt, hex_verts):
    # pt: (x,y)
    # hex_verts: (6,2)
    # Check if pt is inside convex polygon hex_verts
    # Cross product test
    for i in range(6):
        p1 = hex_verts[i]
        p2 = hex_verts[(i+1)%6]
        # Vector p1->p2 and p1->pt
        v1 = p2 - p1
        v2 = pt - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        # All crosses must have same sign (assuming CCW order)
        # We assume get_hex_vertices returns CCW
        if cross < -1e-9: # Allow small tolerance
            return False
    return True

def hexagons_overlap(hex1_verts, hex2_verts):
    # Check if any vertex of hex1 is in hex2
    for v in hex1_verts:
        if points_in_hex(v, hex2_verts):
            return True
    # Check if any vertex of hex2 is in hex1
    for v in hex2_verts:
        if points_in_hex(v, hex1_verts):
            return True
    # Check edge intersections? 
    # If no vertex is inside, they might still cross edges?
    # For convex polygons, if no vertex of A is in B and no vertex of B is in A, 
    # they can still intersect if edges cross.
    # Example: two squares crossing like a star.
    # So we must check edge intersections.
    
    # Simplified: Just check vertex-in-polygon is NOT enough.
    # But for hexagons packing, they usually touch sides or vertices.
    # To be safe, we should check edge intersections.
    
    # This is getting expensive.
    # Alternative: Use a library? No, standard library only.
    # Let's implement a simple segment intersection.
    
    def segments_intersect(p1, p2, p3, p4):
        # p1,p2 is segment 1, p3,p4 is segment 2
        def ccw(a, b, c):
            return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
        return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)
        
    for i in range(6):
        p1 = hex1_verts[i]
        p2 = hex1_verts[(i+1)%6]
        for j in range(6):
            p3 = hex2_verts[j]
            p4 = hex2_verts[(j+1)%6]
            if segments_intersect(p1, p2, p3, p4):
                return True
    return False
