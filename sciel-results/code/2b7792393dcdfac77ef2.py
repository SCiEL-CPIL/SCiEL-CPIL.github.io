    def construct():
        start_time = time.time()
        deadline = start_time + 120.0
        sqrt3 = np.sqrt(3)
        
        # Generate 11 pointy-top hexagons in a valid packing
        # Axial coords (q, r) -> Cartesian (x, y)
        # x = sqrt(3) * (q + r/2)
        # y = 1.5 * r
        axial_coords = [
            (0, 0),
            (1, 0), (-1, 0),
            (0, 1), (0, -1),
            (1, -1), (-1, 1),
            (1, 1), (-1, -1),
            (2, 0), (-2, 0)
        ]
        
        inner = []
        for q, r in axial_coords:
            x = sqrt3 * (q + r / 2.0)
            y = 1.5 * r
            inner.append([x, y, 90.0])
            
        # Verify inner-inner non-overlap
        # Min distance between centers is sqrt(3)
        min_dist = sqrt3
        valid = True
        for i in range(len(inner)):
            for j in range(i + 1, len(inner)):
                dx = inner[i][0] - inner[j][0]
                dy = inner[i][1] - inner[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < min_dist - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to a single hexagon if packing fails (should not happen)
            inner = [[0.0, 0.0, 90.0]]
            
        best_side = 40.0
        best_angle = 0.0
        
        # Search for optimal outer hexagon
        # The outer hexagon should enclose all inner hexagons.
        # We can estimate a good starting side length.
        # Max distance from center to any vertex of inner hexagons.
        max_r = 0
        for ix, iy, iang in inner:
            # Approximate bounding circle radius
            # For pointy-top, max dist to vertex is 1.
            # So max dist from origin is dist(center) + 1
            d = np.sqrt(ix*ix + iy*iy)
            if d + 1 > max_r:
                max_r = d + 1
        
        # Outer hexagon inradius must be >= max_r?
        # No, outer hexagon contains the inner hexagons.
        # The outer hexagon's inradius (apothem) must be large enough to contain the furthest points.
        # Actually, we just binary search the side length.
        
        for angle in range(0, 180, 5): # Coarser search for speed
            lo, hi = 3.0, 20.0
            for _ in range(30):
                mid = (lo + hi) / 2
                if _check_valid(inner, (0.0, 0.0), mid, float(angle)):
                    hi = mid
                else:
                    lo = mid
            if hi < best_side:
                best_side = hi
                best_angle = float(angle)
            
            if time.time() > deadline - 15.0:
                break
        
        return inner, (0.0, 0.0), best_side, best_angle
    