    # Precompute overlaps
    overlap = np.zeros((M, M), dtype=bool)
    
    # Helper to check overlap of two unit hexagons at p1, p2
    def check_overlap_exact(p1, p2):
        dist_sq = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
        if dist_sq > 4.0:
            return False
        
        # SAT
        thresh = math.sqrt(3)
        # Normals
        # n0: (cos30, sin30)
        # n1: (cos90, sin90) = (0, 1)
        # n2: (cos150, sin150)
        # n3: -n0
        # n4: -n1
        # n5: -n2
        
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        
        # Projection on n0
        proj0 = abs(dx * math.cos(math.pi/6) + dy * math.sin(math.pi/6))
        if proj0 > thresh:
            return False
        
        # Projection on n1
        proj1 = abs(dy)
        if proj1 > thresh:
            return False
        
        # Projection on n2
        proj2 = abs(dx * math.cos(5*math.pi/6) + dy * math.sin(5*math.pi/6))
        if proj2 > thresh:
            return False
            
        return True

    for i in range(M):
        p1 = points_np[i]
        for j in range(i + 1, M):
            p2 = points_np[j]
            if check_overlap_exact(p1, p2):
                overlap[i, j] = True
                overlap[j, i] = True
