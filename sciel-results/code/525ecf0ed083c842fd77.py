<<<<<<< SEARCH
    def largest_empty_circle(centers, radii):
        """
        Finds the center and radius of the largest circle that can be added
        to the given set of circles without overlapping, within [0,1]x[0,1].
        """
        if len(centers) == 0:
            return (0.5, 0.5, 0.5)
        
        centers = np.array(centers)
        radii = np.array(radii)
        
        best_r = 0
        best_c = None
        
        # 1. Candidate centers from Voronoi vertices
        if len(centers) >= 3:
            try:
                vor = Voronoi(centers)
                
                # Voronoi vertices
                for v in vor.vertices:
                    if np.any(np.isnan(v)):
                        continue
                    if v[0] < 0 or v[0] > 1 or v[1] < 0 or v[1] > 1:
                        continue
                    
                    r = min(v[0], 1-v[0], v[1], 1-v[1])
                    for i in range(len(centers)):
                        d = np.sqrt((v[0]-centers[i][0])**2 + (v[1]-centers[i][1])**2)
                        r = min(r, d - radii[i])
                    
                    if r > best_r:
                        best_r = r
                        best_c = v
                        
            except Exception:
                pass
        
        # 2. Candidate centers on the boundary of the square
        # Sample the boundary to catch maxima that lie on the box edges
        num_samples = 500
        for side in range(4):
            for t in np.linspace(0, 1, num_samples):
                if side == 0: # y=0
                    x, y = t, 0.0
                elif side == 1: # y=1
                    x, y = t, 1.0
                elif side == 2: # x=0
                    x, y = 0.0, t
                else: # x=1
                    x, y = 1.0, t
                
                r = min(x, 1-x, y, 1-y)
                for i in range(len(centers)):
                    d = np.sqrt((x-centers[i][0])**2 + (y-centers[i][1])**2)
                    r = min(r, d - radii[i])
                
                if r > best_r:
                    best_r = r
                    best_c = np.array([x, y])
        
        # 3. Circle-Circle intersection points
        # The center of the largest empty circle might be at the intersection of two existing circles
        n_centers = len(centers)
        for i in range(n_centers):
            for j in range(i+1, n_centers):
                c1 = centers[i]
                c2 = centers[j]
                r1 = radii[i]
                r2 = radii[j]
                
                d = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                
                # If circles are too far or overlapping too much, no intersection or no valid gap
                if d < 1e-6 or d > r1 + r2 or d < abs(r1 - r2):
                    continue
                
                # Calculate intersection points
                # a = (d^2 + r1^2 - r2^2) / (2d)
                a = (d**2 + r1**2 - r2**2) / (2*d)
                h = np.sqrt(max(0, r1**2 - a**2))
                
                # Midpoint between c1 and c2
                mid_x = c1[0] + a * (c2[0] - c1[0]) / d
                mid_y = c1[1] + a * (c2[1] - c1[1]) / d
                
                # Offset perpendicular to the line connecting centers
                off_x = h * (c2[1] - c1[1]) / d
                off_y = h * (c1[0] - c2[0]) / d
                
                p1 = np.array([mid_x + off_x, mid_y + off_y])
                p2 = np.array([mid_x - off_x, mid_y - off_y])
                
                for p in [p1, p2]:
                    if p[0] < 0 or p[0] > 1 or p[1] < 0 or p[1] > 1:
                        continue
                    
                    r = min(p[0], 1-p[0], p[1], 1-p[1])
                    for k in range(n_centers):
                        d_k = np.sqrt((p[0]-centers[k][0])**2 + (p[1]-centers[k][1])**2)
                        r = min(r, d_k - radii[k])
                    
                    if r > best_r:
                        best_r = r
                        best_c = p
        
        if best_c is None:
            best_c = np.array([0.5, 0.5])
            best_r = 0.01
            
        return best_c[0], best_c[1], max(best_r, 1e-6)
=======
    def largest_empty_circle(centers, radii):
        """
        Finds the center and radius of the largest circle that can be added
        to the given set of circles without overlapping, within [0,1]x[0,1].
        """
        if len(centers) == 0:
            return (0.5, 0.5, 0.5)
        
        centers = np.array(centers)
        radii = np.array(radii)
        
        best_r = 0
        best_c = None
        
        # 1. Candidate centers from Voronoi vertices
        if len(centers) >= 3:
            try:
                vor = Voronoi(centers)
                
                # Voronoi vertices
                for v in vor.vertices:
                    if np.any(np.isnan(v)):
                        continue
                    if v[0] < 0 or v[0] > 1 or v[1] < 0 or v[1] > 1:
                        continue
                    
                    # Vectorized distance check
                    dists = np.sqrt((v[0]-centers[:,0])**2 + (v[1]-centers[:,1])**2)
                    r = min(v[0], 1-v[0], v[1], 1-v[1])
                    r = min(r, np.min(dists - radii))
                    
                    if r > best_r:
                        best_r = r
                        best_c = v
                        
            except Exception:
                pass
        
        # 2. Candidate centers on the boundary of the square
        num_samples = 100
        boundary_points = []
        for side in range(4):
            t = np.linspace(0, 1, num_samples)
            if side == 0:
                boundary_points.append(np.column_stack([t, np.zeros(num_samples)]))
            elif side == 1:
                boundary_points.append(np.column_stack([t, np.ones(num_samples)]))
            elif side == 2:
                boundary_points.append(np.column_stack([np.zeros(num_samples), t]))
            else:
                boundary_points.append(np.column_stack([np.ones(num_samples), t]))
        
        boundary_points = np.vstack(boundary_points)
        
        # Vectorized check for all boundary points
        dists = np.sqrt((boundary_points[:, 0:1] - centers[0:1, :])**2 + 
                        (boundary_points[:, 1:2] - centers[1:2, :])**2)
        
        r_bound = np.minimum.reduce([
            boundary_points[:, 0:1], 
            1 - boundary_points[:, 0:1], 
            boundary_points[:, 1:2], 
            1 - boundary_points[:, 1:2]
        ])
        
        r_dist = dists - radii[0:1, :]
        
        r_max = np.minimum(r_bound, np.min(r_dist, axis=1)[:, np.newaxis])
        
        max_idx = np.argmax(r_max.flatten())
        if r_max.flatten()[max_idx] > best_r:
            best_r = r_max.flatten()[max_idx]
            best_c = boundary_points[max_idx]
        
        # 3. Circle-Circle intersection points
        n_centers = len(centers)
        for i in range(n_centers):
            for j in range(i+1, n_centers):
                c1 = centers[i]
                c2 = centers[j]
                r1 = radii[i]
                r2 = radii[j]
                
                d = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                
                if d < 1e-6 or d > r1 + r2 or d < abs(r1 - r2):
                    continue
                
                a = (d**2 + r1**2 - r2**2) / (2*d)
                h = np.sqrt(max(0, r1**2 - a**2))
                
                mid_x = c1[0] + a * (c2[0] - c1[0]) / d
                mid_y = c1[1] + a * (c2[1] - c1[1]) / d
                
                off_x = h * (c2[1] - c1[1]) / d
                off_y = h * (c1[0] - c2[0]) / d
                
                p1 = np.array([mid_x + off_x, mid_y + off_y])
                p2 = np.array([mid_x - off_x, mid_y - off_y])
                
                for p in [p1, p2]:
                    if p[0] < 0 or p[0] > 1 or p[1] < 0 or p[1] > 1:
                        continue
                    
                    dists = np.sqrt((p[0]-centers[:,0])**2 + (p[1]-centers[:,1])**2)
                    r = min(p[0], 1-p[0], p[1], 1-p[1])
                    r = min(r, np.min(dists - radii))
                    
                    if r > best_r:
                        best_r = r
                        best_c = p
        
        if best_c is None:
            best_c = np.array([0.5, 0.5])
            best_r = 0.01
            
        return best_c[0], best_c[1], max(best_r, 1e-6)
>>>>>>> REPLACE

