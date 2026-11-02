<<<<<<< SEARCH
    def compute_smooth_objective(params, beta=50.0):
        """
        Smooth objective for global search using soft-max penalty for overlaps.
        """
        if time.time() > deadline:
            return 1e10
            
        hexes = get_hexes_from_params(params)
        
        s_min = compute_min_side(hexes)
        
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                # Quick bounding box check
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    # Soft-max smooth penalty
                    penalty += np.log(1.0 + np.exp(beta * overlap)) / beta
                    
        return s_min + penalty
=======
    def compute_smooth_objective(params, beta=30.0):
        """
        Smooth objective for global search using soft penalty for overlaps.
        """
        if time.time() > deadline:
            return 1e10
            
        hexes = get_hexes_from_params(params)
        
        s_min = compute_min_side(hexes)
        
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                # Quick bounding box check
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    # Smooth penalty: use quadratic for small overlaps, linear for large
                    penalty += overlap * overlap * 10.0 + overlap * 2.0
                    
        return s_min + penalty
>>>>>>> REPLACE

