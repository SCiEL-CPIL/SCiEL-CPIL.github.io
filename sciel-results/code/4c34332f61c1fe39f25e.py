<<<<<<< SEARCH
    def covering_energy(x, beta=50.0):
        """
        Soft-max approximation of the covering radius.
        We sample a set of points on the sphere and compute the max distance to nearest config point.
        To make it differentiable, we use a soft-max (log-sum-exp) of the distances.
        Actually, a better energy for covering is the "max-min" distance.
        Let's use a fixed grid of test points.
        """
        pts = x.reshape(-1, 3)
        pts = normalize(pts)
        
        # Use a pre-computed set of test points (Fibonacci sphere)
        # This is expensive to compute every time, so we use a smaller set or cached one.
        # For efficiency in the loop, let's use a moderate number of test points.
        # However, N=50, so the Voronoi cells are large.
        
        # Alternative: Use the convex hull circumradii directly? 
        # The convex hull changes discretely, so it's not differentiable.
        
        # Let's stick to the soft-max of distances to a grid.
        # Generate grid once outside? No, we need it inside.
        # Let's use 2000 test points.
        
        if not hasattr(covering_energy, 'test_pts'):
            # Generate Fibonacci sphere points
            n_test = 2000
            i = np.arange(n_test) + 0.5
            z = 1.0 - 2.0 * i / n_test
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            covering_energy.test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            
        test_pts = covering_energy.test_pts
        
        # Compute distances from test_pts to pts
        # dists[i, j] = arccos(test_pts[i] . pts[j])
        dots = test_pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        
        # Min distance for each test point
        min_dists = np.min(dists, axis=1)
        
        # Soft max: (1/beta) * log(sum(exp(beta * min_dists)))
        # To avoid overflow, subtract max
        max_d = np.max(min_dists)
        if max_d == 0:
            return 0.0
        exp_dists = np.exp(beta * (min_dists - max_d))
        soft_max = max_d + (1.0 / beta) * np.log(np.sum(exp_dists))
        
        return soft_max
=======
    def covering_energy(x, beta=50.0):
        """
        Hybrid energy: combines soft-max over test grid with facet circumradii.
        """
        pts = x.reshape(-1, 3)
        pts = normalize(pts)
        
        if not hasattr(covering_energy, 'test_pts'):
            n_test = 2000
            i = np.arange(n_test) + 0.5
            z = 1.0 - 2.0 * i / n_test
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            covering_energy.test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            
        test_pts = covering_energy.test_pts
        
        # Component 1: soft-max over test grid
        dots = test_pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        
        max_d = np.max(min_dists)
        if max_d == 0:
            grid_energy = 0.0
        else:
            exp_dists = np.exp(beta * (min_dists - max_d))
            grid_energy = max_d + (1.0 / beta) * np.log(np.sum(exp_dists))
        
        # Component 2: soft-max over facet circumradii (if hull is valid)
        facet_radii, _ = compute_facet_circumradii(pts)
        if len(facet_radii) > 0:
            max_f = np.max(facet_radii)
            if max_f == 0:
                facet_energy = 0.0
            else:
                exp_f = np.exp(beta * (facet_radii - max_f))
                facet_energy = max_f + (1.0 / beta) * np.log(np.sum(exp_f))
        else:
            facet_energy = grid_energy
        
        # Combine: use the larger of the two as a proxy for covering radius
        # Weighted combination to smooth transitions
        combined = 0.5 * grid_energy + 0.5 * facet_energy
        return combined
>>>>>>> REPLACE

