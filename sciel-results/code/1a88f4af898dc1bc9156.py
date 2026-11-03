<<<<<<< SEARCH
    def local_search(centers, max_iters=5000, step_size=0.02, seed=0):
        """
        Discrete best-fit local search: try small perturbations to centers,
        accept if it improves the sum of radii.
        """
        rng = np.random.RandomState(seed)
        current_centers = centers.copy()
        current_radii = compute_radii_from_centers(current_centers)
        current_score = np.sum(current_radii)
        
        best_centers = current_centers.copy()
        best_score = current_score
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick a random circle and try to move it
            i = rng.randint(N)
            dx = rng.uniform(-step_size, step_size)
            dy = rng.uniform(-step_size, step_size)
            
            new_centers = current_centers.copy()
            new_centers[i, 0] = np.clip(new_centers[i, 0] + dx, 0.02, 0.98)
            new_centers[i, 1] = np.clip(new_centers[i, 1] + dy, 0.02, 0.98)
            
            new_radii = compute_radii_from_centers(new_centers)
            new_score = np.sum(new_radii)
            
            if new_score > current_score + 1e-10:
                current_centers = new_centers
                current_radii = new_radii
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_centers = current_centers.copy()
        
        return best_centers
=======
    def local_search(centers, max_iters=5000, step_size=0.02, seed=0):
        """
        Discrete best-fit local search with multi-circle moves.
        """
        rng = np.random.RandomState(seed)
        current_centers = centers.copy()
        current_radii = compute_radii_from_centers(current_centers)
        current_score = np.sum(current_radii)
        
        best_centers = current_centers.copy()
        best_score = current_score
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # With some probability, move 2 circles simultaneously
            if rng.random() < 0.3:
                i = rng.randint(N)
                j = rng.randint(N)
                while j == i:
                    j = rng.randint(N)
                new_centers = current_centers.copy()
                new_centers[i, 0] = np.clip(new_centers[i, 0] + rng.uniform(-step_size, step_size), 0.02, 0.98)
                new_centers[i, 1] = np.clip(new_centers[i, 1] + rng.uniform(-step_size, step_size), 0.02, 0.98)
                new_centers[j, 0] = np.clip(new_centers[j, 0] + rng.uniform(-step_size, step_size), 0.02, 0.98)
                new_centers[j, 1] = np.clip(new_centers[j, 1] + rng.uniform(-step_size, step_size), 0.02, 0.98)
            else:
                i = rng.randint(N)
                new_centers = current_centers.copy()
                new_centers[i, 0] = np.clip(new_centers[i, 0] + rng.uniform(-step_size, step_size), 0.02, 0.98)
                new_centers[i, 1] = np.clip(new_centers[i, 1] + rng.uniform(-step_size, step_size), 0.02, 0.98)
            
            new_radii = compute_radii_from_centers(new_centers)
            new_score = np.sum(new_radii)
            
            if new_score > current_score + 1e-10:
                current_centers = new_centers
                current_radii = new_radii
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_centers = current_centers.copy()
        
        return best_centers
>>>>>>> REPLACE

