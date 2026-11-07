    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine split
        n_lhs = int(self.budget * 0.75)
        n_local = self.budget - n_lhs
        
        # 1. Generate LHS samples using global np.random
        # Implementation of simple LHS
        lhs_samples = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            bin_width = (hi[d] - lo[d]) / n_lhs
            # Random points in each bin
            offsets = np.random.random(n_lhs) * bin_width
            points_in_bin = lo[d] + (np.arange(n_lhs) + offsets) * bin_width
            # Shuffle
            np.random.shuffle(points_in_bin)
            lhs_samples[:, d] = points_in_bin
        
        # Evaluate LHS
        archive_X = []
        archive_F = []
        
        for i in range(n_lhs):
            x = lhs_samples[i]
            f = problem(x)
            # Check if dominated by existing archive
            # For initial fill, we can just add and clean up later, 
            # or do incremental non-dominated sorting.
            # Since n_lhs is 150, incremental is fine.
            
            # Check if x is dominated by any in archive
            dominated = False
            for j in range(len(archive_F)):
                if np.all(archive_F[j] <= f) and np.any(archive_F[j] < f):
                    dominated = True
                    break
            
            if not dominated:
                # Remove any points in archive dominated by x
                new_archive_X = []
                new_archive_F = []
                for j in range(len(archive_F)):
                    # If archive_F[j] is dominated by f, skip it
                    if np.all(f <= archive_F[j]) and np.any(f < archive_F[j]):
                        continue
                    new_archive_X.append(archive_X[j])
                    new_archive_F.append(archive_F[j])
                
                new_archive_X.append(x)
                new_archive_F.append(f)
                archive_X = new_archive_X
                archive_F = new_archive_F
        
        # 2. Local Search
        # Use remaining budget
        for _ in range(n_local):
            if not archive_X:
                # Fallback: random sample
                x = lo + (hi - lo) * np.random.random(self.dim)
                f = problem(x)
                archive_X.append(x)
                archive_F.append(f)
                continue
                
            # Select parent
            idx = np.random.randint(len(archive_X))
            parent_x = archive_X[idx]
            
            # Mutate
            # Adaptive sigma? 
            # Let's use a fixed fraction of the range.
            sigma = 0.1 * (hi - lo)
            child_x = parent_x + np.random.normal(0, 1, self.dim) * sigma
            child_x = np.clip(child_x, lo, hi)
            
            child_f = problem(child_x)
            
            # Check dominance
            # Is child dominated by archive?
            dominated = False
            for j in range(len(archive_F)):
                if np.all(archive_F[j] <= child_f) and np.any(archive_F[j] < f):
                    dominated = True
                    break
            
            if not dominated:
                # Remove dominated points from archive
                new_archive_X = []
                new_archive_F = []
                for j in range(len(archive_F)):
                    if np.all(child_f <= archive_F[j]) and np.any(child_f < archive_F[j]):
                        continue
                    new_archive_X.append(archive_X[j])
                    new_archive_F.append(archive_F[j])
                
                new_archive_X.append(child_x)
                new_archive_F.append(child_f)
                archive_X = new_archive_X
                archive_F = new_archive_F
        
        return archive_X
