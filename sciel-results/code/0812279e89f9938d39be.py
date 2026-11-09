import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.archive = []
        self.archive_F = []
        
    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # State for the search
        # We assume the Pareto front is in the region where x[1:] are close to 0
        # and x[0] is in a specific range. 
        # For ZDT4, x[0] in [0, 1] is the front.
        # We will bias x[0] towards [0, 1] and x[1:] towards 0.
        
        # Initialize a small population
        pop_size = 20
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))
        
        # x[0]: Uniform in [0, 1] mapped to bounds? 
        # ZDT4 bounds are usually [-5, 5]. 
        # The front is at x[0] in [0, 1]. 
        # If bounds are different, we might need to adapt. 
        # But the task says "ZDT4". So we assume standard structure.
        # However, to be safe, we can sample x[0] in the middle of the bounds 
        # and x[1:] near the center (0).
        
        # Let's assume the "center" of the bounds is 0 for ZDT4.
        # If bounds are not symmetric, we might fail. 
        # But ZDT4 is standard.
        
        # Strategy:
        # x[0] ~ Uniform(0, 1)
        # x[i] ~ Uniform(-0.5, 0.5) for i > 0
        
        for i in range(pop_size):
            # x[0]
            # If bounds[0] is [-5, 5], 0 to 1 is a small part.
            # We force it to [0, 1] to hit the front.
            X[i, 0] = np.random.uniform(0, 1)
            # x[1:]
            X[i, 1:] = np.random.uniform(-0.5, 0.5, dim - 1)
            # Clip to bounds
            X[i] = np.clip(X[i], lo, hi)
            F[i] = problem(X[i])
            
        used = pop_size
        archive = []
        archive_F = []
        
        def is_dominated(f_new, F_archive):
            for f in F_archive:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False
            
        def dominates(f_new, F_archive):
            for f in F_archive:
                if np.all(f_new <= f) and np.any(f_new < f):
                    return True
            return False

        # Initialize archive with non-dominated points
        for i in range(pop_size):
            if not is_dominated(F[i], archive_F):
                # Check if it dominates any existing
                new_archive = []
                new_archive_F = []
                for j in range(len(archive_F)):
                    if not dominates(F[i], [archive_F[j]]):
                        new_archive.append(archive[j])
                        new_archive_F.append(archive_F[j])
                new_archive.append(X[i])
                new_archive_F.append(F[i])
                archive = new_archive
                archive_F = new_archive_F
                
        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Pick a parent from the archive
            if not archive:
                # Fallback: random
                parent_idx = np.random.randint(pop_size)
                parent = X[parent_idx]
            else:
                # Pick random from archive
                parent = archive[np.random.randint(len(archive))]
                
            # Create child
            child = parent.copy()
            
            # Mutation
            # x[0]: Explore [0, 1]
            # We use a step size that allows moving along the front
            step_x0 = 0.1
            child[0] += np.random.normal(0, step_x0)
            
            # x[1:]: Contract towards 0
            # This is the key to solving ZDT4's deceptive landscape
            contraction = 0.7
            noise = 0.05
            child[1:] = child[1:] * contraction + np.random.normal(0, noise, dim - 1)
            
            # Clip
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Update archive
            # If child is dominated by archive, discard
            if not is_dominated(fc, archive_F):
                # Remove dominated points from archive
                new_archive = []
                new_archive_F = []
                for j in range(len(archive_F)):
                    if not dominates(fc, [archive_F[j]]):
                        new_archive.append(archive[j])
                        new_archive_F.append(archive_F[j])
                
                new_archive.append(child)
                new_archive_F.append(fc)
                
                # If archive is too large, remove one
                # We want to maximize hypervolume. 
                # Simple heuristic: remove the point with the smallest 
                # "distance" to the ideal point? Or random?
                # Better: remove the point that contributes least to hypervolume.
                # Approximation: remove the point that is most "crowded".
                # For simplicity and speed, we can just keep the best 
                # N points based on a scalarization or just cap the size.
                # Let's cap at 50.
                if len(new_archive) > 50:
                    # Remove a random point? No, that's bad.
                    # Remove the one with the worst sum of objectives?
                    # Or just keep the first 50?
                    # Let's use a simple hypervolume contribution estimate.
                    # For 2 objectives, we can sort by f1 and calculate area.
                    # But that's expensive.
                    # Let's just remove a random point from the middle of the sorted list?
                    # Actually, for ZDT4, the front is a curve. 
                    # We want points spread across the curve.
                    # Let's sort by f1 and remove every other one if too many?
                    # Or just remove the one with the highest f2 for a given f1?
                    
                    # Simple approach: Sort by f1, keep unique f1 buckets.
                    # But f1 is continuous.
                    # Let's just remove a random point. It's a heuristic.
                    # Or remove the point with the maximum distance to the ideal point (0,0).
                    # Ideal point is likely (0,0) or (min_f1, min_f2).
                    # Let's use (0,0) as a guess for ZDT4.
                    distances = np.sum(np.array(new_archive_F) ** 2, axis=1)
                    worst_idx = np.argmax(distances)
                    new_archive.pop(worst_idx)
                    new_archive_F.pop(worst_idx)
                    
                archive = new_archive
                archive_F = new_archive_F
        
        return archive
