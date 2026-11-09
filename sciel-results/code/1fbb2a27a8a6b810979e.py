"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin from 150s wall-clock limit
    
    def compute_diameter(points):
        """Compute the maximum distance (diameter) of the point set."""
        n = len(points)
        pts = np.array(points)
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        return np.max(dists)
    
    def score_points(points):
        """
        Compute Delta/n^n where Delta is the product of distances over all ordered pairs
        after scaling the set to have diameter 2.
        """
        n = len(points)
        pts = np.array(points, dtype=float)
        
        # Check for duplicates
        for i in range(n):
            for j in range(i + 1, n):
                if np.allclose(pts[i], pts[j], atol=1e-10):
                    return 0.0
        
        # Compute diameter
        diameter = compute_diameter(pts)
        if diameter < 1e-12:
            return 0.0
        
        # Scale to diameter 2
        scale_factor = 2.0 / diameter
        scaled_pts = pts * scale_factor
        
        # Compute product of distances over ordered pairs
        # Ordered pairs means we count both (i,j) and (j,i) for i != j
        # But typically "product of distances" in this context means over all unordered pairs,
        # or it could mean all ordered pairs. Let me re-read the task.
        
        # "Delta is the product over ordered pairs of distances"
        # So for 8 points, there are 8*7 = 56 ordered pairs (i != j)
        # But wait, if we include ordered pairs, then d_ij * d_ji = d_ij^2, 
        # so the product would be (product over unordered pairs) ^ 2
        
        # Actually, let me reconsider. In many such problems, "product of distances" 
        # refers to the product over all unordered pairs. Let me check the regular octagon.
        
        # For a regular octagon with diameter 2 (so radius 1/sin(pi/8) * something),
        # let's compute.
        
        # Actually, let's just compute it as the product over all unordered pairs (i < j),
        # and see if the regular octagon gives score 1.
        
        # Product over unordered pairs
        log_prod = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((scaled_pts[i] - scaled_pts[j]) ** 2))
                if d < 1e-15:
                    return 0.0
                log_prod += np.log(d)
        
        # The score is Delta / n^n where Delta is the product
        # Let's compute: score = exp(log_prod) / n^n
        score = np.exp(log_prod - n * np.log(n))
        return score
    
    def neg_log_score(points):
        """Negative log of the score for optimization (to minimize)."""
        n = len(points)
        pts = np.array(points, dtype=float).reshape(-1, 2)
        
        # Check for duplicates
        for i in range(n):
            for j in range(i + 1, n):
                if np.allclose(pts[i], pts[j], atol=1e-10):
                    return 1e10
        
        # Compute diameter
        diameter = compute_diameter(pts)
        if diameter < 1e-12:
            return 1e10
        
        # Scale to diameter 2
        scale_factor = 2.0 / diameter
        scaled_pts = pts * scale_factor
        
        # Compute log of product over unordered pairs
        log_prod = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((scaled_pts[i] - scaled_pts[j]) ** 2))
                if d < 1e-15:
                    return 1e10
                log_prod += np.log(d)
        
        # Log score = log_prod - n*log(n)
        # We want to maximize score, so minimize -log_score
        return -(log_prod - n * np.log(n))
    
    # Start with regular octagon
    angles = [2 * np.pi * k / 8 for k in range(8)]
    best_points = np.array([[np.cos(a), np.sin(a)] for a in angles])
    best_score = score_points(best_points.tolist())
    
    # Try to improve using gradient-free optimization on angles
    # Use a perturbation approach with simulated annealing-like strategy
    
    current_points = best_points.copy()
    current_score = best_score
    
    # Define a set of perturbation strategies
    def perturb(points, magnitude):
        """Apply random perturbations to points."""
        perturbed = points.copy()
        noise = np.random.normal(0, magnitude, size=perturbed.shape)
        perturbed = perturbed + noise
        return perturbed
    
    # Optimization loop
    step_size = 0.1
    best_iter = 0
    iter_count = 0
    
    while time.time() < deadline:
        iter_count += 1
        
        # Generate a perturbation
        magnitude = step_size * np.exp(-iter_count / 2000.0)
        candidate = perturb(current_points, magnitude)
        
        # Ensure no duplicates
        valid = True
        for i in range(8):
            for j in range(i + 1, 8):
                if np.allclose(candidate[i], candidate[j], atol=1e-10):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        candidate_score = score_points(candidate.tolist())
        
        # Accept if better, or with some probability (simulated annealing)
        if candidate_score > current_score:
            current_points = candidate
            current_score = candidate_score
            if current_score > best_score:
                best_points = candidate.copy()
                best_score = current_score
                best_iter = iter_count
                # Increase step size slightly when we find improvement
                step_size = min(step_size * 1.1, 0.5)
        else:
            # Sometimes accept worse solutions to escape local optima
            if iter_count < 1000 and np.random.random() < 0.1:
                current_points = candidate
                current_score = candidate_score
        
        # Decrease step size over time
        if iter_count % 100 == 0:
            step_size *= 0.95
    
    # Convert back to list of lists
    return best_points.tolist()
# EVOLVE-BLOCK-END
