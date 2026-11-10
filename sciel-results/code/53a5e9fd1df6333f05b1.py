"""
Strategy: Structured Ring Packing + Global Optimization + Local Polish
1. Model the packing as concentric rings.
2. Use scipy.optimize.differential_evolution to optimize ring radii and angular shifts.
3. Refine with coordinate descent (local search) to adjust individual point positions.
"""
import numpy as np
import time
from scipy.optimize import differential_evolution

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def compute_score(pts):
        """
        Compute the minimum radius r for a given configuration.
        r = min(min_pair_dist / 2, 1 - max_dist_from_origin)
        Returns -1 if invalid (e.g., non-finite, out of bounds in a way that breaks logic, though score handles it).
        """
        if not np.all(np.isfinite(pts)):
            return -1.0
        
        # Pairwise distances
        # Use broadcasting for 300x300. 
        # diff shape (300, 300, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_dist = np.sqrt(np.min(dist_sq))
        
        # Boundary distance
        radii = np.sqrt(np.sum(pts ** 2, axis=1))
        max_radius = np.max(radii)
        
        r_pair = min_pair_dist / 2.0
        r_bound = 1.0 - max_radius
        
        return min(r_pair, r_bound)

    def generate_ring_config(radii, shifts):
        """
        Generate points from ring radii and angular shifts.
        radii: array of shape (K,)
        shifts: array of shape (K,)
        """
        pts = []
        for k in range(len(radii)):
            r = radii[k]
            if r < 1e-6:
                continue
            # Number of points on this ring
            # Approximation: circumference / min_spacing
            # But we need exactly N points total.
            # Strategy: Distribute N points among rings based on circumference.
            # This function assumes the caller has already decided how many points per ring?
            # No, DE optimizes radii. We need a fixed assignment of points to rings.
            # Let's fix the number of points per ring first.
            # Better: Let the function take explicit counts.
            pass
        return np.array(pts)

    def ring_config_to_points(radii, shifts, counts):
        """
        Convert ring parameters to cartesian points.
        radii: (K,)
        shifts: (K,)
        counts: (K,) integers
        """
        pts = []
        for k in range(len(radii)):
            n_k = int(counts[k])
            if n_k <= 0:
                continue
            r = radii[k]
            theta0 = shifts[k]
            thetas = theta0 + 2 * np.pi * np.arange(n_k) / n_k
            xs = r * np.cos(thetas)
            ys = r * np.cos(thetas + np.pi/2) # Wait, sin is better
            ys = r * np.sin(thetas)
            pts.append(np.column_stack((xs, ys)))
        
        if not pts:
            return np.zeros((N, 2))
            
        return np.vstack(pts)

    def objective_ring(params, counts):
        """
        Objective for DE. Minimizes negative score (maximizes score).
        params: flattened [radii..., shifts...]
        """
        K = len(counts)
        radii = params[:K]
        shifts = params[K:]
        
        # Ensure radii are positive and within bounds
        # DE bounds handle this, but let's be safe
        radii = np.clip(radii, 0.0, 1.0)
        
        pts = ring_config_to_points(radii, shifts, counts)
        if len(pts) != N:
            return 1.0 # Invalid
            
        score = compute_score(pts)
        if score < 0:
            return 1.0
        return -score

    def local_polish(pts, max_time):
        """
        Coordinate descent to improve the configuration.
        For each point, try moving it radially and angularly to increase the minimum clearance.
        """
        pts = pts.copy()
        current_score = compute_score(pts)
        best_score = current_score
        best_pts = pts.copy()
        
        # Precompute initial distances? No, update dynamically or recompute.
        # N=300 is small enough to recompute min_dist in O(N^2) per step if steps are few.
        # But we have 300 points. 300 * O(N^2) = 300 * 90000 = 27M ops per full sweep.
        # 145s budget. Python loops are slow. 
        # Vectorized local polish is tricky because each point's optimal move depends on neighbors.
        
        # Simpler local polish: Random perturbations + Accept if score increases.
        # This is essentially a very local SA.
        
        # Let's try a gradient-free local search:
        # For each point i:
        #   Propose small random moves.
        #   If score improves, accept.
        
        t_start = time.time()
        step_size = 0.01
        iterations = 0
        
        while time.time() < max_time:
            # Pick a random point
            i = np.random.randint(N)
            orig = pts[i].copy()
            
            # Try a few random perturbations
            improved = False
            for _ in range(5):
                # Random direction and magnitude
                theta = np.random.uniform(0, 2 * np.pi)
                mag = np.random.uniform(0, step_size)
                dx = mag * np.cos(theta)
                dy = mag * np.sin(theta)
                
                pts[i] = orig + np.array([dx, dy])
                
                # Check if it went out of bounds (1 - r) roughly? 
                # compute_score handles it, but if it goes far out, score drops.
                new_score = compute_score(pts)
                
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_pts = pts.copy()
                    improved = True
                    break
                else:
                    pts[i] = orig
            
            # Adaptive step size?
            if not improved and iterations > 100:
                step_size *= 0.95
            elif improved:
                step_size = min(step_size * 1.05, 0.05)
                
            iterations += 1
            
            # Periodic full recomputation to ensure consistency?
            # compute_score is deterministic.
            
        return best_pts, best_score

    # --- Main Logic ---
    
    # 1. Define Ring Structure
    # How many rings?
    # Optimal packing for 300 circles...
    # Rough estimate: Area of unit disk = pi. Area of 1 circle = pi*r^2.
    # 300 * pi * r^2 < pi => r < 1/sqrt(300) ~ 0.0577.
    # Known best is 0.0531.
    
    # Let's try a few different ring count configurations.
    # Configuration A: 1 center + 6 rings?
    # Configuration B: 10 rings?
    
    # We will search over a few plausible ring counts.
    # For each ring count K, we determine how many points go on each ring.
    # A common heuristic: number of points on ring k is proportional to radius r_k.
    # But we need integers summing to N.
    
    # Let's fix the counts for a given K.
    # Strategy: Start with 1 point at center (if K>0 and we allow r=0).
    # Actually, r=0 is a single point.
    # Let's try K=10 rings.
    # Counts: distribute 300 points.
    # Outer rings have more space.
    # Approximate count on ring k (radius r_k) is ~ 2*pi*r_k / (2*r).
    # This is circular.
    
    # Let's just try a few fixed distributions.
    # Dist 1: Uniform-ish? No, circumference grows.
    # Let's use a heuristic: counts[k] proportional to k (assuming equal radial spacing).
    # If radii are roughly equal spacing, r_k ~ k. Count ~ k.
    # Sum k from 1 to K = K(K+1)/2.
    # If K=24, Sum = 300. Perfect.
    # So K=24 rings, counts = [1, 2, ..., 24]?
    # Wait, ring 0 is center?
    # If K=24, and we use rings 1..24, counts 1..24 sum to 300.
    # But the center point (ring 0) is not included.
    # So we have 24 rings, no center point?
    # Or 1 center + 23 rings?
    # Let's try:
    # Option 1: 1 center + rings with counts summing to 299.
    # Option 2: No center, rings with counts summing to 300.
    
    # Let's try Option 2 with K=24, counts = 1..24.
    # Radii are optimized.
    
    best_overall_score = 0.0
    best_overall_pts = None
    
    # Time budget for DE
    t_de_start = time.time()
    t_de_budget = 60.0 # 60 seconds for global search
    
    # DE Bounds for radii: [0.01, 0.95]
    # DE Bounds for shifts: [0, 2*pi]
    
    # We will run DE for a specific configuration.
    # Let's try K=24, counts=1..24.
    
    K = 24
    counts = np.arange(1, K + 1)
    
    # Initialize DE
    # Bounds:
    # radii: (0.01, 0.95)
    # shifts: (0, 2*np.pi)
    
    bounds = [(0.01, 0.95)] * K + [(0, 2 * np.pi)] * K
    
    # To save time, we might want to limit the population size and iterations.
    # But DE is robust.
    # We need to ensure it finishes in time.
    # Let's wrap it in a time check.
    
    try:
        # maxiter=100, popsize=10 might be too few for 48 dims.
        # Let's try popsize=15, maxiter=50.
        # 48 dims * 15 * 50 = 36000 evaluations.
        # Each eval is O(N^2) = 90000 ops.
        # 3.6M * 90k = 324 billion ops? No.
        # 36000 evals * 90000 flops = 3.24 * 10^9 flops.
        # NumPy is fast, but Python overhead in objective function is high.
        # 36000 calls to compute_score.
        # compute_score takes ~1-2ms?
        # 36000 * 2ms = 72 seconds.
        # This might be too slow for 60s budget.
        
        # Optimization: Vectorize compute_score?
        # It is already vectorized.
        # Let's reduce iterations or population.
        # Or, precompute some parts?
        
        # Let's try maxiter=30, popsize=10.
        # 3000 evals. 3000 * 2ms = 6 seconds.
        # This is safer.
        
        res = differential_evolution(
            objective_ring, 
            bounds, 
            args=(counts,),
            maxiter=30, 
            popsize=10, 
            tol=1e-4, 
            seed=42,
            polish=False # We will do our own polish
        )
        
        if res.success or res.fun < 0: # res.fun is -score
            params = res.x
            radii = params[:K]
            shifts = params[K:]
            pts = ring_config_to_points(radii, shifts, counts)
            score = compute_score(pts)
            if score > best_overall_score:
                best_overall_score = score
                best_overall_pts = pts
                
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to heuristic initialization
        radii = np.linspace(0.1, 0.9, K)
        shifts = np.zeros(K)
        pts = ring_config_to_points(radii, shifts, counts)
        score = compute_score(pts)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_pts = pts

    # If DE didn't find anything good (e.g. timeout or failure), use a heuristic start
    if best_overall_pts is None:
        # Heuristic: Hexagonal lattice clipped to disk
        r_hex = 0.05
        spacing = 2.0 * r_hex
        k = int(1.0 / spacing) + 2
        pts_hex = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - r_hex:
                    pts_hex.append((x, y))
        pts_hex = np.array(pts_hex)
        if len(pts_hex) >= N:
            dists = np.hypot(pts_hex[:, 0], pts_hex[:, 1])
            idx = np.argsort(dists, kind="stable")[:N]
            best_overall_pts = pts_hex[idx]
            best_overall_score = compute_score(best_overall_pts)
        else:
            # Fallback random
            best_overall_pts = np.random.uniform(-0.9, 0.9, (N, 2))
            best_overall_score = compute_score(best_overall_pts)

    # 2. Local Polish
    # Use remaining time for local polish
    t_polish_start = time.time()
    t_polish_budget = deadline - t_polish_start - 5.0 # 5s safety
    
    if t_polish_budget > 5:
        best_pts, best_score = local_polish(best_overall_pts, t_polish_budget)
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall_pts = best_pts

    # Final Check
    # Ensure validity
    if not np.all(np.isfinite(best_overall_pts)):
        return np.zeros((N, 2))
        
    # One last quick check if we have time
    if time.time() < deadline - 2:
        # Maybe a few more random kicks?
        # local_polish already did this.
        pass

    return best_overall_pts

# EVOLVE-BLOCK-END
