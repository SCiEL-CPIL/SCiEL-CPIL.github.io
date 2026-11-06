"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def construct():
    """
    Find 8 points maximizing the product of pairwise distances normalized by diameter^28.
    Strategy:
    1. Global search using Simulated Annealing (SA) and Particle Swarm Optimization (PSO).
    2. Objective: Maximize log(product of distances) subject to diameter <= 2.
    3. Hard constraint handling: After each move, if diameter > 2, scale down to 2.
       This keeps points on the feasible boundary, which is where the maximum lies.
    4. Diverse initial seeds based on high-symmetry configurations (concentric squares, etc.)
       to avoid the regular octagon local optimum.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    n = 8
    
    def get_diameter(pts):
        """Compute the maximum pairwise distance."""
        max_d = 0.0
        # Vectorized computation for speed
        # pts is (n, 2)
        # dist_sq[i, j] = ||p_i - p_j||^2
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        # Mask diagonal
        np.fill_diagonal(dist_sq, 0)
        max_d = np.sqrt(np.max(dist_sq))
        return max_d

    def log_product(pts):
        """
        Compute log of product of all pairwise distances.
        Returns -inf if any distance is 0.
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 0)
        
        # Check for zeros
        if np.any(dist_sq < 1e-12):
            return -1e9
        
        # log(product) = sum(log(dist)) = 0.5 * sum(log(dist_sq))
        # Only sum upper triangle to avoid double counting? 
        # The problem definition says "product over ordered pairs".
        # Ordered pairs (i, j) where i != j.
        # So product = prod_{i != j} ||p_i - p_j||.
        # log_product = sum_{i != j} log(||p_i - p_j||)
        # This is 2 * sum_{i < j} log(||p_i - p_j||)
        
        # dist_sq contains squared distances.
        # We want sum of log(sqrt(dist_sq)) over all i != j.
        # = 0.5 * sum(log(dist_sq)) over i != j.
        
        # dist_sq has 0s on diagonal. log(0) is -inf.
        # We should mask diagonal.
        mask = ~np.eye(n, dtype=bool)
        valid_dist_sq = dist_sq[mask]
        
        if np.any(valid_dist_sq < 1e-12):
            return -1e9
            
        return 0.5 * np.sum(np.log(valid_dist_sq))

    def normalize(pts):
        """Scale points so that diameter is exactly 2."""
        d = get_diameter(pts)
        if d < 1e-9:
            return pts
        return pts * (2.0 / d)

    def evaluate(pts):
        """
        Calculate the score: Delta / n^n.
        Delta is product of distances for set scaled to diameter 2.
        Score = exp(log_product_scaled) / 8^8.
        We maximize log_score = log_product_scaled - 8*log(8).
        """
        pts_norm = normalize(pts)
        lp = log_product(pts_norm)
        if lp == -1e9:
            return -1e9
        log_score = lp - 8 * np.log(8)
        return log_score

    def project(pts):
        """
        Hard projection: If diameter > 2, scale down.
        This ensures we stay in the feasible region.
        """
        d = get_diameter(pts)
        if d > 2.0:
            return pts * (2.0 / d)
        return pts

    def simulated_annealing(initial_pts, iterations=2000):
        """
        SA to maximize log_product subject to diameter <= 2.
        """
        current_pts = initial_pts.copy()
        current_pts = project(current_pts)
        current_score = evaluate(current_pts)
        
        best_pts = current_pts.copy()
        best_score = current_score
        
        T_start = 0.5
        T_end = 0.01
        cooling = (T_end / T_start) ** (1.0 / iterations)
        
        for i in range(iterations):
            if time.time() > deadline:
                break
                
            T = T_start * (cooling ** i)
            
            # Perturb: Add Gaussian noise
            # Step size should be adaptive or fixed small
            # Since we project to diameter 2, the scale is fixed.
            # A step size of ~0.1-0.2 relative to diameter seems reasonable.
            noise = np.random.normal(0, 0.2, size=current_pts.shape)
            new_pts = current_pts + noise
            new_pts = project(new_pts)
            
            new_score = evaluate(new_pts)
            
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
            else:
                # Accept with probability exp((new_score - current_score) / T)
                # Note: We are maximizing, so delta = new - current < 0.
                # exp(delta / T)
                delta = new_score - current_score
                if delta < 0:
                    prob = math.exp(delta / T)
                    if np.random.rand() < prob:
                        current_pts = new_pts
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_pts = current_pts.copy()
        
        return best_pts, best_score

    def particle_swarm(initial_pts, iterations=1000):
        """
        PSO to maximize log_product.
        """
        # Initialize swarm
        swarm_size = 5
        particles = np.zeros((swarm_size, n, 2))
        velocities = np.zeros((swarm_size, n, 2))
        p_best = np.zeros((swarm_size, n, 2))
        p_best_scores = np.full(swarm_size, -1e9)
        
        g_best = initial_pts.copy()
        g_best_score = evaluate(g_best)
        
        # Initialize particles around initial point
        for i in range(swarm_size):
            noise = np.random.normal(0, 0.5, size=(n, 2))
            p = initial_pts + noise
            p = project(p)
            particles[i] = p
            velocities[i] = np.random.normal(0, 0.1, size=(n, 2))
            
            score = evaluate(p)
            if score > p_best_scores[i]:
                p_best[i] = p
                p_best_scores[i] = score
            if score > g_best_score:
                g_best = p
                g_best_score = score
                
        w = 0.9 # Inertia
        c1 = 1.5 # Cognitive
        c2 = 1.5 # Social
        
        for i in range(iterations):
            if time.time() > deadline:
                break
                
            for k in range(swarm_size):
                # Update velocity
                r1 = np.random.rand(n, 2)
                r2 = np.random.rand(n, 2)
                
                vel = w * velocities[k] + c1 * r1 * (p_best[k] - particles[k]) + c2 * r2 * (g_best - particles[k])
                
                # Limit velocity
                max_vel = 0.5
                vel_norm = np.linalg.norm(vel, axis=1, keepdims=True)
                mask = vel_norm > max_vel
                vel[mask] = vel[mask] * (max_vel / vel_norm[mask])
                
                velocities[k] = vel
                
                # Update position
                new_pos = particles[k] + vel
                new_pos = project(new_pos)
                
                new_score = evaluate(new_pos)
                
                if new_score > p_best_scores[k]:
                    p_best[k] = new_pos
                    p_best_scores[k] = new_score
                    if new_score > g_best_score:
                        g_best = new_pos.copy()
                        g_best_score = new_score
                        
                particles[k] = new_pos
                
        return g_best, g_best_score

    # Generate diverse initial configurations
    configs = []
    
    # 1. Regular Octagon (Baseline, though likely local max)
    configs.append(np.array([[np.cos(2*np.pi*k/8), np.sin(2*np.pi*k/8)] for k in range(8)]))
    
    # 2. Two Concentric Squares
    # Outer square radius 1, inner square radius r
    # Rotate inner by 45 deg
    for r in [0.3, 0.5, 0.7, 0.8]:
        outer = np.array([[np.cos(2*np.pi*k/4), np.sin(2*np.pi*k/4)] for k in range(4)])
        inner = np.array([[r*np.cos(2*np.pi*k/4 + np.pi/4), r*np.sin(2*np.pi*k/4 + np.pi/4)] for k in range(4)])
        cfg = np.vstack([outer, inner])
        # Normalize to diameter 2
        cfg = normalize(cfg)
        configs.append(cfg)
        
    # 3. Square with midpoints (closer to center)
    s = 1.0
    sq = np.array([[-s/2, -s/2], [s/2, -s/2], [s/2, s/2], [-s/2, s/2]])
    mids = np.array([[0, -s/2], [s/2, 0], [0, s/2], [-s/2, 0]])
    cfg = np.vstack([sq, mids])
    cfg = normalize(cfg)
    configs.append(cfg)
    
    # 4. Random Perturbations of Octagon
    np.random.seed(42)
    for _ in range(5):
        octa = np.array([[np.cos(2*np.pi*k/8), np.sin(2*np.pi*k/8)] for k in range(8)])
        noise = np.random.normal(0, 0.3, size=(8, 2))
        cfg = octa + noise
        cfg = normalize(cfg)
        configs.append(cfg)
        
    # 5. "Butterfly" / Two Squares
    for r_inner in [0.5, 0.8]:
        p1 = np.array([[r_inner, 0], [-r_inner, 0], [0, r_inner], [0, -r_inner]]) # Diamond
        p2 = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) / np.sqrt(2) # Square
        cfg = np.vstack([p1, p2])
        cfg = normalize(cfg)
        configs.append(cfg)
        
    # 6. Random Points on Circle (various radii)
    for _ in range(5):
        angles = np.sort(np.random.rand(8) * 2 * np.pi)
        radii = np.random.uniform(0.5, 1.0, 8)
        cfg = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        cfg = normalize(cfg)
        configs.append(cfg)

    best_pts_global = configs[0]
    best_score_global = evaluate(best_pts_global)
    
    # Run SA and PSO on each config
    # To save time, we might not run both on all. 
    # SA is robust, PSO is fast for local refinement.
    
    for i, cfg in enumerate(configs):
        if time.time() > deadline:
            break
            
        # Run SA
        pts_sa, score_sa = simulated_annealing(cfg, iterations=1000)
        if score_sa > best_score_global:
            best_score_global = score_sa
            best_pts_global = pts_sa
            
        # Run PSO starting from the SA result (or original if SA failed)
        start_pso = pts_sa if score_sa > evaluate(cfg) else cfg
        pts_pso, score_pso = particle_swarm(start_pso, iterations=500)
        if score_pso > best_score_global:
            best_score_global = score_pso
            best_pts_global = pts_pso

    # Final check: ensure diameter is exactly 2 for the returned points
    d_final = get_diameter(best_pts_global)
    if d_final > 1e-6:
        best_pts_global = best_pts_global * (2.0 / d_final)
        
    # Convert to list of lists of floats
    return [[float(x), float(y)] for x, y in best_pts_global]
# EVOLVE-BLOCK-END
