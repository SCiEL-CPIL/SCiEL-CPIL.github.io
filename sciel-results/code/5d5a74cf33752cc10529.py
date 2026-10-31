"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def score(pts):
        """
        Calculate the squared ratio of max to min pairwise distance.
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        d_min = d.min()
        d_max = d.max()
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def normalize(pts):
        """Center and scale points to unit diameter."""
        pts = pts.copy()
        pts -= pts.mean(axis=0)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        max_d = dists.max()
        if max_d > 1e-12:
            pts /= max_d
        return pts

    def initial_config_hex_ring():
        """
        Create an initial configuration with a central point,
        a hexagonal ring of 6, and an outer ring of 9.
        This provides a structured starting point with good symmetry.
        """
        pts = np.zeros((N, 2))
        # Center
        pts[0] = [0, 0]
        
        # Inner hexagon (6 points)
        r1 = 1.0
        for i in range(6):
            angle = 2 * np.pi * i / 6
            pts[1 + i] = [r1 * np.cos(angle), r1 * np.sin(angle)]
        
        # Outer ring (9 points)
        r2 = 1.8
        for i in range(9):
            angle = 2 * np.pi * i / 9 + np.pi / 9
            pts[7 + i] = [r2 * np.cos(angle), r2 * np.sin(angle)]
            
        return normalize(pts)

    def initial_config_grid():
        """
        Create a 4x4 grid, slightly perturbed to break symmetry.
        """
        pts = np.zeros((N, 2))
        idx = 0
        for i in range(4):
            for j in range(4):
                pts[idx, 0] = j
                pts[idx, 1] = i
                idx += 1
        # Add small noise to break perfect symmetry
        pts += np.random.default_rng(42).normal(0, 0.1, (N, 2))
        return normalize(pts)

    def cma_es_step(pop, mean, sigma, scores, rng):
        """
        Perform one step of a simple CMA-ES-like evolution.
        Returns new population, new mean, new sigma.
        """
        lam = len(pop)
        mu = lam // 2  # Number of parents
        
        # Sort by score (lower is better)
        idx = np.argsort(scores)
        parents = pop[idx[:mu]]
        parent_scores = scores[idx[:mu]]
        
        # Weighted mean
        weights = np.arange(1, mu + 1)
        weights = weights / weights.sum()
        new_mean = np.sum(weights[:, np.newaxis, np.newaxis] * parents, axis=0)
        
        # Update sigma (simple adaptation)
        # If improvement is small, decrease sigma; if large, increase
        old_mean_score = scores.mean()
        new_mean_score = scores[idx[:mu]].mean()
        if new_mean_score < old_mean_score:
            sigma *= 1.05
        else:
            sigma *= 0.95
            
        sigma = np.clip(sigma, 1e-6, 1.0)
        
        # Generate offspring
        new_pop = []
        for _ in range(lam):
            # Pick two parents
            i = rng.integers(mu)
            j = rng.integers(mu)
            while j == i:
                j = rng.integers(mu)
            
            child = (parents[i] + parents[j]) / 2.0 + sigma * rng.normal(0, 1, (N, 2))
            child = normalize(child)
            new_pop.append(child)
            
        return np.array(new_pop), new_mean, sigma

    # Main loop
    rng = np.random.default_rng(123)
    
    # Initialize population with diverse starting points
    pop = []
    pop.append(initial_config_hex_ring())
    pop.append(initial_config_grid())
    
    # Add random perturbations of the best initial
    for _ in range(8):
        p = pop[0] + rng.normal(0, 0.2, (N, 2))
        pop.append(normalize(p))
        
    # Add fully random
    for _ in range(1):
        p = rng.normal(0, 1, (N, 2))
        pop.append(normalize(p))
        
    pop = np.array(pop)
    scores = np.array([score(p) for p in pop])
    
    best_pts = pop[np.argmin(scores)]
    best_score = scores.min()
    
    mean = pop.mean(axis=0)
    sigma = 0.1
    
    stagnation_counter = 0
    max_stagnation = 20
    
    while time.time() < deadline:
        # Evaluate population
        scores = np.array([score(p) for p in pop])
        current_best_idx = np.argmin(scores)
        
        if scores[current_best_idx] < best_score:
            best_score = scores[current_best_idx]
            best_pts = pop[current_best_idx].copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        # If stagnant, diversify
        if stagnation_counter > max_stagnation:
            # Replace worst half with random perturbations of best
            idx = np.argsort(scores)
            for i in range(len(pop) // 2):
                worst_idx = idx[len(pop) - 1 - i]
                pop[worst_idx] = best_pts + rng.normal(0, 0.5, (N, 2))
                pop[worst_idx] = normalize(pop[worst_idx])
            stagnation_counter = 0
            max_stagnation = max(10, max_stagnation // 2)
        
        # CMA-ES step
        pop, mean, sigma = cma_es_step(pop, mean, sigma, scores, rng)
        
        # Check time
        if time.time() > deadline - 5:
            break

    return best_pts

# EVOLVE-BLOCK-END
