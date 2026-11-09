"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
from scipy.spatial.distance import cdist

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time

def construct():
    """
    Place 50 equal spheres in unit cube [-0.5, 0.5]^3 to maximize minimum distance.
    Strategy:
    1. Generate multiple diverse initial guesses (Fibonacci lattice, Halton, jittered grid, random).
    2. Run a repulsion-based pre-optimization to spread points.
    3. Refine using L-BFGS-B with a smoothed log-sum-exp objective (increasing beta).
    4. Polish using Nelder-Mead on the exact non-smooth objective.
    5. Local search: perturb individual points to escape local minima.
    6. Final high-precision polish pass.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin below 150s kill
    margin = 5.0
    effective_limit = time_limit - margin
    
    def check_time():
        return (time.time() - start_time) >= effective_limit
    
    def score_centres(centres):
        """Compute the radius (min of pairwise/2 and boundary distances)."""
        # Boundary distances
        boundary_dists = 0.5 - np.abs(centres)
        min_boundary = np.min(boundary_dists)
        
        # Pairwise distances
        if N > 1:
            dists = cdist(centres, centres)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists)
            min_pair_half = min_pair / 2.0
        else:
            min_pair_half = np.inf
        
        return min(min_boundary, min_pair_half)
    
    def soft_min_score(centres_flat, beta=100.0):
        """
        Smooth approximation of the negative radius.
        Objective to minimize: -soft_min(r_i)
        where r_i are the constraints (pairwise/2, boundary).
        """
        centres = centres_flat.reshape(N, 3)
        
        # Boundary terms: 0.5 - |c_ik|
        boundary_terms = 0.5 - np.abs(centres)  # (N, 3)
        
        # Pairwise distance terms: |c_i - c_j| / 2
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, np.inf)
        iu = np.triu_indices(N, k=1)
        pair_terms = dists[iu] / 2.0
        
        all_terms = np.concatenate([boundary_terms.flatten(), pair_terms])
        
        # Soft-min: -1/beta * log(sum(exp(-beta * x)))
        # To be stable, shift by max
        max_term = np.max(all_terms)
        exp_terms = np.exp(-beta * (all_terms - max_term))
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0:
            return 1e10
        soft_min_val = max_term - np.log(sum_exp) / beta
        
        return -soft_min_val

    def neg_score(centres_flat):
        """Negative exact score for direct methods."""
        centres = centres_flat.reshape(N, 3)
        if not np.all(np.isfinite(centres)):
            return 1e10
        return -score_centres(centres)
    
    def repulsion_objective(centres_flat, k=10.0):
        """
        Repulsion-based objective: minimize sum of -1/d^k over all pairs + boundary penalty.
        This spreads points apart before the main optimization.
        """
        centres = centres_flat.reshape(N, 3)
        
        # Pairwise repulsion
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, 1.0)  # Avoid division by zero
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        
        # Repulsion: sum of -1/d^k (want to minimize, so we want large d)
        repulsion = -np.sum(1.0 / (pair_dists ** k))
        
        # Boundary penalty: push points away from boundaries
        boundary_dists = 0.5 - np.abs(centres)
        boundary_penalty = -np.sum(1.0 / (boundary_dists + 0.01) ** 2)
        
        return repulsion + boundary_penalty

    def fibonacci_lattice(n_points, seed=0):
        """Generate n_points using Fibonacci lattice in the unit cube."""
        rng = np.random.default_rng(seed)
        golden_ratio = (1 + np.sqrt(5)) / 2
        points = []
        for i in range(n_points):
            # Fibonacci sphere parameter
            t = i / n_points
            z = 1 - 2 * t  # z in [-1, 1]
            radius = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / golden_ratio
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            # Scale to fit in [-0.5, 0.5]^3
            point = np.array([x, y, z]) * 0.45
            # Add small jitter
            point += rng.normal(0, 0.01, size=(3,))
            point = np.clip(point, -0.5, 0.5)
            points.append(point)
        return np.array(points)
    
    def halton_sequence(n, base):
        """Generate n points in the Halton sequence with given base."""
        result = []
        f = 1.0
        for i in range(1, n + 1):
            x = 0.0
            i_val = i
            while i_val > 0:
                f /= base
                x += f * (i_val % base)
                i_val //= base
            result.append(x)
        return np.array(result)
    
    def generate_jittered_grid(n_points, seed=0):
        """Generate a jittered grid of n_points in [-0.5, 0.5]^3."""
        rng = np.random.default_rng(seed)
        # Use a grid of 4x4x4 = 64 points, select 50
        grid_size = 4
        coords = np.linspace(-0.45, 0.45, grid_size)
        grid_points = np.array([(x, y, z) for x in coords for y in coords for z in coords])
        
        # Add jitter
        jitter = rng.uniform(-0.03, 0.03, size=(len(grid_points), 3))
        grid_points = grid_points + jitter
        
        # Select n_points points (shuffle and take first n)
        indices = rng.permutation(len(grid_points))[:n_points]
        return grid_points[indices]
    
    def generate_random_points(n_points, seed=0):
        """Generate random points in [-0.45, 0.45]^3."""
        rng = np.random.default_rng(seed)
        return rng.uniform(-0.45, 0.45, size=(n_points, 3))
    
    # Generate multiple diverse initial guesses
    rng_main = np.random.default_rng(42)
    
    # Initial guess 1: Fibonacci lattice
    initial1 = fibonacci_lattice(N, seed=1)
    
    # Initial guess 2: Halton sequence
    h1 = halton_sequence(N, 2)
    h2 = halton_sequence(N, 3)
    h3 = halton_sequence(N, 5)
    scale = 0.9
    initial2 = np.column_stack((
        h1 * scale - scale/2,
        h2 * scale - scale/2,
        h3 * scale - scale/2
    ))
    initial2 += rng_main.normal(0, 0.01, size=(N, 3))
    initial2 = np.clip(initial2, -0.5, 0.5)
    
    # Initial guess 3: Jittered grid
    initial3 = generate_jittered_grid(N, seed=1)
    
    # Initial guess 4: Another Halton with different bases
    h1b = halton_sequence(N, 3)
    h2b = halton_sequence(N, 5)
    h3b = halton_sequence(N, 7)
    initial4 = np.column_stack((
        h1b * scale - scale/2,
        h2b * scale - scale/2,
        h3b * scale - scale/2
    ))
    initial4 += rng_main.normal(0, 0.01, size=(N, 3))
    initial4 = np.clip(initial4, -0.5, 0.5)
    
    # Initial guess 5: Random
    initial5 = generate_random_points(N, seed=2)
    
    # Initial guess 6: Jittered grid with different seed
    initial6 = generate_jittered_grid(N, seed=3)
    
    # Initial guess 7: Fibonacci lattice with different seed
    initial7 = fibonacci_lattice(N, seed=7)
    
    initial_guesses = [initial1, initial2, initial3, initial4, initial5, initial6, initial7]
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_centres = None
    best_score = -np.inf
    
    # Process each initial guess
    for guess_idx, initial_centres in enumerate(initial_guesses):
        if check_time():
            break
        
        current_centres = initial_centres.copy()
        
        # Stage 0: Repulsion pre-optimization
        try:
            result_rep = minimize(
                repulsion_objective,
                current_centres.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 20}
            )
            if np.all(np.isfinite(result_rep.x)) and np.all(result_rep.x >= -0.5) and np.all(result_rep.x <= 0.5):
                current_centres = result_rep.x.reshape(N, 3)
        except Exception:
            pass
        
        # Stage 1: Smooth L-BFGS-B with increasing beta
        try:
            betas = [5.0, 20.0, 50.0, 100.0, 300.0]
            
            for beta in betas:
                if check_time():
                    break
                
                def smooth_obj(centres_flat, b=beta):
                    return soft_min_score(centres_flat, beta=b)
                
                result = minimize(
                    smooth_obj,
                    current_centres.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
                )
                
                if result.fun < smooth_obj(current_centres.flatten()):
                    new_centres = result.x.reshape(N, 3)
                    if np.all(np.isfinite(new_centres)) and np.all(new_centres >= -0.5) and np.all(new_centres <= 0.5):
                        new_score = score_centres(new_centres)
                        if new_score >= score_centres(current_centres) - 1e-10:
                            current_centres = new_centres
                            
        except Exception:
            pass
        
        # Stage 2: High Beta Polish
        try:
            for beta in [500.0, 1000.0, 2000.0, 4000.0]:
                if check_time():
                    break
                
                def smooth_obj_high(centres_flat, b=beta):
                    return soft_min_score(centres_flat, beta=b)
                
                result = minimize(
                    smooth_obj_high,
                    current_centres.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20}
                )
                
                if result.fun < smooth_obj_high(current_centres.flatten()):
                    new_centres = result.x.reshape(N, 3)
                    if np.all(np.isfinite(new_centres)) and np.all(new_centres >= -0.5) and np.all(new_centres <= 0.5):
                        new_score = score_centres(new_centres)
                        if new_score > score_centres(current_centres):
                            current_centres = new_centres
                            
        except Exception:
            pass
        
        # Stage 3: Nelder-Mead polish
        try:
            result_nm = minimize(
                neg_score,
                current_centres.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            
            new_centres = result_nm.x.reshape(N, 3)
            new_score = score_centres(new_centres)
            
            if np.all(np.isfinite(new_centres)) and np.all(new_centres >= -0.5) and np.all(new_centres <= 0.5):
                if new_score > score_centres(current_centres):
                    current_centres = new_centres
                    
        except Exception:
            pass
        
        # Stage 4: Local search - perturb individual points
        if not check_time():
            try:
                local_rng = np.random.default_rng(42 + guess_idx)
                current_score = score_centres(current_centres)
                
                for iteration in range(80):
                    if check_time():
                        break
                    
                    # Pick a random point to perturb
                    idx = local_rng.integers(N)
                    original = current_centres[idx].copy()
                    
                    # Small random perturbation
                    perturbation = local_rng.normal(0, 0.005, size=(3,))
                    current_centres[idx] = np.clip(original + perturbation, -0.5, 0.5)
                    
                    new_score = score_centres(current_centres)
                    
                    if new_score > current_score:
                        current_score = new_score
                    else:
                        # Revert
                        current_centres[idx] = original
                        
            except Exception:
                pass
        
        # Update best
        final_score = score_centres(current_centres)
        if np.isfinite(final_score) and final_score > best_score:
            best_score = final_score
            best_centres = current_centres.copy()
    
    # If we didn't get a valid result, use the first initial guess
    if best_centres is None:
        best_centres = initial1.copy()
        best_score = score_centres(best_centres)
    
    # Final Nelder-Mead polish on the best result
    if not check_time():
        try:
            result_final = minimize(
                neg_score,
                best_centres.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            
            new_centres = result_final.x.reshape(N, 3)
            new_score = score_centres(new_centres)
            
            if np.all(np.isfinite(new_centres)) and np.all(new_centres >= -0.5) and np.all(new_centres <= 0.5):
                if new_score > best_score:
                    best_centres = new_centres.copy()
                    best_score = new_score
                    
        except Exception:
            pass
    
    # Final validation
    final_score = score_centres(best_centres)
    if not np.isfinite(final_score):
        # Fallback
        best_centres = initial1.copy()
    
    return best_centres

# EVOLVE-BLOCK-END
