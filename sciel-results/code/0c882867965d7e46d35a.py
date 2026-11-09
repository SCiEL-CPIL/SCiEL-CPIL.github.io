import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Time management
    start_time = time.time()
    time_limit = 200.0  # Safety margin (evaluator kills at 230s)
    
    # Fast approximate plane-wave solver
    # Use a smaller G-set for speed: 9x9 = 81 plane waves
    N_G_fast = 9
    g_range = np.arange(-N_G_fast//2, N_G_fast//2 + 1)
    Gx, Gy = np.meshgrid(g_range, g_range, indexing='ij')
    G_vectors_fast = np.column_stack((Gx.ravel(), Gy.ravel()))
    N_pw_fast = len(G_vectors_fast)
    
    # K points - use fewer for speed
    k_points = [
        [0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.25, 0.0], [0.5, 0.25],
        [0.25, 0.25], [0.4, 0.1], [0.3, 0.15], [0.45, 0.05], [0.35, 0.2],
        [0.4, 0.3], [0.2, 0.1], [0.45, 0.45], [0.3, 0.3]
    ]
    k_points = np.array(k_points)
    
    # Precompute G^2
    G_sq_fast = np.sum(G_vectors_fast**2, axis=1)
    
    # Precompute index mapping for G differences
    # G_diff components are in [-(N_G_fast-1), (N_G_fast-1)]
    # Index: (dx + offset) + (dy + offset) * N_G_fast
    offset_fast = N_G_fast // 2
    
    def get_fourier_eps_inv_fast(cell_arr):
        """Get Fourier coefficients of 1/eps using FFT"""
        eps = np.where(cell_arr == 1, eps_high, eps_low)
        eps_inv = 1.0 / eps
        coeffs = np.fft.fft2(eps_inv) / (size * size)
        # Map to our G grid
        offset = size // 2
        Gx_grid = G_vectors_fast[:, 0] + offset
        Gy_grid = G_vectors_fast[:, 1] + offset
        return coeffs[Gx_grid, Gy_grid]
    
    def evaluate_fast(cell_arr):
        """Fast approximate evaluation using smaller G-set"""
        eps_inv_coeffs = get_fourier_eps_inv_fast(cell_arr)
        
        min_omega_5 = float('inf')
        max_omega_4 = float('-inf')
        
        for k in k_points:
            k_vec = np.array(k)
            k_plus_G = k_vec[None, :] + G_vectors_fast
            k_plus_G_sq = np.sum(k_plus_G**2, axis=1)
            
            # Build H matrix efficiently
            H = np.zeros((N_pw_fast, N_pw_fast), dtype=complex)
            
            for i in range(N_pw_fast):
                Gi = G_vectors_fast[i]
                diffs = Gi[None, :] - G_vectors_fast
                dx = diffs[:, 0]
                dy = diffs[:, 1]
                mask = (np.abs(dx) <= offset_fast) & (np.abs(dy) <= offset_fast)
                if np.any(mask):
                    idxs = (dx[mask] + offset_fast) + (dy[mask] + offset_fast) * N_G_fast
                    coeffs = eps_inv_coeffs[idxs]
                    H[i, np.where(mask)[0]] = coeffs * k_plus_G_sq[np.where(mask)[0]]
            
            evals = np.linalg.eigvals(H)
            evals = np.sort(np.real(evals))
            
            w4 = evals[3]
            w5 = evals[4]
            
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
        
        if min_omega_5 == float('inf'):
            return 0.0
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap
    
    # Symmetry group operations for 32x32 cell
    # Operations: identity, 90deg, 180deg, 270deg, and their compositions with reflections
    def apply_symmetry(cell_arr):
        """Apply all symmetry operations and average (for continuous) or enforce (for binary)"""
        N = size
        # Generate all 8 symmetry operations
        ops = [
            lambda x, y: (x, y),           # identity
            lambda x, y: (N-1-y, x),       # 90deg CW
            lambda x, y: (N-1-x, N-1-y),   # 180deg
            lambda x, y: (y, N-1-x),       # 270deg CW
            lambda x, y: (x, N-1-y),       # reflect y
            lambda x, y: (N-1-y, N-1-x),   # reflect x
            lambda x, y: (y, x),           # transpose
            lambda x, y: (N-1-x, y),       # reflect anti-diag
        ]
        
        # For binary: majority vote
        result = np.zeros_like(cell_arr)
        for op in ops:
            transformed = np.zeros_like(cell_arr)
            for i in range(N):
                for j in range(N):
                    ri, rj = op(i, j)
                    transformed[i, j] = cell_arr[ri, rj]
            result += transformed
        
        # Majority vote
        result = (result > N/2).astype(int)
        return result
    
    def get_orbit(i, j):
        """Get all pixels in the symmetry orbit of (i, j)"""
        N = size
        points = set()
        ops = [
            lambda x, y: (x, y),
            lambda x, y: (N-1-y, x),
            lambda x, y: (N-1-x, N-1-y),
            lambda x, y: (y, N-1-x),
            lambda x, y: (x, N-1-y),
            lambda x, y: (N-1-y, N-1-x),
            lambda x, y: (y, x),
            lambda x, y: (N-1-x, y),
        ]
        for op in ops:
            points.add(op(i, j))
        return list(points)
    
    # Initialize with a good seed: 4-circle cluster
    cell = np.zeros((size, size), dtype=int)
    centers = [(8, 8), (8, 23), (23, 8), (23, 23)]
    radius = 3
    for i in range(size):
        for j in range(size):
            for cx, cy in centers:
                if (i - cx)**2 + (j - cy)**2 < radius**2:
                    cell[i, j] = 1
                    break
    
    # Ensure symmetry
    cell = apply_symmetry(cell)
    
    # Evaluate initial
    best_score = evaluate_fast(cell)
    best_cell = cell.copy()
    
    # Genetic algorithm
    pop_size = 16
    mutation_rate = 0.3
    
    # Initialize population
    population = [best_cell.copy()]
    for _ in range(pop_size - 1):
        mutant = best_cell.copy()
        for _ in range(3):
            i = np.random.randint(size)
            j = np.random.randint(size)
            orbit = get_orbit(i, j)
            for (r, c) in orbit:
                mutant[r, c] = 1 - mutant[r, c]
        mutant = apply_symmetry(mutant)
        population.append(mutant)
    
    # Evaluate initial population
    pop_scores = [evaluate_fast(ind) for ind in population]
    
    generation = 0
    while time.time() - start_time < time_limit - 30:  # Leave 30s buffer
        generation += 1
        
        # Check time
        if time.time() - start_time > time_limit - 30:
            break
        
        # Sort by score
        idx_sorted = np.argsort(pop_scores)[::-1]
        population = [population[i].copy() for i in idx_sorted]
        pop_scores = [pop_scores[i] for i in idx_sorted]
        
        # Elitism: keep top half
        elite_size = pop_size // 2
        new_population = [population[i].copy() for i in range(elite_size)]
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() - start_time > time_limit - 30:
                break
            
            # Crossover (uniform)
            if np.random.random() < 0.5:
                parent1 = population[np.random.randint(elite_size)]
                parent2 = population[np.random.randint(elite_size)]
                mask = np.random.randint(0, 2, (size, size))
                child = np.where(mask == 1, parent1, parent2)
            else:
                parent1 = population[np.random.randint(elite_size)]
                child = parent1.copy()
            
            # Mutation
            n_mutations = np.random.poisson(mutation_rate * size * size / 100)
            for _ in range(n_mutations):
                i = np.random.randint(size)
                j = np.random.randint(size)
                orbit = get_orbit(i, j)
                for (r, c) in orbit:
                    child[r, c] = 1 - child[r, c]
            
            # Enforce symmetry
            child = apply_symmetry(child)
            new_population.append(child)
        
        # Evaluate new population
        new_scores = [evaluate_fast(ind) for ind in new_population]
        
        # Update best
        max_idx = np.argmax(new_scores)
        if new_scores[max_idx] > best_score:
            best_score = new_scores[max_idx]
            best_cell = new_population[max_idx].copy()
        
        # Combine and select
        population = new_population
        pop_scores = new_scores
    
    # Return best_cell as nested list
    return best_cell.tolist()
