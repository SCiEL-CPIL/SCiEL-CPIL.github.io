"""Differential Evolution based circle packing in perimeter-4 rectangle."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def evaluate(x):
        """
        x: array of shape (43,)
        x[0]: W (width)
        x[1..42]: 21 * 2 circle centers (x, y)
        
        Returns: sum of radii
        """
        W = x[0]
        H = 2.0 - W
        if W < 0.1 or W > 1.9:
            return -1e9
        
        centers = x[1:].reshape(-1, 2)
        
        # Compute radii for each circle
        # r_i = min(
        #   x_i, W - x_i, y_i, H - y_i,
        #   min_{j != i} (dist(i, j) / 2)
        # )
        
        radii = np.zeros(N)
        
        # Distance to walls
        wall_dists = np.minimum(
            np.minimum(centers[:, 0], W - centers[:, 0]),
            np.minimum(centers[:, 1], H - centers[:, 1])
        )
        
        # Pairwise distances
        # To avoid N^2 full matrix calculation every step if possible, 
        # but N=21 is small, so full matrix is fine.
        dists = np.zeros((N, N))
        for i in range(N):
            dists[i, :] = np.linalg.norm(centers - centers[i], axis=1)
            
        # Set diagonal to inf so min ignores self
        np.fill_diagonal(dists, np.inf)
        
        # The radius is limited by half the distance to the nearest neighbor
        neighbor_limits = np.min(dists, axis=1) / 2.0
        
        radii = np.minimum(wall_dists, neighbor_limits)
        
        return np.sum(radii)

    def generate_lattice_seed(W, H, pattern='grid'):
        """
        Generate a structured initial population member.
        Returns x array of shape (43,)
        """
        x = np.zeros(43)
        x[0] = W
        
        if pattern == 'grid':
            # 5x5 grid, remove 4 corners to get 21? No, 5x5=25. 
            # 21 circles. Maybe 5 rows of 4? 5*4=20. Add 1 in middle?
            # Or 7x3? 7*3=21.
            # Let's try 7x3 grid
            rows, cols = 3, 7
            for i in range(N):
                r = i // cols
                c = i % cols
                x[1 + 2*i] = (c + 0.5) * W / cols
                x[2 + 2*i] = (r + 0.5) * H / rows
                
        elif pattern == 'hex':
            # Hexagonal packing approximation
            # 21 circles.
            # Rows: 4, 5, 4, 5, 3? Total 21.
            # Or 4, 5, 4, 4, 4?
            # Let's just use a jittered grid for now, DE will fix it.
            # Actually, let's try a 5x5 grid with 4 removed.
            # Simpler: 7x3 is compact.
            rows, cols = 3, 7
            for i in range(N):
                r = i // cols
                c = i % cols
                # Hex offset
                offset = 0.5 if r % 2 == 1 else 0.0
                x[1 + 2*i] = (c + offset) * W / cols
                x[2 + 2*i] = (r + 0.5) * H / rows
                
        return x

    # Differential Evolution Parameters
    pop_size = 30
    F = 0.8  # Mutation scale
    CR = 0.9  # Crossover rate
    
    # Initialize Population
    pop = np.zeros((pop_size, 43))
    
    # Add some structured seeds
    # Square
    pop[0] = generate_lattice_seed(1.0, 1.0, 'grid')
    # Rect 1.5 x 0.5
    pop[1] = generate_lattice_seed(1.5, 0.5, 'grid')
    # Rect 0.5 x 1.5
    pop[2] = generate_lattice_seed(0.5, 1.5, 'grid')
    # Hex on square
    pop[3] = generate_lattice_seed(1.0, 1.0, 'hex')
    
    # Random seeds
    for i in range(4, pop_size):
        W = np.random.uniform(0.5, 1.5)
        H = 2.0 - W
        pop[i, 0] = W
        # Random positions
        pop[i, 1::2] = np.random.uniform(0.05, W - 0.05, N)
        pop[i, 2::2] = np.random.uniform(0.05, H - 0.05, N)
        
    # Evaluate initial population
    scores = np.array([evaluate(x) for x in pop])
    best_idx = np.argmax(scores)
    best_x = pop[best_idx].copy()
    best_score = scores[best_idx]
    
    # DE Loop
    gen = 0
    while time.time() < deadline:
        gen += 1
        for i in range(pop_size):
            if time.time() > deadline - 5:
                break
                
            # Select 3 distinct individuals other than i
            mask = np.ones(pop_size, dtype=bool)
            mask[i] = False
            candidates = np.where(mask)[0]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            # Mutation
            v = pop[a] + F * (pop[b] - pop[c])
            
            # Crossover
            u = np.empty_like(v)
            j_rand = np.random.randint(0, 43)
            for j in range(43):
                if np.random.random() < CR or j == j_rand:
                    u[j] = v[j]
                else:
                    u[j] = pop[i][j]
                    
            # Ensure W is within bounds for evaluation
            # If W is out of bounds, evaluate will return -1e9, which is fine.
            # But we can clip W to [0.1, 1.9] to keep population diverse?
            # Let's just let it be, the fitness function handles it.
            
            u_score = evaluate(u)
            
            if u_score > scores[i]:
                pop[i] = u
                scores[i] = u_score
                
                if u_score > best_score:
                    best_score = u_score
                    best_x = u.copy()
                    
        # Elitism: keep the best individual in the population
        # Find the current best in pop
        current_best_idx = np.argmax(scores)
        if scores[current_best_idx] < best_score:
            # This shouldn't happen if we update best_score correctly, 
            # but just in case, replace the worst with the global best
            worst_idx = np.argmin(scores)
            pop[worst_idx] = best_x
            scores[worst_idx] = best_score

    # Finalize best solution
    W = best_x[0]
    H = 2.0 - W
    centers = best_x[1:].reshape(-1, 2)
    
    # Recalculate radii for the best solution to ensure consistency
    wall_dists = np.minimum(
        np.minimum(centers[:, 0], W - centers[:, 0]),
        np.minimum(centers[:, 1], H - centers[:, 1])
    )
    
    dists = np.zeros((N, N))
    for i in range(N):
        dists[i, :] = np.linalg.norm(centers - centers[i], axis=1)
    np.fill_diagonal(dists, np.inf)
    neighbor_limits = np.min(dists, axis=1) / 2.0
    
    radii = np.minimum(wall_dists, neighbor_limits)
    
    # Construct output array (N, 3): x, y, r
    result = np.column_stack([centers, radii])
    
    return result

# EVOLVE-BLOCK-END
