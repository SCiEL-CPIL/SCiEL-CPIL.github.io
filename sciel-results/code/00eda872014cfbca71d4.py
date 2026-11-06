import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # Precompute all 4-term arithmetic progressions
    # AP: (a, a+d, a+2d, a+3d) with d >= 1, a+3d <= n
    # Using 0-based indices: (i, i+d, i+2d, i+3d) where i+3d < n
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    aps = np.array(aps, dtype=np.int32)
    num_aps = len(aps)
    
    def count_mono(color):
        """Count monochromatic 4-term APs. color is 0/1 array of length n."""
        # For each AP, check if all 4 positions have the same color
        # c0, c1, c2, c3 are the colors at the 4 positions
        c0 = color[aps[:, 0]]
        c1 = color[aps[:, 1]]
        c2 = color[aps[:, 2]]
        c3 = color[aps[:, 3]]
        # Monochromatic if all equal: c0==c1 and c1==c2 and c2==c3
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(np.sum(mono))
    
    # Initialize population with diverse starting points
    pop_size = 20
    population = []
    scores = []
    
    # Thue-Morse sequence
    tm = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    population.append(tm.copy())
    scores.append(count_mono(tm))
    
    # Random initializations
    for i in range(pop_size - 1):
        if time.time() > deadline:
            break
        # Mix of random and perturbed Thue-Morse
        if i % 2 == 0:
            c = np.random.randint(0, 2, size=n).astype(np.int8)
        else:
            c = tm.copy()
            # Flip random fraction
            flip_mask = np.random.random(n) < 0.15
            c[flip_mask] = 1 - c[flip_mask]
        population.append(c)
        scores.append(count_mono(c))
    
    # Genetic algorithm with local search
    best_color = population[0].copy()
    best_score = min(scores)
    
    # Find best initial
    for i in range(len(population)):
        if scores[i] < best_score:
            best_score = scores[i]
            best_color = population[i].copy()
    
    # Local search function
    def local_search(color, max_iters=200):
        """Hill climbing: flip individual bits to reduce mono count."""
        color = color.copy()
        current_score = count_mono(color)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try flipping each position, compute delta efficiently
            # For position p, flipping changes its contribution to all APs containing p
            # We'll use a simpler approach: try random flips and keep if better
            
            improved = False
            # Try a few random positions
            for _ in range(10):
                if time.time() > deadline:
                    break
                p = random.randint(0, n - 1)
                color[p] = 1 - color[p]
                new_score = count_mono(color)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    color[p] = 1 - color[p]  # revert
            
            if not improved:
                # Try more aggressive: flip up to 3 random positions
                improved = False
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    flips = random.sample(range(n), 3)
                    orig = [color[p] for p in flips]
                    for p in flips:
                        color[p] = 1 - color[p]
                    new_score = count_mono(color)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        for j, p in enumerate(flips):
                            color[p] = orig[j]
                if not improved:
                    break
        
        return color, current_score
    
    # Apply local search to best individuals
    for i in range(min(5, len(population))):
        if time.time() > deadline:
            break
        c, s = local_search(population[i], max_iters=100)
        if s < best_score:
            best_score = s
            best_color = c
    
    # Main GA loop
    gen = 0
    while time.time() < deadline:
        gen += 1
        
        # Create next generation
        new_pop = []
        new_scores = []
        
        # Elitism: keep best 2
        sorted_idx = np.argsort(scores)
        for i in range(min(2, len(population))):
            idx = sorted_idx[i]
            new_pop.append(population[idx].copy())
            new_scores.append(scores[idx])
        
        # Fill rest with offspring
        while len(new_pop) < pop_size:
            if time.time() > deadline:
                break
            
            # Selection: tournament
            def tournament():
                idxs = random.sample(range(len(population)), min(3, len(population)))
                best_idx = min(idxs, key=lambda x: scores[x])
                return population[best_idx]
            
            parent1 = tournament()
            parent2 = tournament()
            
            # Crossover: uniform
            child = parent1.copy()
            mask = np.random.random(n) < 0.5
            child[mask] = parent2[mask]
            
            # Mutation
            if random.random() < 0.3:
                num_flips = random.randint(1, 5)
                flip_positions = random.sample(range(n), num_flips)
                for p in flip_positions:
                    child[p] = 1 - child[p]
            
            new_pop.append(child)
            new_scores.append(count_mono(child))
        
        population = new_pop
        scores = new_scores
        
        # Apply local search to top few
        sorted_idx = np.argsort(scores)
        for i in range(min(3, len(population))):
            if time.time() > deadline:
                break
            idx = sorted_idx[i]
            c, s = local_search(population[idx], max_iters=50)
            if s < scores[idx]:
                population[idx] = c
                scores[idx] = s
        
        # Update best
        for i in range(len(population)):
            if scores[i] < best_score:
                best_score = scores[i]
                best_color = population[i].copy()
    
    return best_color.tolist()
