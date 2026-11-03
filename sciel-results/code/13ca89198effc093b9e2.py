import numpy as np
import time
import random

N = 600
TIME_LIMIT = 115  # Safety margin

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Use FFT for fast convolution
    conv = np.fft.irfft(np.fft.rfft(f) ** 2, n=2*N-1)
    return np.max(conv) / (s * s)

def normalize(f):
    """Normalize f to have a fixed sum for stable comparison, but keep nonnegative."""
    s = np.sum(f)
    if s < 1e-15:
        return np.ones(N) * (1.0 / N)
    return f / s

def generate_initial_population(pop_size):
    """Generate a diverse initial population."""
    population = []
    # Some random uniform
    for _ in range(pop_size // 4):
        f = np.random.rand(N)
        population.append(f)
    
    # Some blocky structures (fewer distinct values)
    for _ in range(pop_size // 4):
        blocks = np.random.randint(2, 10)
        f = np.zeros(N)
        for i in range(blocks):
            start = int(i * N / blocks)
            end = int((i + 1) * N / blocks)
            val = np.random.rand()
            f[start:end] = val
        population.append(f)
        
    # Some smooth sine waves
    for _ in range(pop_size // 4):
        freq = np.random.randint(1, 5)
        phase = np.random.rand() * 2 * np.pi
        x = np.linspace(0, 2 * np.pi * freq, N)
        f = 0.5 + 0.5 * np.sin(x + phase)
        population.append(f)
        
    # Fill rest with random
    while len(population) < pop_size:
        f = np.random.rand(N)
        population.append(f)
        
    return population

def tournament_selection(population, scores, k=3):
    """Select k individuals randomly and return the best one."""
    indices = np.random.choice(len(population), size=k, replace=False)
    best_idx = indices[0]
    best_score = scores[best_idx]
    for idx in indices[1:]:
        if scores[idx] < best_score:
            best_score = scores[idx]
            best_idx = idx
    return population[best_idx].copy()

def blend_crossover(parent1, parent2, alpha):
    """Blend crossover: child = alpha * p1 + (1-alpha) * p2"""
    return alpha * parent1 + (1 - alpha) * parent2

def mutate(f, rate, strength):
    """Apply Gaussian mutation to a fraction of elements."""
    f_mut = f.copy()
    mask = np.random.rand(N) < rate
    if np.any(mask):
        noise = np.random.randn(np.sum(mask)) * strength
        f_mut[mask] += noise
    # Ensure nonnegativity
    f_mut = np.maximum(f_mut, 0.0)
    return f_mut

def local_hill_climb(f, max_iterations=5000, step_size=0.05):
    """Fast local refinement by perturbing random bins and accepting improvements."""
    current_f = f.copy()
    current_score = compute_score(current_f)
    
    for _ in range(max_iterations):
        # Pick a few random indices to perturb
        n_perturb = np.random.randint(1, 5)
        indices = np.random.choice(N, size=n_perturb, replace=False)
        
        new_f = current_f.copy()
        for idx in indices:
            new_f[idx] += np.random.randn() * step_size
            new_f[idx] = max(0.0, new_f[idx])
            
        new_score = compute_score(new_f)
        
        if new_score < current_score:
            current_f = new_f
            current_score = new_score
            # Slightly reduce step size as we converge
            step_size *= 0.999
            
    return current_f, current_score

def construct():
    start_time = time.time()
    pop_size = 50
    
    # Initialize population
    population = generate_initial_population(pop_size)
    scores = [compute_score(f) for f in population]
    
    best_f = population[np.argmin(scores)].copy()
    best_score = min(scores)
    
    # Simulated Annealing parameters
    T = 0.1  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.995
    generations = 0
    
    # Main GA loop
    while time.time() - start_time < TIME_LIMIT - 20:
        generations += 1
        new_population = []
        new_scores = []
        
        # Elitism: keep the best individual
        best_idx = np.argmin(scores)
        new_population.append(population[best_idx].copy())
        new_scores.append(scores[best_idx])
        
        # Generate new individuals
        while len(new_population) < pop_size:
            if time.time() - start_time > TIME_LIMIT - 25:
                break
                
            # Select parents
            p1 = tournament_selection(population, scores)
            p2 = tournament_selection(population, scores)
            
            # Crossover
            alpha = np.random.rand()
            child = blend_crossover(p1, p2, alpha)
            
            # Mutation
            mutation_rate = 0.05 + 0.05 * (T / 0.1)  # Higher mutation when T is high
            mutation_strength = 0.1 + 0.1 * (T / 0.1)
            child = mutate(child, mutation_rate, mutation_strength)
            
            # Evaluate
            child_score = compute_score(child)
            
            # Simulated Annealing acceptance
            # Compare with the worst in current population (or random)
            worst_idx = np.argmax(scores)
            if child_score < scores[worst_idx]:
                accept = True
            else:
                delta = child_score - scores[worst_idx]
                if delta < 0:
                    accept = True
                else:
                    accept = np.random.rand() < np.exp(-delta / (T + 1e-12))
            
            if accept:
                new_population.append(child)
                new_scores.append(child_score)
            else:
                # If rejected, just add a mutated version of a random parent to maintain diversity
                random_parent = population[np.random.randint(len(population))]
                rejected_child = mutate(random_parent, 0.1, 0.2)
                rejected_score = compute_score(rejected_child)
                new_population.append(rejected_child)
                new_scores.append(rejected_score)
        
        # Update population
        population = new_population
        scores = new_scores
        
        # Track global best
        current_best_idx = np.argmin(scores)
        if scores[current_best_idx] < best_score:
            best_score = scores[current_best_idx]
            best_f = population[current_best_idx].copy()
            
        # Cool down
        T *= cooling_rate
        
    # Final local refinement on the best solution found
    if time.time() - start_time < TIME_LIMIT - 5:
        refined_f, refined_score = local_hill_climb(best_f, max_iterations=10000, step_size=0.05)
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Ensure nonnegative and return
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
