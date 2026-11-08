"""Direct per-point optimization with coordinate descent for Erdos minimum overlap."""
import time
import numpy as np

N = 100
dx = 2.0 / N

def _compute_max_overlap(h):
    """Compute max over all circular shifts k of sum_j h[j]*(1-h[(j+k)%N]) * dx."""
    n = len(h)
    H = np.fft.rfft(h, n=2*n)
    autocorr = np.fft.irfft(H * np.conj(H), n=2*n)[:n]
    # autocorr[k] = sum_j h[j] * h[(j-k)%n]
    # We want C[k] = sum_j h[j] * h[(j+k)%n] = autocorr[(-k)%n]
    C = np.zeros(n)
    C[0] = autocorr[0]
    if n > 1:
        C[1:] = autocorr[n-1:0:-1]
    overlaps = (n/2.0 - C) * dx
    return np.max(overlaps)

def _project_to_valid(h):
    """Project h to satisfy 0<=h<=1 and sum(h)=N/2."""
    h = np.clip(h, 0.0, 1.0)
    current_sum = np.sum(h)
    target_sum = N / 2.0
    diff = target_sum - current_sum
    if abs(diff) > 1e-12:
        if diff > 0:
            mask = h < 1.0 - 1e-12
            if np.sum(mask) > 0:
                h[mask] += diff / np.sum(mask)
        else:
            mask = h > 1e-12
            if np.sum(mask) > 0:
                h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    return h

def _coordinate_descent(h, max_iters=5000, tol=1e-10, time_limit=120.0):
    """Coordinate descent on individual h[j] values."""
    start_time = time.time()
    n = len(h)
    best_score = _compute_max_overlap(h)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for j in range(n):
            old_val = h[j]
            # Try small perturbations
            for delta in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001]:
                new_val = old_val + delta
                if new_val < 0 or new_val > 1:
                    continue
                h[j] = new_val
                # Project to maintain sum constraint
                h = _project_to_valid(h)
                score = _compute_max_overlap(h)
                if score < best_score - tol:
                    best_score = score
                    improved = True
                    break
                else:
                    h[j] = old_val
            if not improved:
                h[j] = old_val
        
        if not improved:
            break
    
    return h, best_score

def _try_binary_candidates():
    """Try various binary (0/1) patterns with exactly N/2 ones."""
    candidates = []
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)
    
    # Random binary sequences
    np.random.seed(42)
    for _ in range(200):
        idx = np.random.choice(N, size=N//2, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0
        score = _compute_max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Structured patterns
    # Alternating blocks
    for block_size in range(1, N//2 + 1):
        h = np.zeros(N)
        for i in range(0, N, 2*block_size):
            h[i:i+block_size] = 1.0
        if np.sum(h) == N//2:
            score = _compute_max_overlap(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Fibonacci-like or other structured patterns
    for start in range(N):
        h = np.zeros(N)
        for i in range(N//2):
            h[(start + i*2) % N] = 1.0
        score = _compute_max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    return best_h, best_score

def _genetic_algorithm(h_init, pop_size=50, n_generations=200, time_limit=120.0):
    """Simple genetic algorithm on per-point values."""
    start_time = time.time()
    n = len(h_init)
    
    # Initialize population
    pop = [h_init.copy() for _ in range(pop_size)]
    # Add some random perturbations
    for i in range(pop_size):
        if i > 0:
            mask = np.random.random(n) < 0.1
            pop[i][mask] += np.random.uniform(-0.2, 0.2, np.sum(mask))
            pop[i] = _project_to_valid(pop[i])
    
    best_h = h_init.copy()
    best_score = _compute_max_overlap(h_init)
    
    for gen in range(n_generations):
        if time.time() - start_time > time_limit:
            break
        
        # Evaluate fitness
        scores = [_compute_max_overlap(h) for h in pop]
        
        # Find best
        idx_best = np.argmin(scores)
        if scores[idx_best] < best_score:
            best_score = scores[idx_best]
            best_h = pop[idx_best].copy()
        
        # Selection: keep top half
        sorted_idx = np.argsort(scores)
        survivors = [pop[i].copy() for i in sorted_idx[:pop_size//2]]
        
        # Crossover and mutation to fill rest
        while len(survivors) < pop_size:
            # Crossover
            i1, i2 = np.random.randint(0, len(survivors), 2)
            crossover_point = np.random.randint(1, n)
            child1 = np.concatenate([survivors[i1][:crossover_point], survivors[i2][crossover_point:]])
            child2 = np.concatenate([survivors[i2][:crossover_point], survivors[i1][crossover_point:]])
            
            # Mutation
            for child in [child1, child2]:
                mask = np.random.random(n) < 0.05
                child[mask] += np.random.uniform(-0.1, 0.1, np.sum(mask))
                child = _project_to_valid(child)
            
            survivors.append(child1)
            if len(survivors) < pop_size:
                survivors.append(child2)
        
        pop = survivors
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Start with constant baseline
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)
    
    # Try binary candidates
    if time.time() < deadline:
        h_bin, score_bin = _try_binary_candidates()
        if score_bin < best_score:
            best_score = score_bin
            best_h = h_bin.copy()
    
    # Coordinate descent from best
    if time.time() < deadline - 10:
        h_cd, score_cd = _coordinate_descent(best_h.copy(), max_iters=3000, time_limit=60.0)
        if score_cd < best_score:
            best_score = score_cd
            best_h = h_cd.copy()
    
    # Genetic algorithm
    if time.time() < deadline - 15:
        h_ga, score_ga = _genetic_algorithm(best_h.copy(), pop_size=30, n_generations=100, time_limit=50.0)
        if score_ga < best_score:
            best_score = score_ga
            best_h = h_ga.copy()
    
    # Final coordinate descent polish
    if time.time() < deadline - 5:
        h_final, score_final = _coordinate_descent(best_h.copy(), max_iters=2000, time_limit=30.0)
        if score_final < best_score:
            best_score = score_final
            best_h = h_final.copy()
    
    # Final validation
    best_h = np.clip(best_h, 0.0, 1.0)
    if abs(np.sum(best_h) - N/2) > 1e-6:
        best_h = np.full(N, 0.5)
    
    return best_h
