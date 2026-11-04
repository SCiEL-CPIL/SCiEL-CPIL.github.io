"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def _autocorr_peak(h):
    """Compute max absolute autocorrelation at non-zero lags."""
    n = len(h)
    if n <= 1:
        return 0.0
    # Use FFT for efficient autocorrelation
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # acf[k] for k in 1..n-1 gives positive lags
    # For real signals, acf is symmetric, so max over 1..n-1 suffices
    return float(np.max(np.abs(acf[1:n])))

def _score(h):
    """Upper bound on C3: max absolute autocorrelation at non-zero lags."""
    return _autocorr_peak(h)

def _fast_greedy(h, time_limit=2.0):
    """
    Fast greedy local search.
    1. Flip signs of individual bits.
    2. Zero out bits.
    3. Activate zero bits with +/- 1.
    Uses random ordering to escape local minima.
    Optimized for speed: only one pass per 'while' iteration, checking time frequently.
    """
    n = len(h)
    best_h = h.copy()
    best_score = _score(best_h)
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        # We use a single pass over random indices for speed, 
        # but we can do multiple passes if time permits.
        
        # 1. Try flipping signs of non-zero elements
        nonzero_idx = np.nonzero(best_h)[0]
        if len(nonzero_idx) > 0:
            # Sample a subset if the list is large to keep it fast per step
            # For N=400, full permutation is fine, but let's be safe
            k = min(len(nonzero_idx), 40) 
            order = np.random.choice(len(nonzero_idx), size=k, replace=False)
            for idx_pos in order:
                if time.time() - start > time_limit:
                    break
                i = nonzero_idx[idx_pos]
                trial = best_h.copy()
                trial[i] = -trial[i]
                s = _score(trial)
                if s < best_score - 1e-12:
                    best_h = trial
                    best_score = s
                    improved = True
                    break
        
        # 2. Try zeroing out
        if not improved:
            nonzero_idx = np.nonzero(best_h)[0]
            if len(nonzero_idx) > 0:
                k = min(len(nonzero_idx), 40)
                order = np.random.choice(len(nonzero_idx), size=k, replace=False)
                for idx_pos in order:
                    if time.time() - start > time_limit:
                        break
                    i = nonzero_idx[idx_pos]
                    trial = best_h.copy()
                    trial[i] = 0.0
                    s = _score(trial)
                    if s < best_score - 1e-12:
                        best_h = trial
                        best_score = s
                        improved = True
                        break
        
        # 3. Try activating zeros
        if not improved:
            zero_idx = np.nonzero(best_h == 0)[0]
            if len(zero_idx) > 0:
                k = min(20, len(zero_idx))
                order = np.random.choice(len(zero_idx), size=k, replace=False)
                for idx_pos in order:
                    if time.time() - start > time_limit:
                        break
                    i = zero_idx[idx_pos]
                    for sign in [1.0, -1.0]:
                        trial = best_h.copy()
                        trial[i] = sign
                        s = _score(trial)
                        if s < best_score - 1e-12:
                            best_h = trial
                            best_score = s
                            improved = True
                            break
                    if improved:
                        break
        
        if not improved:
            # If no improvement in a full random pass, we might be stuck.
            # We break to let the outer loop (GA or SA) handle global moves.
            # However, for the memetic step, we want to keep trying different random orders.
            # Let's just continue the while loop to try new random orders.
            # If we really want to stop, we could count consecutive non-improvements.
            pass
            
    return best_h, best_score

def _simulated_annealing(h, time_limit=5.0):
    """
    Simulated annealing to escape local minima.
    """
    n = len(h)
    current_h = h.copy()
    current_score = _score(current_h)
    best_h = current_h.copy()
    best_score = current_score
    
    start = time.time()
    T = 5.0
    cooling = 0.995
    min_T = 0.1
    
    while T > min_T and time.time() - start < time_limit:
        # Pick a random move
        i = np.random.randint(n)
        trial = current_h.copy()
        
        # Randomly choose between flip, zero, or activate
        r = np.random.random()
        if r < 0.6:
            # Flip sign (or activate if 0)
            if trial[i] == 0:
                trial[i] = 1.0 if np.random.random() < 0.5 else -1.0
            else:
                trial[i] = -trial[i]
        elif r < 0.8:
            # Zero out
            trial[i] = 0.0
        else:
            # Activate (if zero) or flip (if non-zero)
            if trial[i] == 0:
                trial[i] = 1.0 if np.random.random() < 0.5 else -1.0
            else:
                trial[i] = -trial[i]
        
        new_score = _score(trial)
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_h = trial
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = current_h.copy()
        
        T *= cooling
        
    return best_h, best_score

def _quasiperiodic_init(frequencies, phases, amplitudes, threshold=0.0):
    """
    Generate a signed step function based on a sum of complex exponentials.
    """
    n = N
    t = np.arange(n)
    signal = np.zeros(n)
    
    for f, p, a in zip(frequencies, phases, amplitudes):
        signal += a * np.cos(2 * np.pi * f * t + p)
    
    h = np.zeros(n)
    h[signal > threshold] = 1.0
    h[signal < -threshold] = -1.0
    
    return h

def _genetic_algorithm(time_budget):
    """
    Memetic Algorithm: GA with local search refinement.
    """
    pop_size = 60
    generations = 40
    tournament_k = 3
    crossover_rate = 0.8
    mutation_rate = 0.1
    
    # Initialize population
    population = []
    scores = []
    
    # 1. Quasi-periodic individuals
    for _ in range(20):
        freqs = [1/np.sqrt(2), 1/np.sqrt(3), 1/np.sqrt(5)]
        phases = np.random.uniform(0, 2*np.pi, size=3)
        amplitudes = np.random.uniform(0.5, 1.5, size=3)
        thresh = np.random.uniform(0.0, 0.5)
        h = _quasiperiodic_init(freqs, phases, amplitudes, thresh)
        population.append(h)
        
    # 2. Sparse random individuals
    for _ in range(20):
        k = np.random.randint(20, 100)
        indices = np.random.choice(N, size=k, replace=False)
        h = np.zeros(N)
        h[indices] = np.random.choice([-1.0, 1.0], size=k)
        population.append(h)
        
    # 3. Dense random individuals
    for _ in range(20):
        h = np.random.choice([-1.0, 0.0, 1.0], size=N)
        population.append(h)

    # Evaluate initial population
    for h in population:
        scores.append(_score(h))
    
    best_h = population[np.argmin(scores)]
    best_score = min(scores)
    
    start_time = time.time()
    
    for gen in range(generations):
        if time.time() - start_time > time_budget:
            break
            
        # Selection and Crossover
        new_population = []
        
        # Elitism: keep best 2
        sorted_indices = np.argsort(scores)
        for i in range(2):
            new_population.append(population[sorted_indices[i]].copy())
        
        while len(new_population) < pop_size:
            # Tournament selection
            p1 = _tournament_select(population, scores, tournament_k)
            p2 = _tournament_select(population, scores, tournament_k)
            
            if np.random.random() < crossover_rate:
                # Uniform crossover
                mask = np.random.random(N) < 0.5
                child = np.where(mask, p1, p2)
            else:
                child = p1.copy()
            
            # Mutation
            if np.random.random() < mutation_rate:
                # Random single point mutation or small block
                num_mut = np.random.randint(1, 5)
                for _ in range(num_mut):
                    idx = np.random.randint(N)
                    r = np.random.random()
                    if r < 0.5:
                        child[idx] = -child[idx] if child[idx] != 0 else (1.0 if np.random.random()<0.5 else -1.0)
                    elif r < 0.8:
                        child[idx] = 0.0
                    else:
                        if child[idx] == 0:
                            child[idx] = 1.0 if np.random.random()<0.5 else -1.0
                        else:
                            child[idx] = -child[idx]
            
            new_population.append(child)
        
        population = new_population
        
        # Evaluate new population
        new_scores = [_score(h) for h in population]
        
        # Memetic Step: Refine the best few individuals with local search
        # We only refine the top 5 to save time
        top_k = 5
        sorted_indices = np.argsort(new_scores)
        
        for i in range(top_k):
            idx = sorted_indices[i]
            # Check time
            if time.time() - start_time > time_budget - 1.0:
                break
            
            # Apply fast greedy local search
            refined_h, refined_score = _fast_greedy(population[idx], time_limit=1.0)
            
            # Update population and scores
            population[idx] = refined_h
            new_scores[idx] = refined_score
            
            # Track global best
            if refined_score < best_score:
                best_score = refined_score
                best_h = refined_h.copy()
        
        scores = new_scores
        
        # If no improvement in several generations, inject random individuals (restarting)
        if gen > 0 and gen % 10 == 0 and scores[np.argmin(scores)] >= best_score * 1.01:
             # Inject 10 random individuals
             for _ in range(10):
                 idx = np.random.randint(pop_size)
                 k = np.random.randint(20, 100)
                 indices = np.random.choice(N, size=k, replace=False)
                 h = np.zeros(N)
                 h[indices] = np.random.choice([-1.0, 1.0], size=k)
                 population[idx] = h
                 scores[idx] = _score(h)
        
    return best_h, best_score

def _tournament_select(population, scores, k):
    """Tournament selection."""
    indices = np.random.randint(0, len(population), size=k)
    best_idx = indices[np.argmin([scores[i] for i in indices])]
    return population[best_idx]

def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound.
    
    Strategy:
    1. Use a Memetic Genetic Algorithm to explore the search space broadly.
    2. Use Fast Greedy Local Search to refine candidates.
    3. Use Simulated Annealing for final polishing on the best candidate.
    """
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Memetic GA
    # Allocate 80% of the time to GA
    ga_time = 110.0
    start = time.time()
    
    # Run GA
    h_ga, score_ga = _genetic_algorithm(ga_time)
    
    if score_ga < best_score:
        best_score = score_ga
        best_h = h_ga.copy()
    
    # Phase 2: Final Polish with Simulated Annealing
    # Use remaining time for SA on the best candidate
    elapsed = time.time() - start
    remaining_time = deadline - time.time()
    
    if best_h is not None and remaining_time > 5.0:
        # Use most of the remaining time for SA
        sa_time = max(2.0, remaining_time - 2.0)
        h_sa, score_sa = _simulated_annealing(best_h, time_limit=sa_time)
        
        if score_sa < best_score:
            best_score = score_sa
            best_h = h_sa.copy()
            
        # One last quick greedy pass if time permits
        if time.time() < deadline - 1.0:
            h_greedy, score_greedy = _fast_greedy(best_h, time_limit=1.0)
            if score_greedy < best_score:
                best_score = score_greedy
                best_h = h_greedy.copy()
    else:
        # Fallback if GA didn't finish or returned bad result
        # Try a quick sparse random init + greedy
        if time.time() < deadline - 5.0:
            for trial in range(5):
                k = np.random.randint(30, 80)
                indices = np.random.choice(N, size=k, replace=False)
                h = np.zeros(N)
                h[indices] = 1.0
                h_opt, score = _fast_greedy(h, time_limit=5.0)
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
                if time.time() > deadline - 10.0:
                    break
    
    # Fallback: if no solution found
    if best_h is None:
        best_h = np.ones(N)
        best_score = _score(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
