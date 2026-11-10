"""Optimized 2-colouring of [1,256] to minimize monochromatic 4-term APs."""
import random
import time
import numpy as np

N = 256
TIME_LIMIT = 115.0  # safety margin before 120s kill
SEED = 42


def _build_aps():
    """Build all 4-term arithmetic progressions as index tuples (0-indexed)."""
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps


def _build_aps_array(aps):
    """Convert AP list to numpy array for vectorized counting."""
    return np.array(aps, dtype=np.int32)


def _count_mono_vec(aps_arr, colors):
    """Count monochromatic 4-APs using vectorized numpy ops."""
    # colors is a numpy array of 0/1
    # For each AP, check if all 4 positions have same colour
    c0 = colors[aps_arr[:, 0]]
    c1 = colors[aps_arr[:, 1]]
    c2 = colors[aps_arr[:, 2]]
    c3 = colors[aps_arr[:, 3]]
    return int(np.sum((c0 == c1) & (c1 == c2) & (c2 == c3)))


def _count_mono_fast(aps, colors):
    """Count monochromatic 4-APs with Python loop (for small use)."""
    count = 0
    for a, b, c, d in aps:
        if colors[a] == colors[b] == colors[c] == colors[d]:
            count += 1
    return count


def _random_initial():
    """Random binary coloring."""
    return np.random.randint(0, 2, N).astype(np.int32)


def _greedy_local_search(aps_arr, colors, max_passes=30, deadline=None):
    """Greedy local search: repeatedly flip bits that reduce mono-AP count."""
    colors = colors.copy()
    n = len(colors)
    best = _count_mono_vec(aps_arr, colors)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Shuffle order for diversity
        order = np.random.permutation(n)
        for i in order:
            if deadline is not None and time.time() > deadline:
                break
            # Try flipping bit i
            colors[i] = 1 - colors[i]
            new_count = _count_mono_vec(aps_arr, colors)
            if new_count < best:
                best = new_count
                improved = True
            else:
                colors[i] = 1 - colors[i]
        if not improved:
            break
    return colors, best


def _simulated_annealing(aps_arr, colors, start_temp=1.0, end_temp=0.01,
                         max_iter=5000, deadline=None):
    """Simulated annealing to escape local minima."""
    colors = colors.copy()
    n = len(colors)
    current = _count_mono_vec(aps_arr, colors)
    best_colors = colors.copy()
    best = current
    T = start_temp
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        # Pick random position to flip
        i = np.random.randint(n)
        colors[i] = 1 - colors[i]
        new_count = _count_mono_vec(aps_arr, colors)
        delta = new_count - current
        
        if delta < 0 or (T > 0 and np.random.random() < np.exp(-delta / max(T, 1e-9))):
            current = new_count
            if current < best:
                best = current
                best_colors = colors.copy()
        else:
            colors[i] = 1 - colors[i]
        
        # Cool down
        T = start_temp * (end_temp / start_temp) ** (it / max_iter)
    
    return best_colors, best


def _uniform_crossover(parent1, parent2):
    """Uniform crossover: each position taken randomly from one parent."""
    mask = np.random.randint(0, 2, N)
    child = np.where(mask == 1, parent1, parent2)
    return child


def _mutate(colors, rate=0.02):
    """Random mutation: flip each bit with probability rate."""
    colors = colors.copy()
    mask = np.random.random(N) < rate
    colors[mask] = 1 - colors[mask]
    return colors


def _tournament_select(population, scores, k=3):
    """Tournament selection: pick k random, return best."""
    idxs = np.random.randint(0, len(population), k)
    best_idx = idxs[np.argmin([scores[i] for i in idxs])]
    return population[best_idx].copy()


def construct():
    random.seed(SEED)
    np.random.seed(SEED)
    
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Precompute APs
    aps_list = _build_aps()
    aps_arr = _build_aps_array(aps_list)
    
    # Initialize population with diverse starting points
    pop_size = 20
    population = []
    scores = []
    
    # Start with random individuals
    for _ in range(pop_size):
        c = _random_initial()
        population.append(c)
    
    # Also add some structured starts
    # Block pattern
    c_block = np.array([(i // 4) % 2 for i in range(N)], dtype=np.int32)
    population.append(c_block)
    # Alternating pairs
    c_pairs = np.array([(i // 2) % 2 for i in range(N)], dtype=np.int32)
    population.append(c_pairs)
    # Thue-Morse-like pattern
    c_tm = np.zeros(N, dtype=np.int32)
    for i in range(N):
        c_tm[i] = bin(i).count('1') % 2
    population.append(c_tm)
    
    # Evaluate all
    for c in population:
        scores.append(_count_mono_vec(aps_arr, c))
    
    best_score = min(scores)
    best_colors = population[scores.index(best_score)].copy()
    
    # Run genetic algorithm with local search
    gen = 0
    while time.time() < deadline - 15:  # Leave 15s buffer
        gen += 1
        
        # Select parents
        new_pop = []
        for _ in range(pop_size):
            p1 = _tournament_select(population, scores, k=3)
            p2 = _tournament_select(population, scores, k=3)
            child = _uniform_crossover(p1, p2)
            child = _mutate(child, rate=0.03)
            new_pop.append(child)
        
        # Local search on each child (shorter passes for speed)
        for i in range(len(new_pop)):
            if time.time() > deadline - 15:
                break
            c, s = _greedy_local_search(aps_arr, new_pop[i], max_passes=10, deadline=deadline - 10)
            new_pop[i] = c
            scores[i] = s
        
        # Update population: keep best from old and new
        all_candidates = [(c, s) for c, s in zip(population, scores)] + \
                         [(c, s) for c, s in zip(new_pop, scores[:len(new_pop)])]
        all_candidates.sort(key=lambda x: x[1])
        
        population = [c for c, s in all_candidates[:pop_size]]
        scores = [s for c, s in all_candidates[:pop_size]]
        
        # Track best
        if scores[0] < best_score:
            best_score = scores[0]
            best_colors = population[0].copy()
        
        # Periodically apply SA to the best individual
        if gen % 5 == 0 and time.time() < deadline - 20:
            sa_colors, sa_score = _simulated_annealing(
                aps_arr, best_colors, start_temp=2.0, end_temp=0.01,
                max_iter=2000, deadline=deadline - 10
            )
            if sa_score < best_score:
                best_score = sa_score
                best_colors = sa_colors
                # Insert into population
                population[0] = best_colors
                scores[0] = best_score
                population.sort(key=lambda c: _count_mono_vec(aps_arr, c))
                scores.sort()
    
    # Final local search on best
    if time.time() < deadline - 2:
        final_colors, final_score = _greedy_local_search(
            aps_arr, best_colors, max_passes=50, deadline=deadline - 1
        )
        if final_score < best_score:
            best_score = final_score
            best_colors = final_colors
    
    # Return as list of ints
    return [int(x) for x in best_colors]
