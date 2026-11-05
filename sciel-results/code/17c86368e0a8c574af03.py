"""Genetic algorithm for 3-AP maximization with no 4-AP constraint using bitmask fitness."""
import numpy as np
from itertools import combinations

N = 256

def precompute_data():
    """Precompute all 3-APs, 4-APs, and contribution tables."""
    # 3-AP: (a, a+d, a+2d) with a >= 1, a+2d <= N, d >= 1
    three_aps = []
    for a in range(1, N + 1):
        for d in range(1, N - a + 1):
            c = a + 2 * d
            if c <= N:
                three_aps.append((a - 1, a + d - 1, c - 1))
    three_aps = np.array(three_aps, dtype=np.int32)
    
    # 4-AP: (a, a+d, a+2d, a+3d) with a >= 1, a+3d <= N, d >= 1
    four_aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            c = a + 3 * d
            if c <= N:
                four_aps.append((a - 1, a + d - 1, a + 2 * d - 1, c - 1))
    four_aps = np.array(four_aps, dtype=np.int32)
    
    # Build contribution table: for each element i, which 3-APs does it participate in?
    # contrib[i] = list of 3-AP indices where i appears
    contrib = [[] for _ in range(N)]
    for ap_idx, (a, b, c) in enumerate(three_aps):
        contrib[a].append(ap_idx)
        contrib[b].append(ap_idx)
        contrib[c].append(ap_idx)
    
    # For each element i, which 4-APs does it participate in?
    four_contrib = [[] for _ in range(N)]
    for ap_idx, (a, b, c, d) in enumerate(four_aps):
        four_contrib[a].append(ap_idx)
        four_contrib[b].append(ap_idx)
        four_contrib[c].append(ap_idx)
        four_contrib[d].append(ap_idx)
    
    return three_aps, four_aps, contrib, four_contrib

THREE_APS, FOUR_APS, CONTRIB, FOUR_CONTRIB = precompute_data()
NUM_THREE_APS = len(THREE_APS)
NUM_FOUR_APS = len(FOUR_APS)

def count_three_aps_fast(mask):
    """Count 3-APs using vectorized check."""
    if NUM_THREE_APS == 0:
        return 0
    return int(np.sum(np.all(mask[THREE_APS], axis=1)))

def has_four_ap_fast(mask):
    """Check if mask contains any 4-AP."""
    if NUM_FOUR_APS == 0:
        return False
    return bool(np.any(np.all(mask[FOUR_APS], axis=1)))

def get_violating_4aps(mask):
    """Get indices of all violating 4-APs."""
    if NUM_FOUR_APS == 0:
        return np.array([], dtype=np.int32)
    return np.where(np.all(mask[FOUR_APS], axis=1))[0]

def repair_4aps(mask, max_removals=20):
    """Remove elements to eliminate all 4-APs."""
    mask = mask.copy()
    for _ in range(max_removals):
        violating = get_violating_4aps(mask)
        if len(violating) == 0:
            break
        # Pick a random violating 4-AP and remove a random element from it
        ap_idx = violating[np.random.randint(len(violating))]
        ap = FOUR_APS[ap_idx]
        # Prefer removing elements that are in fewer 3-APs
        scores = [len(CONTRIB[i]) for i in ap]
        min_score = min(scores)
        candidates = [i for i, s in zip(ap, scores) if s == min_score]
        remove_idx = candidates[np.random.randint(len(candidates))]
        mask[remove_idx] = False
    return mask

def greedy_construct():
    """Greedy construction: add elements in order, skip if creates 4-AP."""
    mask = np.zeros(N, dtype=bool)
    for idx in range(N):
        mask[idx] = True
        if has_four_ap_fast(mask):
            mask[idx] = False
    return mask

def random_construct():
    """Random 4-AP-free subset via randomized greedy."""
    mask = np.zeros(N, dtype=bool)
    order = np.random.permutation(N)
    for idx in order:
        mask[idx] = True
        if has_four_ap_fast(mask):
            mask[idx] = False
    return mask

def local_search(mask, max_iter=100):
    """Local search: try adding/removing elements to improve 3-AP count."""
    best_mask = mask.copy()
    best_fit = count_three_aps_fast(best_mask)
    
    for _ in range(max_iter):
        improved = False
        
        # Try adding random elements
        candidates = np.where(~best_mask)[0]
        if len(candidates) > 0:
            for _ in range(20):
                idx = candidates[np.random.randint(len(candidates))]
                test_mask = best_mask.copy()
                test_mask[idx] = True
                if not has_four_ap_fast(test_mask):
                    new_fit = count_three_aps_fast(test_mask)
                    if new_fit > best_fit:
                        best_mask = test_mask
                        best_fit = new_fit
                        improved = True
                        break
        
        # Try removing random elements (less likely to help but can enable additions)
        candidates = np.where(best_mask)[0]
        if len(candidates) > 0:
            for _ in range(5):
                idx = candidates[np.random.randint(len(candidates))]
                test_mask = best_mask.copy()
                test_mask[idx] = False
                new_fit = count_three_aps_fast(test_mask)
                if new_fit > best_fit:
                    best_mask = test_mask
                    best_fit = new_fit
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_mask

def crossover(mask1, mask2):
    """Uniform crossover with repair."""
    cross_points = np.random.random(N) < 0.5
    child = np.where(cross_points, mask1, mask2)
    child = repair_4aps(child, max_removals=30)
    return child

def mutate(mask, rate=0.08):
    """Mutation: flip random bits and repair."""
    child = mask.copy()
    flip_mask = np.random.random(N) < rate
    child[flip_mask] = ~child[flip_mask]
    child = repair_4aps(child, max_removals=30)
    return child

def genetic_algorithm(pop_size=40, generations=80, seed=42):
    """Run genetic algorithm."""
    np.random.seed(seed)
    
    # Initialize population with diverse starting points
    population = [greedy_construct()]
    for _ in range(pop_size - 1):
        population.append(random_construct())
    
    fitnesses = np.array([count_three_aps_fast(m) for m in population])
    best_mask = population[np.argmax(fitnesses)].copy()
    best_fit = int(np.max(fitnesses))
    
    for gen in range(generations):
        new_population = []
        
        for _ in range(pop_size):
            # Tournament selection
            t1 = np.random.randint(0, pop_size, 3)
            p1 = population[t1[np.argmax(fitnesses[t1])]]
            t2 = np.random.randint(0, pop_size, 3)
            p2 = population[t2[np.argmax(fitnesses[t2])]]
            
            # Crossover + mutation
            child = crossover(p1, p2)
            child = mutate(child, rate=0.06)
            
            # Local search
            child = local_search(child, max_iter=60)
            
            new_population.append(child)
        
        new_fitnesses = np.array([count_three_aps_fast(m) for m in new_population])
        
        # Elitism
        old_best_idx = np.argmax(fitnesses)
        new_best_idx = np.argmax(new_fitnesses)
        
        if new_fitnesses[new_best_idx] > fitnesses[old_best_idx]:
            fitnesses = new_fitnesses
            population = new_population
        else:
            worst_idx = np.argmin(new_fitnesses)
            new_population[worst_idx] = population[old_best_idx]
            new_fitnesses[worst_idx] = fitnesses[old_best_idx]
            population = new_population
            fitnesses = new_fitnesses
        
        # Update global best
        cur_best_idx = np.argmax(fitnesses)
        if fitnesses[cur_best_idx] > best_fit:
            best_fit = int(fitnesses[cur_best_idx])
            best_mask = population[cur_best_idx].copy()
    
    return best_mask

def construct():
    """Construct a 4-AP-free subset of [1, 256] maximizing 3-APs."""
    best_mask = genetic_algorithm(pop_size=40, generations=80, seed=42)
    
    result = sorted([i + 1 for i in range(N) if best_mask[i]])
    
    if not result:
        result = [1]
    
    return result
