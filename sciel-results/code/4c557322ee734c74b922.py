"""Genetic algorithm with bitmask fitness for 3-AP maximization with no 4-AP constraint."""
import numpy as np
import time

N = 256

# Precompute all 3-APs and 4-APs
def precompute_progressions():
    """Precompute all 3-APs and 4-APs."""
    three_aps = []
    for a in range(1, N + 1):
        for d in range(1, N - a + 1):
            c = a + 2 * d
            if c <= N:
                three_aps.append((a - 1, a + d - 1, c - 1))
    three_aps = np.array(three_aps, dtype=np.int32)
    
    four_aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            c = a + 3 * d
            if c <= N:
                four_aps.append((a - 1, a + d - 1, a + 2 * d - 1, c - 1))
    four_aps = np.array(four_aps, dtype=np.int32)
    
    return three_aps, four_aps

THREE_APS, FOUR_APS = precompute_progressions()

# Precompute for each element, which 3-APs it participates in
ELEMENT_TO_3APS = {}
for i in range(N):
    ELEMENT_TO_3APS[i] = []
for idx, (a, b, c) in enumerate(THREE_APS):
    ELEMENT_TO_3APS[a].append(idx)
    ELEMENT_TO_3APS[b].append(idx)
    ELEMENT_TO_3APS[c].append(idx)

# Precompute for each element, which 4-APs it participates in
ELEMENT_TO_4APS = {}
for i in range(N):
    ELEMENT_TO_4APS[i] = []
for idx, (a, b, c, d) in enumerate(FOUR_APS):
    ELEMENT_TO_4APS[a].append(idx)
    ELEMENT_TO_4APS[b].append(idx)
    ELEMENT_TO_4APS[c].append(idx)
    ELEMENT_TO_4APS[d].append(idx)

def mask_to_bitset(mask):
    """Convert boolean mask to list of ints for bit operations."""
    return [int(mask[i]) for i in range(N)]

def count_3aps_mask(mask):
    """Count 3-APs using numpy vectorized check."""
    if len(THREE_APS) == 0:
        return 0
    valid = np.all(mask[THREE_APS], axis=1)
    return int(np.sum(valid))

def has_4ap_mask(mask):
    """Check if mask contains any 4-AP."""
    if len(FOUR_APS) == 0:
        return False
    valid = np.all(mask[FOUR_APS], axis=1)
    return bool(np.any(valid))

def find_4aps(mask):
    """Find indices of all 4-APs present in mask."""
    if len(FOUR_APS) == 0:
        return []
    valid = np.all(mask[FOUR_APS], axis=1)
    return np.where(valid)[0].tolist()

def repair_mask(mask):
    """Remove elements to eliminate all 4-APs, minimizing 3-AP loss."""
    mask = mask.copy()
    while has_4ap_mask(mask):
        four_ap_indices = find_4aps(mask)
        if not four_ap_indices:
            break
        
        # For each 4-AP, find the element whose removal loses the fewest 3-APs
        best_remove = None
        best_loss = float('inf')
        
        for ap_idx in four_ap_indices:
            ap = FOUR_APS[ap_idx]
            for elem in ap:
                # Count 3-APs that would be lost if we remove elem
                # A 3-AP is lost if elem is in it and the other two are in mask
                loss = 0
                for three_ap_idx in ELEMENT_TO_3APS[elem]:
                    a, b, c = THREE_APS[three_ap_idx]
                    if mask[a] and mask[b] and mask[c]:
                        loss += 1
                if loss < best_loss:
                    best_loss = loss
                    best_remove = elem
        
        if best_remove is not None:
            mask[best_remove] = False
        else:
            # Fallback: remove first element of first 4-AP
            mask[FOUR_APS[four_ap_indices[0]][0]] = False
    
    return mask

def greedy_4ap_free():
    """Greedy construction: add elements in order, skip if creates 4-AP."""
    mask = np.zeros(N, dtype=bool)
    for idx in range(N):
        mask[idx] = True
        if has_4ap_mask(mask):
            mask[idx] = False
    return mask

def random_4ap_free():
    """Random 4-AP-free subset via greedy with random order."""
    mask = np.zeros(N, dtype=bool)
    order = np.random.permutation(N)
    for idx in order:
        mask[idx] = True
        if has_4ap_mask(mask):
            mask[idx] = False
    return mask

def local_search(mask, max_iter=100):
    """Local search: try adding/removing/swapping elements."""
    best_mask = mask.copy()
    best_fit = count_3aps_mask(best_mask)
    
    for _ in range(max_iter):
        improved = False
        
        # Try adding elements (prioritize those that create many 3-APs)
        candidates = np.where(~best_mask)[0]
        if len(candidates) > 0:
            # Score each candidate by potential 3-AP gain
            scores = []
            for idx in candidates:
                test = best_mask.copy()
                test[idx] = True
                if not has_4ap_mask(test):
                    gain = count_3aps_mask(test) - best_fit
                    scores.append((gain, idx))
                else:
                    scores.append((-1, idx))
            
            scores.sort(reverse=True)
            for gain, idx in scores[:5]:
                if gain > 0:
                    test = best_mask.copy()
                    test[idx] = True
                    if not has_4ap_mask(test):
                        best_mask = test
                        best_fit += gain
                        improved = True
                        break
        
        # Try removing elements that contribute little
        candidates = np.where(best_mask)[0]
        if len(candidates) > 0:
            for idx in candidates:
                # Check if removing idx loses few 3-APs
                loss = 0
                for three_ap_idx in ELEMENT_TO_3APS[idx]:
                    a, b, c = THREE_APS[three_ap_idx]
                    if best_mask[a] and best_mask[b] and best_mask[c]:
                        loss += 1
                if loss == 0:
                    test = best_mask.copy()
                    test[idx] = False
                    new_fit = count_3aps_mask(test)
                    if new_fit >= best_fit:
                        best_mask = test
                        best_fit = new_fit
                        improved = True
                        break
        
        # Try swaps: remove one, add another
        if not improved:
            removed_candidates = np.where(best_mask)[0]
            added_candidates = np.where(~best_mask)[0]
            if len(removed_candidates) > 0 and len(added_candidates) > 0:
                # Try a few random swaps
                for _ in range(20):
                    rem = removed_candidates[np.random.randint(len(removed_candidates))]
                    add = added_candidates[np.random.randint(len(added_candidates))]
                    test = best_mask.copy()
                    test[rem] = False
                    test[add] = True
                    if not has_4ap_mask(test):
                        new_fit = count_3aps_mask(test)
                        if new_fit > best_fit:
                            best_mask = test
                            best_fit = new_fit
                            improved = True
                            break
        
        if not improved:
            break
    
    return best_mask

def crossover(m1, m2):
    """Uniform crossover with repair."""
    child = np.where(np.random.random(N) < 0.5, m1, m2)
    return repair_mask(child)

def mutate(mask, rate=0.05):
    """Bit-flip mutation with repair."""
    child = mask.copy()
    flip_mask = np.random.random(N) < rate
    child[flip_mask] = ~child[flip_mask]
    return repair_mask(child)

def genetic_algorithm(pop_size=40, generations=60, time_budget=50):
    """Main GA loop with time budget."""
    np.random.seed(42)
    start_time = time.time()
    
    # Initialize population with diversity
    population = [greedy_4ap_free()]
    for _ in range(pop_size - 1):
        population.append(random_4ap_free())
    
    fitnesses = np.array([count_3aps_mask(m) for m in population])
    best_mask = population[np.argmax(fitnesses)].copy()
    best_fit = int(np.max(fitnesses))
    
    for gen in range(generations):
        if time.time() - start_time > time_budget:
            break
        
        new_population = []
        
        for _ in range(pop_size):
            # Tournament selection
            t1 = np.random.randint(0, pop_size, 4)
            p1 = population[t1[np.argmax(fitnesses[t1])]]
            t2 = np.random.randint(0, pop_size, 4)
            p2 = population[t2[np.argmax(fitnesses[t2])]]
            
            # Crossover + mutation
            child = crossover(p1, p2)
            child = mutate(child, rate=0.03 + 0.04 * gen / generations)
            
            # Local search
            child = local_search(child, max_iter=30)
            
            new_population.append(child)
        
        new_fitnesses = np.array([count_3aps_mask(m) for m in new_population])
        
        # Elitism
        old_best = population[np.argmax(fitnesses)].copy()
        old_best_fit = int(np.max(fitnesses))
        
        new_best_idx = np.argmax(new_fitnesses)
        if new_fitnesses[new_best_idx] > old_best_fit:
            fitnesses = new_fitnesses
            population = new_population
            best_fit = int(new_fitnesses[new_best_idx])
            best_mask = population[new_best_idx].copy()
        else:
            # Replace worst with old best
            worst_idx = np.argmin(new_fitnesses)
            new_population[worst_idx] = old_best
            new_fitnesses[worst_idx] = old_best_fit
            population = new_population
            fitnesses = new_fitnesses
        
        # Update best
        cur_best_idx = np.argmax(fitnesses)
        if fitnesses[cur_best_idx] > best_fit:
            best_fit = int(fitnesses[cur_best_idx])
            best_mask = population[cur_best_idx].copy()
    
    return best_mask

def construct():
    """Construct a 4-AP-free subset of [1, 256] maximizing 3-APs."""
    best_mask = genetic_algorithm(pop_size=35, generations=50, time_budget=50)
    result = sorted([i + 1 for i in range(N) if best_mask[i]])
    if not result:
        result = [1]
    return result
