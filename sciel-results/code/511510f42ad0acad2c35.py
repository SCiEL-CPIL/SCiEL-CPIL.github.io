"""Genetic algorithm approach for 3-AP maximization with no 4-AP constraint."""
import numpy as np
from itertools import combinations

# Precompute all 3-APs and 4-APs in [1, 256]
N = 256

def precompute_progressions():
    """Precompute all 3-APs and 4-APs as index triples/quadruples."""
    # 3-AP: (a, a+d, a+2d) with a >= 1, a+2d <= 256, d >= 1
    three_aps = []
    for a in range(1, N + 1):
        for d in range(1, N - a + 1):
            c = a + 2 * d
            if c <= N:
                three_aps.append((a - 1, a + d - 1, c - 1))  # 0-indexed
    three_aps = np.array(three_aps, dtype=np.int32)
    
    # 4-AP: (a, a+d, a+2d, a+3d) with a >= 1, a+3d <= 256, d >= 1
    four_aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            c = a + 3 * d
            if c <= N:
                four_aps.append((a - 1, a + d - 1, a + 2 * d - 1, c - 1))  # 0-indexed
    four_aps = np.array(four_aps, dtype=np.int32)
    
    # Inverse mapping: for each element, which 3-APs/4-APs does it belong to?
    inv_three_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(three_aps):
        for elem in ap:
            inv_three_aps[elem].append(idx)
            
    inv_four_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(four_aps):
        for elem in ap:
            inv_four_aps[elem].append(idx)

    return three_aps, four_aps, inv_three_aps, inv_four_aps

# Precompute at module level
THREE_APS, FOUR_APS, INV_THREE_APS, INV_FOUR_APS = precompute_progressions()

def count_three_aps(mask):
    """Count 3-APs in a bitmask (numpy boolean array of length N)."""
    if len(THREE_APS) == 0:
        return 0
    valid = np.all(mask[THREE_APS], axis=1)
    return int(np.sum(valid))

def has_four_ap(mask):
    """Check if mask contains any 4-AP."""
    if len(FOUR_APS) == 0:
        return False
    valid = np.all(mask[FOUR_APS], axis=1)
    return bool(np.any(valid))

def fitness(mask):
    """Fitness = number of 3-APs if no 4-AP, else negative large number."""
    if has_four_ap(mask):
        return -100000
    return count_three_aps(mask)

def greedy_4ap_free_subset():
    """Greedy: add elements in order of potential 3-AP contribution, skip if creates 4-AP."""
    mask = np.zeros(N, dtype=bool)
    # Calculate potential contribution of each element to 3-APs
    # An element i is in len(INV_THREE_APS[i]) 3-APs.
    # We prioritize elements with high 3-AP degree.
    degrees = np.array([len(inv) for inv in INV_THREE_APS])
    order = np.argsort(-degrees)
    
    for idx in order:
        mask[idx] = True
        if has_four_ap(mask):
            mask[idx] = False
    return mask

def local_search(mask, max_iter=100):
    """Improve mask via local search: add/remove/swap to increase 3-APs while maintaining 4-AP-free."""
    best_mask = mask.copy()
    best_fit = fitness(best_mask)
    
    for _ in range(max_iter):
        improved = False
        
        # Try adding every non-present element
        candidates = np.where(~best_mask)[0]
        if len(candidates) > 0:
            for idx in candidates:
                # Check if adding idx creates a 4-AP
                creates_4ap = False
                for ap_idx in INV_FOUR_APS[idx]:
                    ap = FOUR_APS[ap_idx]
                    # Check if the other 3 elements are in best_mask
                    others_in = all(best_mask[e] for e in ap if e != idx)
                    if others_in:
                        creates_4ap = True
                        break
                
                if not creates_4ap:
                    # Check gain in 3-APs
                    gain = 0
                    for ap_idx in INV_THREE_APS[idx]:
                        ap = THREE_APS[ap_idx]
                        others_in = all(best_mask[e] for e in ap if e != idx)
                        if others_in:
                            gain += 1
                    
                    if gain > 0:
                        best_mask[idx] = True
                        best_fit += gain
                        improved = True
                        break # Restart search after improvement
        
        # Try removing every present element
        candidates = np.where(best_mask)[0]
        if len(candidates) > 0:
            for idx in candidates:
                # Calculate loss in 3-APs
                loss = 0
                for ap_idx in INV_THREE_APS[idx]:
                    ap = THREE_APS[ap_idx]
                    others_in = all(best_mask[e] for e in ap if e != idx)
                    if others_in:
                        loss += 1
                
                # We only remove if loss is 0 (doesn't reduce 3-APs) and maybe helps future additions?
                # Actually, removing an element never helps 3-AP count directly.
                # It might help by allowing future additions that create more 3-APs.
                # But simple local search usually doesn't do "sacrificial" removals unless part of a swap.
                # Let's try swaps instead.
                pass

        # Try swaps: remove one, add another
        # This is complex to do efficiently in a single pass. 
        # Let's stick to Add-only for now, and maybe a simple Remove if it unblocks?
        # Actually, if we are stuck, we might need to remove an element that blocks many additions.
        # Let's try: for each present element, if removing it allows adding 2+ new elements with net gain > loss, do it.
        # This is O(N^2 * cost_check). N=256, so ~65k checks. Each check is small.
        
        # Simpler: just do Add. If stuck, break.
        if not improved:
             # Try a swap: pick a random present element to remove, then try to add best possible
             if len(candidates) > 0:
                 idx_remove = candidates[np.random.randint(len(candidates))]
                 test_mask = best_mask.copy()
                 test_mask[idx_remove] = False
                 
                 # Try to add any element
                 found_swap = False
                 for idx_add in np.where(~test_mask)[0]:
                     creates_4ap = False
                     for ap_idx in INV_FOUR_APS[idx_add]:
                         ap = FOUR_APS[ap_idx]
                         others_in = all(test_mask[e] for e in ap if e != idx_add)
                         if others_in:
                             creates_4ap = True
                             break
                     
                     if not creates_4ap:
                         gain = 0
                         for ap_idx in INV_THREE_APS[idx_add]:
                             ap = THREE_APS[ap_idx]
                             others_in = all(test_mask[e] for e in ap if e != idx_add)
                             if others_in:
                                 gain += 1
                         
                         # Calculate loss from removal
                         loss = 0
                         for ap_idx in INV_THREE_APS[idx_remove]:
                             ap = THREE_APS[ap_idx]
                             # Check if other elements are in test_mask (which has idx_remove removed)
                             others_in = all(test_mask[e] for e in ap if e != idx_remove)
                             if others_in:
                                 loss += 1
                         
                         if gain - loss > 0:
                             test_mask[idx_add] = True
                             best_mask = test_mask
                             best_fit += gain - loss
                             improved = True
                             found_swap = True
                             break
                 
                 if found_swap:
                     continue # Restart loop

             break

    return best_mask

def crossover(mask1, mask2):
    """Uniform crossover between two masks."""
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    cross_points = np.random.random(N) < 0.5
    child = np.where(cross_points, mask1, mask2)
    # Repair: remove any 4-APs
    while has_four_ap(child):
        valid = np.all(child[FOUR_APS], axis=1)
        ap_indices = np.where(valid)[0]
        if len(ap_indices) == 0:
            break
        ap_idx = ap_indices[np.random.randint(len(ap_indices))]
        ap = FOUR_APS[ap_idx]
        remove_idx = ap[np.random.randint(4)]
        child[remove_idx] = False
    return child

def mutate(mask, mutation_rate=0.1):
    """Random mutation: flip some bits, then repair 4-APs."""
    child = mask.copy().astype(bool)
    flip_mask = np.random.random(N) < mutation_rate
    child[flip_mask] = ~child[flip_mask]
    while has_four_ap(child):
        valid = np.all(child[FOUR_APS], axis=1)
        ap_indices = np.where(valid)[0]
        if len(ap_indices) == 0:
            break
        ap_idx = ap_indices[np.random.randint(len(ap_indices))]
        ap = FOUR_APS[ap_idx]
        remove_idx = ap[np.random.randint(4)]
        child[remove_idx] = False
    return child

def genetic_algorithm(pop_size=50, generations=100, max_local_iter=100):
    """Run genetic algorithm to find a good 4-AP-free subset."""
    np.random.seed(42)
    
    # Initialize population
    population = []
    population.append(greedy_4ap_free_subset())
    for _ in range(pop_size - 1):
        # Random initialization with repair
        mask = np.zeros(N, dtype=bool)
        # Randomly add elements
        order = np.random.permutation(N)
        for idx in order:
            mask[idx] = True
            if has_four_ap(mask):
                mask[idx] = False
        population.append(mask)
    
    fitnesses = np.array([fitness(mask) for mask in population])
    
    best_mask = population[np.argmax(fitnesses)]
    best_fit = np.max(fitnesses)
    
    for gen in range(generations):
        new_population = []
        for _ in range(pop_size):
            tournament = np.random.randint(0, pop_size, 3)
            winner = tournament[np.argmax(fitnesses[tournament])]
            parent1 = population[winner]
            
            tournament2 = np.random.randint(0, pop_size, 3)
            winner2 = tournament2[np.argmax(fitnesses[tournament2])]
            parent2 = population[winner2]
            
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate=0.05)
            child = local_search(child, max_iter=max_local_iter)
            
            new_population.append(child)
        
        new_fitnesses = np.array([fitness(mask) for mask in new_population])
        
        old_best_idx = np.argmax(fitnesses)
        new_best_idx = np.argmax(new_fitnesses)
        
        if new_fitnesses[new_best_idx] > fitnesses[old_best_idx]:
            fitnesses = new_fitnesses
            population = new_population
            best_fit = np.max(fitnesses)
            best_mask = population[np.argmax(fitnesses)]
        else:
            worst_idx = np.argmin(new_fitnesses)
            new_population[worst_idx] = population[old_best_idx]
            new_fitnesses[worst_idx] = fitnesses[old_best_idx]
            population = new_population
            fitnesses = new_fitnesses
        
        if np.max(fitnesses) > best_fit:
            best_fit = np.max(fitnesses)
            best_mask = population[np.argmax(fitnesses)]
    
    return best_mask

def construct():
    """Construct a 4-AP-free subset of [1, 256] maximizing 3-APs."""
    best_mask = genetic_algorithm(pop_size=30, generations=50, max_local_iter=50)
    result = sorted([i + 1 for i in range(N) if best_mask[i]])
    if not result:
        result = [1]
    return result

_CACHED_RESULT = None

def _ensure_cached():
    global _CACHED_RESULT
    if _CACHED_RESULT is None:
        _CACHED_RESULT = construct()
    return _CACHED_RESULT
