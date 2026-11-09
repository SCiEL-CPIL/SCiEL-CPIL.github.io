"""Search for a size-efficient 16-wire sorting network using layer-based genetic algorithm."""
import time
import random
import numpy as np

N = 16
ALL_INPUTS = 1 << N  # 65536
TIME_LIMIT = 78.0    # safety margin under 90s kill
_start = time.time()


def _elapsed():
    return time.time() - _start


def _apply_comparators(inputs, comps):
    """Apply a list of (i,j) comparators to a (65536, 16) uint8 matrix."""
    x = inputs
    for i, j in comps:
        a = x[:, i]
        b = x[:, j]
        mn = np.minimum(a, b)
        mx = np.maximum(a, b)
        x[:, i] = mn
        x[:, j] = mx
    return x


def _is_sorted(x):
    """True if every row is non-decreasing."""
    return bool(np.all(x[:, :-1] <= x[:, 1:]))


def _verify(comps):
    """Return True if the comparator list sorts all 65536 Boolean inputs."""
    bits = np.arange(ALL_INPUTS, dtype=np.uint32)[:, None]
    shifts = (np.arange(N - 1, -1, -1, dtype=np.uint32))[None, :]
    x = ((bits >> shifts) & 1).astype(np.uint8)
    x = _apply_comparators(x, comps)
    return _is_sorted(x)


def _baseline_network():
    """Odd-even transposition network: 120 comparators."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _layer_from_perm(perm):
    """Create a layer (set of non-overlapping comparators) from a permutation."""
    # perm is a permutation of 0..15; pair adjacent elements
    comps = []
    for k in range(N // 2):
        a = perm[2 * k]
        b = perm[2 * k + 1]
        if a > b:
            a, b = b, a
        comps.append((a, b))
    return comps


def _network_from_layers(layers):
    """Flatten layers into a single comparator list."""
    comps = []
    for layer in layers:
        comps.extend(layer)
    return comps


def _random_layer(rng):
    """Generate a random layer."""
    perm = list(range(N))
    rng.shuffle(perm)
    return _layer_from_perm(perm)


def _count_inversions(comps):
    """Count total inversions remaining after applying comparators to all inputs.
    Used as a fitness proxy."""
    bits = np.arange(ALL_INPUTS, dtype=np.uint32)[:, None]
    shifts = (np.arange(N - 1, -1, -1, dtype=np.uint32))[None, :]
    x = ((bits >> shifts) & 1).astype(np.uint8)
    x = _apply_comparators(x, comps)
    # Count inversions: for each row, count pairs (i,j) where i<j but x[i]>x[j]
    # Since values are 0/1, inversion = x[i]=1 and x[j]=0 for i<j
    # Sum over all adjacent and non-adjacent pairs
    total = 0
    for i in range(N):
        for j in range(i + 1, N):
            total += int(np.sum((x[:, i] == 1) & (x[:, j] == 0)))
    return total


def _fitness(comps):
    """Fitness = -inversion_count (higher is better). If sorted, inversion=0."""
    inv = _count_inversions(comps)
    return -inv


def _ga_search(rng, time_budget):
    """Genetic algorithm over layer sequences."""
    n_layers = 8  # start with 8 layers
    pop_size = 30
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        layers = [_random_layer(rng) for _ in range(n_layers)]
        population.append(layers)
    
    best_network = None
    best_fitness = -1
    best_len = 120
    
    gen = 0
    while _elapsed() < time_budget:
        gen += 1
        # Evaluate fitness
        scored = []
        for layers in population:
            comps = _network_from_layers(layers)
            f = _fitness(comps)
            scored.append((f, layers, comps))
        
        # Track best
        scored.sort(key=lambda x: x[0], reverse=True)
        best_f, best_layers, best_comps = scored[0]
        if best_f > best_fitness:
            best_fitness = best_f
            best_network = best_comps
            best_len = len(best_comps)
        
        # Check if we found a valid network
        if best_fitness == 0:  # zero inversions means sorted
            if len(best_comps) < best_len:
                best_len = len(best_comps)
        
        # Selection: keep top half
        elite = [x[1] for x in scored[:pop_size // 2]]
        
        # Generate offspring
        new_pop = list(elite)
        while len(new_pop) < pop_size:
            if _elapsed() > time_budget:
                break
            # Crossover: combine layers from two parents
            p1 = rng.choice(elite)
            p2 = rng.choice(elite)
            child = []
            for i in range(len(p1)):
                if rng.random() < 0.5:
                    child.append(list(p1[i]))
                else:
                    child.append(list(p2[i]))
            
            # Mutation: replace a random layer
            if rng.random() < 0.7:
                mi = rng.randrange(len(child))
                child[mi] = _random_layer(rng)
            
            # Occasionally add or remove a layer
            if rng.random() < 0.1 and len(child) < 12:
                child.append(_random_layer(rng))
            elif rng.random() < 0.05 and len(child) > 5:
                ri = rng.randrange(len(child))
                child.pop(ri)
            
            new_pop.append(child)
        
        population = new_pop
    
    return best_network, best_len


def _local_search(comps, rng, time_budget):
    """Local search: try deletions and single-comparator replacements."""
    cur = list(comps)
    best = list(cur)
    best_len = len(cur)
    
    # First, verify it's valid
    if not _verify(cur):
        return None, 9999
    
    # Deletion pass
    changed = True
    while changed and _elapsed() < time_budget:
        changed = False
        for idx in range(len(cur) - 1, -1, -1):
            if _elapsed() > time_budget:
                break
            cand = cur[:idx] + cur[idx + 1:]
            if _verify(cand):
                cur = cand
                changed = True
                if len(cur) < best_len:
                    best_len = len(cur)
                    best = list(cur)
    
    # Replacement search: try replacing one comparator with another
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    for _ in range(200):
        if _elapsed() > time_budget:
            break
        idx = rng.randrange(len(cur))
        new_comp = all_pairs[rng.randrange(len(all_pairs))]
        if new_comp == cur[idx]:
            continue
        cand = list(cur)
        cand[idx] = new_comp
        if _verify(cand):
            cur = cand
            # Try deletion after replacement
            for d in range(len(cur) - 1, -1, -1):
                if _elapsed() > time_budget:
                    break
                cand2 = cur[:d] + cur[d + 1:]
                if _verify(cand2):
                    cur = cand2
                    if len(cur) < best_len:
                        best_len = len(cur)
                        best = list(cur)
                    break
    
    return best, best_len


def construct():
    rng = random.Random(42)
    best = _baseline_network()
    best_len = len(best)
    
    # Phase 1: GA search over layer sequences
    ga_budget = min(40.0, TIME_LIMIT * 0.5)
    ga_network, ga_len = _ga_search(rng, ga_budget)
    
    if ga_network is not None and ga_len < best_len:
        best = ga_network
        best_len = ga_len
    
    # Phase 2: Local search on the best found
    remaining = TIME_LIMIT - _elapsed() - 5.0  # leave 5s buffer
    if remaining > 5.0:
        ls_best, ls_len = _local_search(best, rng, _elapsed() + remaining)
        if ls_best is not None and ls_len < best_len:
            best = ls_best
            best_len = ls_len
    
    # Phase 3: Final deletion pass
    remaining = TIME_LIMIT - _elapsed() - 2.0
    if remaining > 2.0:
        cur = list(best)
        if _verify(cur):
            changed = True
            while changed and _elapsed() < _start + TIME_LIMIT:
                changed = False
                for idx in range(len(cur) - 1, -1, -1):
                    cand = cur[:idx] + cur[idx + 1:]
                    if _verify(cand):
                        cur = cand
                        changed = True
                        if len(cur) < best_len:
                            best_len = len(cur)
                            best = list(cur)
                        break
    
    # Ensure validity; fallback to baseline
    result = best
    if len(result) > 120:
        result = _baseline_network()
    
    return result
