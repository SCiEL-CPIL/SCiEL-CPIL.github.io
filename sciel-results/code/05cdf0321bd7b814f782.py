import time
import random
import itertools

def _compute_alpha(triples):
    """Compute independence number: largest subset with no triple fully contained."""
    # Represent each triple as a bitmask
    triple_masks = []
    for t in triples:
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(m)
    
    n = len(triple_masks)
    if n == 0:
        return 16
    
    # Check all subsets from largest to smallest
    # We want the max popcount of a mask S such that no triple is subset of S
    # Iterate over all 2^16 masks
    best = 0
    for mask in range(1 << 16):
        # Check if any triple is fully contained in mask
        independent = True
        for tm in triple_masks:
            if (mask & tm) == tm:
                independent = False
                break
        if independent:
            pc = bin(mask).count('1')
            if pc > best:
                best = pc
    return best

def _is_linear(triples):
    """Check that any two triples share at most one point."""
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True

def _greedy_build(seed=0):
    """Build a maximal linear triple system using greedy randomized approach."""
    rng = random.Random(seed)
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    rng.shuffle(all_triples)
    
    chosen = []
    # Track point usage: for each pair, how many triples contain it (must be <= 1 for linear)
    # Actually for linear: any two triples share at most 1 point.
    # So for each pair of points, at most one triple can contain that pair.
    pair_used = {}  # (a,b) -> True if a triple contains both a and b
    
    for t in all_triples:
        # Check if adding t violates linearity
        valid = True
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = t[i], t[j]
                key = (min(a, b), max(a, b))
                if key in pair_used:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = t[i], t[j]
                    key = (min(a, b), max(a, b))
                    pair_used[key] = True
            chosen.append(list(t))
    
    return chosen

def _local_search(triples, seed=0, max_iters=5000):
    """Try to improve by swapping triples."""
    rng = random.Random(seed)
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    current = [list(t) for t in triples]
    current_alpha = _compute_alpha(current)
    
    # Build pair_used for current
    def get_pair_used(triples):
        pair_used = set()
        for t in triples:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = t[i], t[j]
                    pair_used.add((min(a, b), max(a, b)))
        return pair_used
    
    pair_used = get_pair_used(current)
    
    for _ in range(max_iters):
        # Pick a random triple to remove
        idx = rng.randrange(len(current))
        removed = current[idx]
        
        # Remove its pairs
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = removed[i], removed[j]
                pair_used.discard((min(a, b), max(a, b)))
        
        # Find a random replacement triple that doesn't conflict
        candidates = []
        for t in all_triples:
            valid = True
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = t[i], t[j]
                    if (min(a, b), max(a, b)) in pair_used:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                candidates.append(t)
        
        if not candidates:
            # Restore
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = removed[i], removed[j]
                    pair_used.add((min(a, b), max(a, b)))
            current[idx] = removed
            continue
        
        new_triple = list(rng.choice(candidates))
        current[idx] = new_triple
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = new_triple[i], new_triple[j]
                pair_used.add((min(a, b), max(a, b)))
        
        new_alpha = _compute_alpha(current)
        if new_alpha < current_alpha:
            current_alpha = new_alpha
        # Keep the change regardless (accept all for exploration)
    
    return current, current_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_triples = None
    best_alpha = 16
    
    # Phase 1: Multiple greedy builds with different seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        triples = _greedy_build(seed)
        if not _is_linear(triples):
            continue
        alpha = _compute_alpha(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
    
    # Phase 2: Local search on the best solution
    if best_triples is not None:
        if time.time() < deadline - 5:
            improved, alpha = _local_search(best_triples, seed=42, max_iters=3000)
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = improved
    
    # Phase 3: More greedy + local search if time permits
    for seed in range(50, 200):
        if time.time() > deadline:
            break
        triples = _greedy_build(seed)
        if not _is_linear(triples):
            continue
        alpha = _compute_alpha(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
        # Quick local search
        if time.time() < deadline - 3:
            improved, a2 = _local_search(triples, seed=seed + 1000, max_iters=1000)
            if a2 < best_alpha:
                best_alpha = a2
                best_triples = improved
    
    if best_triples is None:
        # Fallback: simple disjoint triples
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
