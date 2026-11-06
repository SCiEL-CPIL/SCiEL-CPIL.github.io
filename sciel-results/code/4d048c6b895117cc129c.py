import time
import random
import itertools
import math

def _compute_alpha(triples):
    """
    Compute the independence number of the triple system.
    Returns the size of the largest subset of points containing no triple.
    """
    if not triples:
        return 16
        
    # Precompute masks for the triples
    triple_masks = []
    for t in triples:
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(m)
    
    n = 16
    # Check for independent sets of size k, starting from largest
    # We use itertools.combinations to generate subsets of size k
    # If we find one, that's the alpha.
    # To optimize, we can check k from 16 down to 1.
    
    # Optimization: 
    # If k is large, there are few subsets.
    # If k is small, there are many, but we stop early if we find one.
    # However, alpha is likely around 6-8.
    
    # Let's iterate k from 16 down to 1
    for k in range(16, 0, -1):
        # Generate all combinations of size k
        # To make it faster, we can break early if found
        found = False
        for comb in itertools.combinations(range(16), k):
            mask = 0
            for p in comb:
                mask |= (1 << p)
            
            # Check if any triple is fully contained in mask
            independent = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    independent = False
                    break
            
            if independent:
                return k
        
    return 0

def _is_linear(triples):
    """Check that any two triples share at most one point."""
    # Using pair count is faster
    pair_counts = {}
    for t in triples:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = t[i], t[j]
                key = (min(a, b), max(a, b))
                pair_counts[key] = pair_counts.get(key, 0) + 1
                if pair_counts[key] > 1:
                    return False
    return True

def _greedy_packing(seed=0):
    """
    Build a maximal linear triple system using a greedy randomized approach.
    """
    rng = random.Random(seed)
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    rng.shuffle(all_triples)
    
    chosen = []
    used_pairs = set()
    
    for t in all_triples:
        valid = True
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = t[i], t[j]
                key = (min(a, b), max(a, b))
                if key in used_pairs:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = t[i], t[j]
                    key = (min(a, b), max(a, b))
                    used_pairs.add(key)
            chosen.append(list(t))
            
    return chosen

def _local_search(triples, start_temp=1.0, max_iters=2000, seed=0):
    """
    Simulated Annealing to minimize alpha.
    """
    rng = random.Random(seed)
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    current = [list(t) for t in triples]
    current_alpha = _compute_alpha(current)
    best_alpha = current_alpha
    best_triples = [list(t) for t in current]
    
    # Precompute pair masks for all triples to speed up compatibility check?
    # Or just use sets of pairs.
    
    def get_used_pairs(triples):
        used = set()
        for t in triples:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = t[i], t[j]
                    used.add((min(a, b), max(a, b)))
        return used
    
    used_pairs = get_used_pairs(current)
    
    temp = start_temp
    cooling = 0.99
    
    for i in range(max_iters):
        # Pick a random triple to remove
        idx = rng.randrange(len(current))
        removed = current[idx]
        
        # Remove its pairs
        for k in range(3):
            for l in range(k + 1, 3):
                a, b = removed[k], removed[l]
                used_pairs.discard((min(a, b), max(a, b)))
        
        # Find a random replacement triple that is compatible
        # To be efficient, we pick a random triple from all_triples and check
        # If it fails, we might try a few more or just skip
        # Better: Filter candidates? 560 triples is small.
        
        # Shuffle a slice of all_triples to pick a random valid one
        candidates = []
        # For speed, we can just iterate through a shuffled list and take the first valid one
        # Or pick random ones until we find a valid one
        # Since the system is dense, many triples will be invalid.
        # Let's pre-shuffle all_triples once per iteration? No, that's 560 log 560.
        # Let's just pick random triples and check.
        
        new_triple = None
        for _ in range(50): # Try 50 times to find a valid replacement
            t = rng.choice(all_triples)
            valid = True
            for k in range(3):
                for l in range(k + 1, 3):
                    a, b = t[k], t[l]
                    if (min(a, b), max(a, b)) in used_pairs:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                new_triple = list(t)
                break
        
        if new_triple is None:
            # Restore removed triple
            for k in range(3):
                for l in range(k + 1, 3):
                    a, b = removed[k], removed[l]
                    used_pairs.add((min(a, b), max(a, b)))
            current[idx] = removed
            continue
            
        # Apply change
        current[idx] = new_triple
        for k in range(3):
            for l in range(k + 1, 3):
                a, b = new_triple[k], new_triple[l]
                used_pairs.add((min(a, b), max(a, b)))
                
        new_alpha = _compute_alpha(current)
        
        if new_alpha < current_alpha:
            current_alpha = new_alpha
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_triples = [list(t) for t in current]
        else:
            # Simulated Annealing acceptance
            delta = new_alpha - current_alpha
            if delta > 0:
                prob = math.exp(-delta / temp)
                if rng.random() > prob:
                    # Reject, restore
                    used_pairs.discard((min(new_triple[0], new_triple[1]), max(new_triple[0], new_triple[1])))
                    used_pairs.discard((min(new_triple[0], new_triple[2]), max(new_triple[0], new_triple[2])))
                    used_pairs.discard((min(new_triple[1], new_triple[2]), max(new_triple[1], new_triple[2])))
                    
                    used_pairs.add((min(removed[0], removed[1]), max(removed[0], removed[1])))
                    used_pairs.add((min(removed[0], removed[2]), max(removed[0], removed[2])))
                    used_pairs.add((min(removed[1], removed[2]), max(removed[1], removed[2])))
                    
                    current[idx] = removed
                else:
                    # Accept
                    current_alpha = new_alpha
            else:
                # Accept (should be handled by < case, but delta==0)
                current_alpha = new_alpha
        
        temp *= cooling
        
    return best_triples, best_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s limit
    
    best_triples = None
    best_alpha = 16
    
    # Phase 1: Generate several greedy packings and pick the best
    # Greedy packings vary in size and structure
    for seed in range(20):
        if time.time() > deadline:
            break
        triples = _greedy_packing(seed)
        if not _is_linear(triples):
            continue
        alpha = _compute_alpha(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            
    # Phase 2: Local Search (Simulated Annealing) on the best solution
    if best_triples is not None:
        if time.time() < deadline - 10:
            improved, alpha = _local_search(best_triples, start_temp=0.5, max_iters=500, seed=100)
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = improved
                
    # Phase 3: Try other greedy starts with local search if time permits
    for seed in range(20, 50):
        if time.time() > deadline:
            break
        triples = _greedy_packing(seed)
        if not _is_linear(triples):
            continue
        alpha = _compute_alpha(triples)
        
        # Only do local search if this is promising or if we have time
        if alpha <= best_alpha:
            if time.time() < deadline - 5:
                improved, a2 = _local_search(triples, start_temp=0.5, max_iters=200, seed=seed + 500)
                if a2 < best_alpha:
                    best_alpha = a2
                    best_triples = improved
        else:
            # Even if worse, maybe local search can fix it?
            # But to save time, only do it if we are close to deadline or if alpha is not too bad
            if alpha <= best_alpha + 1 and time.time() < deadline - 5:
                improved, a2 = _local_search(triples, start_temp=0.5, max_iters=200, seed=seed + 500)
                if a2 < best_alpha:
                    best_alpha = a2
                    best_triples = improved

    if best_triples is None:
        # Fallback
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
