"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def _independence_number(triples, n=16):
    """Compute the largest subset of points containing no triple using bitmasks."""
    if not triples:
        return n
    # Precompute triple bitmasks
    triple_masks = []
    for t in triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    # Check sizes from n down to 1
    for size in range(n, 0, -1):
        for subset in combinations(range(n), size):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            # Check if any triple is fully contained
            independent = True
            for tm in triple_masks:
                if (tm & mask) == tm:
                    independent = False
                    break
            if independent:
                return size
    return 0

def _is_linear(triples):
    """Check if any two triples share at most one point."""
    for i in range(len(triples)):
        for j in range(i+1, len(triples)):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True

def _generate_random_triples(n=16, num_triples=40, seed=42):
    """Generate random linear triples."""
    rng = random.Random(seed)
    triples = []
    attempts = 0
    max_attempts = 20000
    # Precompute pair-to-triples mapping for faster linearity check
    pair_to_triple = {}  # (a,b) -> index of triple containing both
    
    while len(triples) < num_triples and attempts < max_attempts:
        attempts += 1
        a, b, c = rng.sample(range(n), 3)
        t = tuple(sorted((a, b, c)))
        # Check linearity: no pair in t can already be in an existing triple
        linear = True
        for i in range(3):
            for j in range(i+1, 3):
                pair = (t[i], t[j])
                if pair in pair_to_triple:
                    linear = False
                    break
            if not linear:
                break
        if linear:
            idx = len(triples)
            triples.append(list(t))
            for i in range(3):
                for j in range(i+1, 3):
                    pair_to_triple[(t[i], t[j])] = idx
    
    return triples

def _algebraic_triples():
    """Generate triples from F2^4 structure: {a, b, a^b} for distinct nonzero a,b."""
    triples = []
    seen = set()
    for a in range(1, 16):
        for b in range(a+1, 16):
            c = a ^ b
            if c == 0 or c == a or c == b:
                continue
            t = tuple(sorted((a, b, c)))
            if t not in seen:
                seen.add(t)
                triples.append(list(t))
    # Also add triples involving point 0: {0, a, a} is invalid, so use other constructions
    # Add random triples to cover point 0
    rng = random.Random(123)
    pair_to_triple = {}
    for t in triples:
        for i in range(3):
            for j in range(i+1, 3):
                pair_to_triple[(t[i], t[j])] = True
    
    for _ in range(200):
        a, b, c = rng.sample(range(16), 3)
        t = tuple(sorted((a, b, c)))
        linear = True
        for i in range(3):
            for j in range(i+1, 3):
                if (t[i], t[j]) in pair_to_triple:
                    linear = False
                    break
            if not linear:
                break
        if linear:
            triples.append(list(t))
            for i in range(3):
                for j in range(i+1, 3):
                    pair_to_triple[(t[i], t[j])] = True
    return triples

def _local_search(triples, n=16, max_iter=500, seed=0, deadline=None):
    """Improve triples via local search to minimize independence number."""
    rng = random.Random(seed)
    triples = [list(t) for t in triples]
    
    if not _is_linear(triples):
        return triples
    
    # Build pair-to-triple-index map for O(1) linearity checks
    pair_to_idx = {}
    for i, t in enumerate(triples):
        for a in range(3):
            for b in range(a+1, 3):
                pair_to_idx[(t[a], t[b])] = i
    
    best_triples = [list(t) for t in triples]
    best_indep = _independence_number(triples, n)
    
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try to add a new random triple
        t = tuple(sorted(rng.sample(range(n), 3)))
        linear = True
        for i in range(3):
            for j in range(i+1, 3):
                if (t[i], t[j]) in pair_to_idx:
                    linear = False
                    break
            if not linear:
                break
        if linear:
            new_triples = triples + [list(t)]
            indep = _independence_number(new_triples, n)
            if indep < best_indep:
                best_indep = indep
                best_triples = [list(x) for x in new_triples]
                triples = new_triples
                # Update pair_to_idx
                idx = len(triples) - 1
                for i in range(3):
                    for j in range(i+1, 3):
                        pair_to_idx[(t[i], t[j])] = idx
        else:
            # Try to replace a random triple
            if triples:
                idx = rng.randint(0, len(triples)-1)
                old = triples[idx]
                # Remove old pairs from map
                for i in range(3):
                    for j in range(i+1, 3):
                        del pair_to_idx[(old[i], old[j])]
                
                # Try several replacements
                replaced = False
                for _ in range(20):
                    t = tuple(sorted(rng.sample(range(n), 3)))
                    linear = True
                    for i in range(3):
                        for j in range(i+1, 3):
                            if (t[i], t[j]) in pair_to_idx:
                                linear = False
                                break
                        if not linear:
                            break
                    if linear:
                        new_triples = triples[:idx] + [list(t)] + triples[idx+1:]
                        indep = _independence_number(new_triples, n)
                        if indep < best_indep:
                            best_indep = indep
                            best_triples = [list(x) for x in new_triples]
                            triples = new_triples
                            for i in range(3):
                                for j in range(i+1, 3):
                                    pair_to_idx[(t[i], t[j])] = idx
                        else:
                            # Restore old triple
                            triples[idx] = old
                            for i in range(3):
                                for j in range(i+1, 3):
                                    pair_to_idx[(old[i], old[j])] = idx
                        replaced = True
                        break
                
                if not replaced:
                    # Restore old triple
                    triples[idx] = old
                    for i in range(3):
                        for j in range(i+1, 3):
                            pair_to_idx[(old[i], old[j])] = idx
    
    return best_triples

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15 second safety margin
    
    best_triples = None
    best_indep = 16
    
    # Start with algebraic F2^4 construction
    alg = _algebraic_triples()
    if _is_linear(alg):
        best_triples = alg
        best_indep = _independence_number(alg, 16)
    
    # Also try random initial sets
    for seed in [42, 123, 777]:
        if time.time() > deadline:
            break
        initial = _generate_random_triples(16, 50, seed=seed)
        if _is_linear(initial):
            indep = _independence_number(initial, 16)
            if indep < best_indep:
                best_indep = indep
                best_triples = initial
    
    # Local search from multiple starting points
    start_points = []
    if best_triples:
        start_points.append(best_triples)
    
    for seed in range(30):
        if time.time() > deadline:
            break
        triples = _generate_random_triples(16, 50, seed=seed)
        if _is_linear(triples):
            start_points.append(triples)
    
    for i, sp in enumerate(start_points):
        if time.time() > deadline:
            break
        improved = _local_search(sp, 16, max_iter=300, seed=1000+i, deadline=deadline)
        indep = _independence_number(improved, 16)
        if indep < best_indep:
            best_indep = indep
            best_triples = improved
    
    if best_triples is None:
        # Fallback to simple valid system
        best_triples = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
# EVOLVE-BLOCK-END
