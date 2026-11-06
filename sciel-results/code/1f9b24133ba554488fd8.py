import random
import time
import numpy as np

# Precompute all masks and popcount once at module level
_ALL_MASKS = np.arange(1 << 16, dtype=np.int32)
_POPCOUNT = np.zeros(1 << 16, dtype=np.int8)
for _i in range(1, 1 << 16):
    _POPCOUNT[_i] = _POPCOUNT[_i >> 1] + (_i & 1)

def compute_alpha_np(triples):
    """
    Computes the independence number of a linear triple system on 16 points.
    Uses NumPy for vectorized subset checking.
    """
    if not triples:
        return 16
        
    # Precompute masks for triples
    triple_masks = []
    for t in triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks.append(m)
    
    triple_masks_np = np.array(triple_masks, dtype=np.int32)
    
    # Check if any triple is contained in each mask
    intersects = (_ALL_MASKS[:, None] & triple_masks_np[None, :]) == triple_masks_np[None, :]
    has_triple = intersects.any(axis=1)
    independent = ~has_triple
    
    if not independent.any():
        return 0
        
    max_alpha = _POPCOUNT[independent].max()
    return int(max_alpha)

def generate_all_triples():
    """Generate all possible triples on 16 points."""
    points = range(16)
    triples = []
    for a in points:
        for b in points:
            if b <= a: continue
            for c in points:
                if c <= b: continue
                triples.append((a, b, c))
    return triples

def greedy_construct(all_triples, seed=None):
    """
    Randomized greedy construction of a linear triple system.
    """
    if seed is not None:
        random.seed(seed)
        
    candidates = list(all_triples)
    random.shuffle(candidates)
    
    used_pairs = set()
    triples = []
    
    for t in candidates:
        a, b, c = t
        pairs = [(a, b), (a, c), (b, c)]
        
        # Check if any pair is already used
        conflict = False
        for p in pairs:
            if p in used_pairs:
                conflict = True
                break
                
        if not conflict:
            triples.append(t)
            for p in pairs:
                used_pairs.add(p)
                
    return triples

def is_linear(triples):
    """Check if a set of triples is linear (any two share at most one point)."""
    n = len(triples)
    for i in range(n):
        for j in range(i+1, n):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True

def local_search(triples, all_triples, max_iter=5000, seed=None):
    """
    Local search to improve the independence number.
    Tries to swap or add triples to reduce alpha.
    """
    if seed is not None:
        random.seed(seed)
    
    if not triples:
        return triples
    
    # Make a copy
    current = [list(t) for t in triples]
    current_alpha = compute_alpha_np(current)
    
    # Build used pairs
    used_pairs = set()
    for t in current:
        a, b, c = t
        used_pairs.add((min(a,b), max(a,b)))
        used_pairs.add((min(a,c), max(a,c)))
        used_pairs.add((min(b,c), max(b,c)))
    
    # Available triples (those that don't conflict with current)
    # We'll try to swap: remove one triple, add another
    
    best = [list(t) for t in current]
    best_alpha = current_alpha
    
    for iteration in range(max_iter):
        # Pick a random triple to remove
        if not current:
            break
        idx = random.randint(0, len(current)-1)
        removed = current.pop(idx)
        
        # Remove its pairs from used_pairs
        a, b, c = removed
        used_pairs.discard((min(a,b), max(a,b)))
        used_pairs.discard((min(a,c), max(a,c)))
        used_pairs.discard((min(b,c), max(b,c)))
        
        # Find a random triple to add
        added = None
        for _ in range(100):  # Try up to 100 random candidates
            t = all_triples[random.randint(0, len(all_triples)-1)]
            x, y, z = t
            pairs = [(min(x,y), max(x,y)), (min(x,z), max(x,z)), (min(y,z), max(y,z))]
            if all(p not in used_pairs for p in pairs):
                added = t
                break
        
        if added is None:
            # Put back the removed triple
            current.append(removed)
            a, b, c = removed
            used_pairs.add((min(a,b), max(a,b)))
            used_pairs.add((min(a,c), max(a,c)))
            used_pairs.add((min(b,c), max(b,c)))
            continue
        
        # Add the new triple
        current.append(added)
        x, y, z = added
        used_pairs.add((min(x,y), max(x,y)))
        used_pairs.add((min(x,z), max(x,z)))
        used_pairs.add((min(y,z), max(y,z)))
        
        # Evaluate
        alpha = compute_alpha_np(current)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best = [list(t) for t in current]
        else:
            # Revert: remove added, put back removed
            current.pop()
            x, y, z = added
            used_pairs.discard((min(x,y), max(x,y)))
            used_pairs.discard((min(x,z), max(x,z)))
            used_pairs.discard((min(y,z), max(y,z)))
            current.append(removed)
            a, b, c = removed
            used_pairs.add((min(a,b), max(a,b)))
            used_pairs.add((min(a,c), max(a,c)))
            used_pairs.add((min(b,c), max(b,c)))
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    all_triples = generate_all_triples()
    
    best_alpha = 16
    best_triples = []
    
    # Phase 1: Random greedy with multiple seeds
    for i in range(50):
        if time.time() > deadline:
            break
        triples = greedy_construct(all_triples, seed=i)
        alpha = compute_alpha_np(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
    
    # Phase 2: Local search on the best solution found
    if best_triples and time.time() < deadline - 10:
        improved = local_search(best_triples, all_triples, max_iter=10000, seed=42)
        alpha = compute_alpha_np(improved)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = improved
    
    # Phase 3: More local search with different seeds if time allows
    if best_triples and time.time() < deadline - 5:
        for ls_seed in range(10):
            if time.time() > deadline - 5:
                break
            improved = local_search(best_triples, all_triples, max_iter=5000, seed=100+ls_seed)
            alpha = compute_alpha_np(improved)
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = improved
    
    return [list(t) for t in best_triples]
