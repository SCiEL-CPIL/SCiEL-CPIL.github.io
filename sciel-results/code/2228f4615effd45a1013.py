import time
import random
import itertools
import sys

def _independence_number(triples, n=16):
    """Compute the independence number (largest set containing no triple) using branch and bound."""
    if not triples:
        return n
    
    # Precompute for each point, which triples contain it
    point_triples = [[] for _ in range(n)]
    for idx, t in enumerate(triples):
        for p in t:
            point_triples[p].append(idx)
    
    # For each point, precompute the set of points in triples with it (for conflict detection)
    # Actually, we need: for a candidate set S, S is independent iff no triple is fully contained in S.
    # We'll use a recursive search with pruning.
    
    best = 0
    remaining = set(range(n))
    chosen = []
    
    # Convert triples to frozensets for quick lookup
    triple_set = set(frozenset(t) for t in triples)
    
    # We need to check if adding a point creates a complete triple with existing chosen points
    # For efficiency, track for each pair of chosen points, whether adding a third creates a triple
    
    # Alternative: use a different approach. 
    # Let's use a simple but effective branch and bound.
    
    # For each subset, we need to check if it contains any triple.
    # With n=16, there are 2^16 = 65536 subsets. We can just enumerate all of them!
    
    # Actually, let's just enumerate all subsets and check if they're independent.
    # For each subset, check if any triple is a subset of it.
    
    # Precompute: for each triple, its bitmask
    triple_masks = []
    for t in triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)
    
    # For each subset mask, check if any triple mask is a subset
    # We can do this efficiently: for each subset, check if (subset & triple_mask) == triple_mask for any triple
    
    # But 65536 * len(triples) might be slow if len(triples) is large.
    # Let's use a different approach: for each triple, mark all subsets that contain it as "not independent"
    
    # Actually, let's just do the direct check for each subset.
    # 65536 subsets, and for each we check up to ~200 triples. 65536 * 200 = 13 million, which is fine.
    
    max_size = 0
    for mask in range(1 << n):
        # Quick size check
        size = bin(mask).count('1')
        if size <= max_size:
            continue
        
        # Check if any triple is fully contained
        is_independent = True
        for tm in triple_masks:
            if (mask & tm) == tm:
                is_independent = False
                break
        if is_independent:
            max_size = size
    
    return max_size


def _valid_system(triples):
    """Check if any two triples share at most one point."""
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            shared = len(set(triples[i]) & set(triples[j]))
            if shared > 1:
                return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    
    # Precompute compatibility: two triples are compatible if they share at most 1 point
    # We'll use this for greedy construction
    
    best_triples = None
    best_score = n + 1  # Independence number, we want to minimize
    
    # Strategy: Generate random packings (sets of mutually compatible triples)
    # and evaluate their independence number.
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        random.seed(restart * 7919 + 1)  # Deterministic for reproducibility
        
        # Greedy random construction
        available = list(range(len(all_triples)))
        random.shuffle(available)
        
        triples = []
        used_points_in_triples = {}  # point -> list of triples containing it
        
        for idx in available:
            if time.time() > deadline:
                break
            
            t = all_triples[idx]
            # Check compatibility with existing triples
            compatible = True
            for existing in triples:
                shared = len(set(t) & set(existing))
                if shared > 1:
                    compatible = False
                    break
            
            if compatible:
                triples.append(t)
        
        if not triples:
            continue
        
        # Evaluate independence number
        score = _independence_number(triples)
        
        if score < best_score:
            best_score = score
            best_triples = [list(t) for t in triples]
        
        if best_score <= 7:  # Can't do much better
            break
    
    if best_triples is None:
        # Fallback: return the seed solution
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
