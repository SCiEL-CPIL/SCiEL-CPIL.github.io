"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _independence_number(triples, n=16):
    """Compute the independence number: largest set with no triple fully contained."""
    # Convert to set for fast lookup
    triple_set = set(tuple(sorted(t)) for t in triples)
    
    # Try all subsets in decreasing size order
    # For n=16, we can be smart about this
    # Start from size n down to 1
    for size in range(n, 0, -1):
        for subset in combinations(range(n), size):
            # Check if any triple is contained in this subset
            has_triple = False
            for t in triple_set:
                if all(x in subset for x in t):
                    has_triple = True
                    break
            if not has_triple:
                return size
    return 1

def _independence_number_fast(triples, n=16):
    """Faster independence number computation using bitmasks."""
    if not triples:
        return n
    
    # Represent each point as a bit
    # For each triple, create a bitmask
    triple_masks = []
    for t in triples:
        mask = 0
        for x in t:
            mask |= (1 << x)
        triple_masks.append(mask)
    
    # Check subsets in decreasing size
    for size in range(n, 0, -1):
        for subset in combinations(range(n), size):
            subset_mask = 0
            for x in subset:
                subset_mask |= (1 << x)
            # Check if any triple is contained
            found = False
            for tm in triple_masks:
                if (tm & subset_mask) == tm:
                    found = True
                    break
            if not found:
                return size
    return 1

def _can_add(triples, new_triple):
    """Check if new_triple can be added without violating linearity."""
    new_set = set(new_triple)
    for t in triples:
        overlap = len(new_set & set(t))
        if overlap > 1:
            return False
    return True

def _greedy_construct(seed, deadline):
    """Randomized greedy construction with deadline guard."""
    rng = random.Random(seed)
    triples = []
    n = 16
    
    # Precompute all possible triples
    all_triples = list(combinations(range(n), 3))
    
    # Score function: prefer triples that help reduce independence number
    # We'll use a heuristic: add triples that cover points with high "exposure"
    
    while time.time() < deadline:
        # Find all valid triples that can be added
        valid_triples = [t for t in all_triples if _can_add(triples, t)]
        if not valid_triples:
            break
        
        # Heuristic: prefer triples that include points not well-covered
        # Count how many times each point appears in existing triples
        point_count = [0] * n
        for t in triples:
            for x in t:
                point_count[x] += 1
        
        # Score each valid triple: prefer those with lower total point count
        # (i.e., points that are less covered)
        def score_triple(t):
            return sum(point_count[x] for x in t)
        
        # Sort by score, take top candidates, then pick randomly
        valid_triples.sort(key=score_triple)
        top_k = min(10, len(valid_triples))
        candidate = rng.choice(valid_triples[:top_k])
        triples.append(list(candidate))
        
        # Early termination: if we've added many triples, check progress
        if len(triples) > 40:
            break
    
    return triples

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    best_triples = None
    best_indep = 17  # Start high
    
    # Try multiple random seeds
    for seed in range(100):
        if time.time() > deadline - 5.0:
            break
        
        triples = _greedy_construct(seed, deadline - 5.0)
        if not triples:
            continue
        
        indep = _independence_number_fast(triples)
        if indep < best_indep:
            best_indep = indep
            best_triples = triples
    
    if best_triples is None:
        # Fallback: return the original
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples

# EVOLVE-BLOCK-END
