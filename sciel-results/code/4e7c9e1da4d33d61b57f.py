"""Construct a linear triple system on 16 points with small independence number."""
import itertools

# EVOLVE-BLOCK-START

def _independence_number(triples, n=16):
    """
    Compute the independence number (largest subset containing no triple)
    for a hypergraph on n points, given as a list of triples.
    Uses bitmask enumeration for efficiency.
    """
    if not triples:
        return n

    # Build a set of bitmask representations of triples
    triple_masks = set()
    for t in triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.add(mask)

    # We check subsets by descending size to find the maximum quickly
    # But 2^16 = 65536 is small, so we can just iterate all masks
    # To be faster, we can iterate by size descending and stop at first independent set
    max_indep = 0

    # Precompute popcount for all masks? Or use bit_count (Python 3.10+)
    # For compatibility, let's use bin(x).count("1") or a lookup if needed.
    # Actually, iterating all 65536 masks and checking subset is fast enough.
    
    # Optimization: If a subset S is independent, all its subsets are independent.
    # So we can iterate masks in descending order of popcount.
    # However, generating masks by popcount is complex. 
    # Simple iteration is ~0.1s in Python for 65k masks with simple checks.
    
    # Let's try a slightly smarter check:
    # For a mask M, it contains a triple if any triple_mask is a subset of M.
    # triple_mask & M == triple_mask
    
    # We can precompute for each mask whether it's independent?
    # Or just check on the fly.
    
    # To speed up, we can group triple masks.
    # If we iterate masks from 2^n - 1 down to 0, the first independent mask has the max popcount?
    # No, we want max popcount among independent masks.
    # So we should iterate by popcount descending.
    
    # Let's generate masks by popcount descending.
    # There are C(16, k) masks of size k.
    # Max k is 16, min is 0.
    
    # We can use a recursive generator or just sort all masks by popcount descending.
    # Sorting 65536 integers is fast.
    
    masks = list(range(1 << n))
    masks.sort(key=lambda x: x.bit_count(), reverse=True)
    
    for mask in masks:
        pop = mask.bit_count()
        if pop <= max_indep:
            break # Since we are going descending, no future mask will have higher popcount
        
        # Check if mask contains any triple
        contains_triple = False
        for tm in triple_masks:
            if (tm & mask) == tm:
                contains_triple = True
                break
        
        if not contains_triple:
            max_indep = pop
            break # Found the largest independent set
            
    return max_indep

def _greedy_construct(ordering_key, n=16):
    """
    Greedily construct a linear triple system based on a given ordering of triples.
    Returns the list of triples.
    """
    all_triples = list(itertools.combinations(range(n), 3))
    
    # Sort based on the provided key
    all_triples.sort(key=ordering_key, reverse=True)
    
    triples = []
    pair_count = {}
    
    for t in all_triples:
        a, b, c = t
        pairs = [(a, b), (a, c), (b, c)]
        ok = True
        for p in pairs:
            if p in pair_count:
                ok = False
                break
        if ok:
            triples.append(t)
            for p in pairs:
                pair_count[p] = 1
                
    return triples

def _construct_algebraic():
    """
    Construct a linear triple system using algebraic properties of GF(2)^4.
    
    Strategy:
    1. Use a deterministic greedy approach.
    2. Prioritize triples that are "affine lines" or have high Hamming spread.
    3. The key insight is to cover points uniformly to minimize independence number.
    
    We will try several deterministic orderings and pick the best one.
    """
    n = 16
    
    # Heuristic 1: Maximize minimum Hamming distance between points in the triple.
    # This encourages triples to be "spread out" in the vector space.
    def spread_score(t):
        a, b, c = t
        # Hamming weight of XOR
        w_ab = (a ^ b).bit_count()
        w_bc = (b ^ c).bit_count()
        w_ac = (a ^ c).bit_count()
        return min(w_ab, w_bc, w_ac)
    
    # Heuristic 2: Maximize sum of Hamming distances.
    def sum_score(t):
        a, b, c = t
        return (a ^ b).bit_count() + (b ^ c).bit_count() + (a ^ c).bit_count()
    
    # Heuristic 3: Prefer triples that sum to 0 (affine lines through origin) or specific patterns?
    # In GF(2)^4, a+b+c=0 means c=a+b. These are lines through the origin (if 0 is included) or parallel lines.
    # Actually, any triple {a, b, a+b} is a line.
    # Let's check how many such triples exist.
    # For each pair (a,b) with a!=b and a+b != a,b (always true if a!=b!=0), we get a triple.
    # But we need distinct points.
    # Let's just use the spread score, which naturally favors lines in the Hamming cube.
    
    candidates = [
        _greedy_construct(spread_score, n),
        _greedy_construct(sum_score, n),
    ]
    
    # Also try a mixed heuristic: prioritize triples that include high-degree points?
    # Hard to do dynamically in a simple sort.
    
    # Let's also try a "reverse" sort for spread (minimize spread)? Unlikely to be better.
    
    best_triples = []
    best_indep = float('inf')
    
    for triples in candidates:
        indep = _independence_number(triples, n)
        if indep < best_indep:
            best_indep = indep
            best_triples = triples
    
    # If the best independence number is still high, try a local search or different heuristic.
    # For now, this deterministic approach should be fast and decent.
    
    return best_triples

def construct():
    triples = _construct_algebraic()
    return [list(t) for t in triples]

# EVOLVE-BLOCK-END
