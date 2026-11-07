"""Rewritten program using heuristic chain/layer decomposition for subset comparability maximization."""
# EVOLVE-BLOCK-START
import itertools

N = 12
M = 40

def construct():
    """
    Build a family of 40 subsets of {0,...,11} maximizing comparable pairs.
    
    Strategy: 
    1. Start with a maximal chain: {}, {0}, {0,1}, ..., {0,...,11} (13 sets)
    2. Greedily add sets that maximize the number of comparable pairs with existing sets.
    """
    # Precompute subset and superset relationships for all masks
    all_masks = list(range(1 << N))
    
    # For each mask, precompute which other masks are comparable
    # comparable[mask] = set of masks that are comparable with mask (strict subset or superset)
    comparable = [set() for _ in range(1 << N)]
    
    for mask in all_masks:
        # Proper subsets
        sub = mask
        while sub:
            if sub != mask:
                comparable[mask].add(sub)
                comparable[sub].add(mask)
            sub = (sub - 1) & mask
        # Empty set is subset of all
        comparable[mask].add(0)
        if mask != 0:
            comparable[0].add(mask)
        
        # Proper supersets
        complement = ((1 << N) - 1) ^ mask
        extra = complement
        while extra:
            sup = mask | extra
            if sup != mask:
                comparable[mask].add(sup)
                comparable[sup].add(mask)
            extra = (extra - 1) & complement
    
    # Start with a maximal chain
    family = set()
    result = []
    
    for i in range(N + 1):
        mask = (1 << i) - 1  # {0, 1, ..., i-1}
        if mask not in family:
            family.add(mask)
            result.append([j for j in range(N) if (mask >> j) & 1])
    
    # Greedily add remaining sets
    available = [m for m in all_masks if m not in family]
    
    while len(result) < M and available:
        best_mask = -1
        best_score = -1
        
        for mask in available:
            score = len(comparable[mask] & family)
            if score > best_score:
                best_score = score
                best_mask = mask
            elif score == best_score and best_mask != -1:
                # Tie-break: prefer sets with smaller size (more flexibility)
                if bin(mask).count('1') < bin(best_mask).count('1'):
                    best_mask = mask
        
        if best_mask == -1:
            break
        
        family.add(best_mask)
        result.append([j for j in range(N) if (best_mask >> j) & 1])
        available.remove(best_mask)
    
    # Ensure exactly M sets
    while len(result) < M:
        # This shouldn't happen, but just in case
        result.append([])
    
    return result[:M]
# EVOLVE-BLOCK-END
