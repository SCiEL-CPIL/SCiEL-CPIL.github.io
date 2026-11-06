import itertools
import random
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    n = 16
    all_points = list(range(n))
    all_triples = list(itertools.combinations(all_points, 3))
    triple_masks = [(1 << a) | (1 << b) | (1 << c) for a, b, c in all_triples]
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute popcounts for all masks
    max_mask = 1 << n
    popcounts = [0] * max_mask
    for i in range(1, max_mask):
        popcounts[i] = popcounts[i >> 1] + (i & 1)

    best_alpha = n
    best_system = []
    rng = random.Random(42)

    def get_alpha_fast(system_indices):
        if not system_indices:
            return n
        
        sys_masks = [triple_masks[i] for i in system_indices]
        best_found = 0
        
        # Iterate all masks
        # We can't easily skip based on popcount without checking it first,
        # but we have popcounts array.
        for mask in range(max_mask):
            pc = popcounts[mask]
            if pc <= best_found:
                continue
            
            # Check if independent
            is_independent = True
            for sm in sys_masks:
                if (mask & sm) == sm:
                    is_independent = False
                    break
            
            if is_independent:
                best_found = pc
                if best_found == n:
                    return n
        
        return best_found

    while time.time() < deadline:
        rng.shuffle(all_triples)
        
        used_pairs = set()
        current_system_indices = []
        
        for t in all_triples:
            a, b, c = t
            pairs = {(a, b), (a, c), (b, c)}
            if not (used_pairs & pairs):
                current_system_indices.append(triple_to_idx[t])
                used_pairs |= pairs
                
        alpha = get_alpha_fast(current_system_indices)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = [list(all_triples[i]) for i in current_system_indices]
            
    return best_system
# EVOLVE-BLOCK-END
