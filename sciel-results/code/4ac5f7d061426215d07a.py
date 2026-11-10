"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def construct():
    """
    Construct a linear triple system on 16 points with small independence number.
    
    Strategy:
    1. Represent triples as bitmasks for efficient compatibility checks.
    2. Use a greedy clique expansion algorithm with a degree heuristic to build 
       a large set of pairwise compatible triples (linear system).
    3. Use bitmask precomputation to efficiently calculate the independence number.
    4. Run multiple iterations with randomization to find a system with a low independence number.
    """
    points = list(range(16))
    n_points = 16
    
    # Precompute all triples as bitmasks
    all_triples = []
    for t in itertools.combinations(points, 3):
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        all_triples.append(mask)
        
    n_triples = len(all_triples)
    
    # Precompute compatibility matrix or use bit operations on the fly?
    # On the fly is O(1) per check. (mask1 & mask2).bit_count() <= 1
    
    def compute_independence_number(triple_masks):
        """
        Compute the independence number using bitmask precomputation.
        A subset S is independent if no triple is fully contained in S.
        """
        if not triple_masks:
            return n_points
            
        # is_independent[subset_mask] = True if no triple is a subset of subset_mask
        # We start with all True, then mark False for any superset of a triple.
        is_independent = [True] * (1 << n_points)
        
        # For each triple, mark all its supersets as not independent
        for t_mask in triple_masks:
            # Iterate through all supersets of t_mask
            # A superset is formed by adding any subset of the remaining bits
            remaining = ((1 << n_points) - 1) ^ t_mask
            sub = remaining
            while True:
                superset = t_mask | sub
                is_independent[superset] = False
                if sub == 0:
                    break
                sub = (sub - 1) & remaining
        
        # Find the largest independent set
        max_size = 0
        for mask in range(1 << n_points):
            if is_independent[mask]:
                size = mask.bit_count()
                if size > max_size:
                    max_size = size
                    if max_size == n_points: # Can't get better
                        break
        return max_size

    def greedy_clique(seed=42):
        """
        Greedy clique expansion using degree heuristic.
        """
        rng = random.Random(seed)
        
        # List of available triples (indices)
        available = list(range(n_triples))
        rng.shuffle(available) # Initial randomization
        
        selected = []
        selected_masks = []
        
        # To speed up, we can maintain a list of available indices
        # But checking compatibility with all selected is O(K) where K is size of selected
        # Since max size is ~40, this is fast.
        
        while available:
            # Calculate degree for each available triple
            # Degree = number of other available triples compatible with this one
            # Note: We want to pick a triple that allows many future picks.
            # High degree in the remaining graph is a good heuristic.
            
            best_idx = -1
            best_degree = -1
            best_compatible_count = 0
            
            # Optimization: If the list is small, just pick the first compatible one?
            # No, degree heuristic is better for size.
            
            # To avoid O(N^2) in the inner loop if N is large, 
            # but N=560 is small enough. 560^2 * 40 iterations is ~12M ops, which is fine in Python?
            # 12M ops might take ~1s. We have 120s.
            
            # Let's refine: 
            # We pick the triple with the highest number of compatible neighbors in 'available'.
            
            # We can sample or just iterate.
            # Iterating 560 items, and for each checking against 560 others is 313k checks per step.
            # 40 steps * 313k = 12.5M checks.
            # Each check is (mask1 & mask2).bit_count() <= 1.
            # This should be fast enough.
            
            for i_idx, t_idx in enumerate(available):
                t_mask = all_triples[t_idx]
                deg = 0
                for j_idx, other_idx in enumerate(available):
                    if i_idx == j_idx:
                        continue
                    other_mask = all_triples[other_idx]
                    if (t_mask & other_mask).bit_count() <= 1:
                        deg += 1
                
                if deg > best_degree:
                    best_degree = deg
                    best_idx = i_idx
                    best_compatible_count = deg
                    
            if best_idx == -1:
                break
                
            # Select the best triple
            sel_idx = available.pop(best_idx)
            selected.append(sel_idx)
            sel_mask = all_triples[sel_idx]
            selected_masks.append(sel_mask)
            
            # Remove incompatible triples from available
            # We need to filter the list
            new_available = []
            for t_idx in available:
                t_mask = all_triples[t_idx]
                if (t_mask & sel_mask).bit_count() <= 1:
                    new_available.append(t_idx)
            available = new_available
            
        return selected_masks

    # Run multiple seeds
    best_score = n_points + 1
    best_triples = None
    
    # We can also try to improve the initial shuffle or use different heuristics
    # Let's run 100 iterations
    for seed in range(100):
        triples = greedy_clique(seed)
        if not triples:
            continue
            
        # Quick check: if we have few triples, independence number is likely high
        # But let's compute it properly
        score = compute_independence_number(triples)
        
        if score < best_score:
            best_score = score
            best_triples = triples
            
            # Early termination if we hit a theoretical lower bound?
            # For 16 points, what is the min independence number?
            # If we have a Steiner Triple System (which we don't), alpha would be small.
            # For random linear hypergraphs, alpha is roughly n / (2 * d^(1/2)) ?
            # Let's just run all 100.
            
    if best_triples is None:
        # Fallback
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
        
    # Convert back to list of lists for the interface
    return [list(bin(t).count('1') for _ in [0]) for t in []] # Placeholder, need to convert masks to lists
    
    # Proper conversion
    result = []
    for mask in best_triples:
        pts = []
        for i in range(16):
            if mask & (1 << i):
                pts.append(i)
        result.append(pts)
        
    return result

# EVOLVE-BLOCK-END
