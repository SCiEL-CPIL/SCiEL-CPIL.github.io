"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n = 16
    # 1. Generate all possible triples (indices 0..559)
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # 2. Generate all K(2,2,2) configurations
    # A K(2,2,2) is defined by 3 disjoint pairs of points.
    # We need to choose 6 distinct points, then partition them into 3 pairs.
    # Number of ways to choose 6 points: C(16, 6) = 8008
    # Number of ways to partition 6 points into 3 pairs: 15
    # Total Ks: 8008 * 15 = 120120
    
    # Hardcoding the 15 matchings of indices 0-5
    matchings = [
        [(0,1),(2,3),(4,5)],
        [(0,1),(2,4),(3,5)],
        [(0,1),(2,5),(3,4)],
        [(0,2),(1,3),(4,5)],
        [(0,2),(1,4),(3,5)],
        [(0,2),(1,5),(3,4)],
        [(0,3),(1,2),(4,5)],
        [(0,3),(1,4),(2,5)],
        [(0,3),(1,5),(2,4)],
        [(0,4),(1,2),(3,5)],
        [(0,4),(1,3),(2,5)],
        [(0,4),(1,5),(2,3)],
        [(0,5),(1,2),(3,4)],
        [(0,5),(1,3),(2,4)],
        [(0,5),(1,4),(2,3)]
    ]
    
    # Precompute for each triple index, a list of masks of the OTHER 7 triples
    # in each K(2,2,2) that contains it.
    # mask_7 is an integer where bits are set for the indices of the other 7 triples.
    triple_to_other_masks = [[] for _ in range(num_triples)]
    
    # We also need the bit for each triple itself for the current set mask
    triple_bits = [1 << i for i in range(num_triples)]
    
    # Precomputation is the bottleneck initially, but it's done once.
    # 8008 * 15 = 120,120 Ks.
    # For each K, we generate 8 triples.
    # For each of the 8 triples, we compute the mask of the other 7.
    
    for six_points in combinations(range(n), 6):
        for matching in matchings:
            p1 = (six_points[matching[0][0]], six_points[matching[0][1]])
            p2 = (six_points[matching[1][0]], six_points[matching[1][1]])
            p3 = (six_points[matching[2][0]], six_points[matching[2][1]])
            
            # Generate the 8 triple indices
            t_indices = []
            for x1 in p1:
                for x2 in p2:
                    for x3 in p3:
                        t = tuple(sorted((x1, x2, x3)))
                        t_indices.append(triple_to_idx[t])
                        
            # For each triple in this K, the mask of the other 7 is (full_mask ^ bit_i)
            # But we need to store it in triple_to_other_masks[t_i]
            # full_mask is the bitmask of all 8 indices
            full_mask = 0
            for idx in t_indices:
                full_mask |= triple_bits[idx]
                
            for idx in t_indices:
                # The mask of the other 7 is full_mask with the bit for idx cleared
                other_mask = full_mask ^ triple_bits[idx]
                triple_to_other_masks[idx].append(other_mask)
                
    # Greedy Construction with Bitmasks
    best_mask = 0
    best_count = 0
    
    # Helper to check if a triple can be added to a given mask
    # Returns True if adding idx to current_mask does not complete any K
    def can_add(idx, current_mask):
        # For each K containing idx, check if the other 7 are in current_mask
        for other_mask in triple_to_other_masks[idx]:
            if (current_mask & other_mask) == other_mask:
                return False
        return True

    # Run multiple random restarts
    while time.time() < deadline - 5.0: # Leave 5s for final polish and return
        current_mask = 0
        indices = list(range(num_triples))
        random.shuffle(indices)
        
        for idx in indices:
            if can_add(idx, current_mask):
                current_mask |= triple_bits[idx]
                
        count = current_mask.bit_count()
        if count > best_count:
            best_count = count
            best_mask = current_mask

    # Local Improvement / Polish
    # Try to add any remaining triple that fits.
    # Repeat until no more can be added.
    # We do this on the best_mask found.
    changed = True
    polish_passes = 0
    while changed and polish_passes < 50 and time.time() < deadline - 1.0:
        changed = False
        # We need to check all triples not in best_mask
        # To make it faster, we can iterate through all and check
        for idx in range(num_triples):
            if not (best_mask & triple_bits[idx]):
                if can_add(idx, best_mask):
                    best_mask |= triple_bits[idx]
                    changed = True
        polish_passes += 1
        
    # Update best_count after polish
    final_count = best_mask.bit_count()
    
    # Convert best_mask back to list of triples
    result = []
    for i in range(num_triples):
        if best_mask & triple_bits[i]:
            result.append(list(all_triples[i]))
            
    return result
# EVOLVE-BLOCK-END
