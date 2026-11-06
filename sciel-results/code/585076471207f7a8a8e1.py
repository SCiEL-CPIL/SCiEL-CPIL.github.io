"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute all K(2,2,2) configurations
    # A K(2,2,2) is defined by 3 disjoint pairs of points.
    # For each 6-subset and each of the 15 pairings, we get 8 triples.
    
    # For efficiency, we represent each K as a bitmask of 8 triple indices.
    # We also need a reverse map: for each triple, which Ks does it belong to?
    # But storing all Ks as sets of 8 indices is memory heavy (120k * 8 * 8 bytes ~ 7.6MB, ok).
    # However, checking if adding a triple completes a K requires checking if the other 7 are present.
    
    # Let's use a different strategy for the greedy step.
    # Instead of checking all Ks containing the new triple, we can maintain a count of how many
    # triples from each K are already in the solution.
    # If we add a triple t, for every K containing t, we increment the count.
    # If any count reaches 8, we have a conflict.
    
    # This requires:
    # 1. A list of all Ks (as sets of triple indices).
    # 2. For each triple, the list of K indices it belongs to.
    # 3. An array `k_counts` of length num_ks, tracking how many triples of each K are in the solution.
    
    # Let's build these structures.
    
    triple_to_ks = [[] for _ in range(num_triples)]
    all_ks = []  # List of sets of triple indices
    
    # Optimization: Generate Ks efficiently
    # We iterate over all 6-subsets.
    # For each 6-subset, we iterate over the 15 pairings.
    # For each pairing, we generate the 8 triples.
    
    # To speed up triple lookup, triple_to_idx is a dict.
    
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
    
    for six_points in combinations(range(n), 6):
        for matching in matchings:
            p1 = (six_points[matching[0][0]], six_points[matching[0][1]])
            p2 = (six_points[matching[1][0]], six_points[matching[1][1]])
            p3 = (six_points[matching[2][0]], six_points[matching[2][1]])
            
            t_indices = []
            for x1 in p1:
                for x2 in p2:
                    for x3 in p3:
                        t = tuple(sorted((x1, x2, x3)))
                        t_indices.append(triple_to_idx[t])
            
            k_set = frozenset(t_indices)
            k_idx = len(all_ks)
            all_ks.append(k_set)
            
            for ti in t_indices:
                triple_to_ks[ti].append(k_idx)
    
    num_ks = len(all_ks)
    
    # Greedy Construction with Local Search / Local Improvement
    best_triples = set()
    best_count = 0
    
    # We will run multiple random restarts.
    # In each restart, we use a greedy approach with a specific ordering.
    # Then we try to improve by adding more triples that don't conflict.
    # Or we can use a "maximal" greedy where we keep scanning until no more can be added.
    
    max_restarts = 200  # Increase restarts since the check is now O(1) amortized per addition if we maintain counts
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        # k_counts tracks how many triples of each K are in the current solution
        k_counts = np.zeros(num_ks, dtype=np.int8)
        current_triples = set()
        
        # Random order
        indices = list(range(num_triples))
        random.shuffle(indices)
        
        # Phase 1: Greedy Addition
        # We iterate through indices. If adding idx doesn't cause any k_count to exceed 7, we add it.
        # Wait, if we add idx, k_count for all Ks containing idx increases by 1.
        # If any of those Ks already had 7, it becomes 8 -> Conflict.
        # So the check is: for all k in triple_to_ks[idx], is k_counts[k] < 7?
        
        for idx in indices:
            # Check if we can add idx
            can_add = True
            for k_idx in triple_to_ks[idx]:
                if k_counts[k_idx] == 7:
                    can_add = False
                    break
            
            if can_add:
                current_triples.add(idx)
                for k_idx in triple_to_ks[idx]:
                    k_counts[k_idx] += 1
        
        # Phase 2: Local Improvement (try to add more)
        # The greedy pass above is maximal with respect to the order.
        # But maybe we can swap? Or just try to add any remaining triple that fits?
        # Actually, the greedy pass iterates through ALL indices. So if a triple wasn't added,
        # it was because at that moment, it would conflict.
        # However, as we add other triples later, does the "conflict" status change?
        # No, adding more triples only increases k_counts, making it harder to add more.
        # So the set produced by the greedy pass is maximal (no single triple can be added).
        
        # To improve, we might need to remove some triples to add others (2-opt style).
        # But that's complex. Let's stick to more restarts with different orderings.
        
        if len(current_triples) > best_count:
            best_count = len(current_triples)
            best_triples = current_triples.copy()
            
            # Early termination if we hit a theoretical bound?
            # Max possible is C(16,3) = 560.
            # If we get close, we can stop.
            if best_count > 400:
                 # Just a heuristic, keep going a bit more to be safe
                 pass

    # Convert indices back to triples
    result = [list(all_triples[i]) for i in best_triples]
    
    return result
# EVOLVE-BLOCK-END
