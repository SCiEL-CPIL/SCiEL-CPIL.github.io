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
    
    # For efficiency, we represent a K by the set of 8 triple indices.
    # We also need to map each triple index to the list of K indices it belongs to.
    
    # Precomputation might be heavy for 120k Ks if done naively in pure Python lists of sets.
    # Let's try to be smart.
    
    # Instead of storing all Ks, we can just generate them on the fly? No, we need to check "is any K complete?".
    # Better: For each triple t, store the list of other 7 triples that would complete a K with t.
    # But a triple t is in many Ks.
    
    # Let's stick to the incidence map.
    # k_indices: list of sets of triple indices
    # triple_to_ks: list of lists of k indices
    
    # Generating 120,120 Ks and their 8 triples might take a few seconds.
    # Let's optimize generation.
    
    # We can iterate over all 6-subsets.
    # For each 6-subset, generate the 15 pairings.
    # For each pairing, generate the 8 triples.
    
    # To save memory and time, we can just do the greedy pass directly with a smart check?
    # No, the check "does adding t complete a K" requires knowing the other 7.
    
    # Let's build the incidence structure.
    triple_to_ks = [[] for _ in range(num_triples)]
    all_ks = [] # List of frozensets of triple indices
    
    # Optimization: We can just store the Ks as tuples of sorted triple indices.
    
    # This precomputation is O(120000 * 8). ~1M operations. Should be < 5s.
    
    for six_points in combinations(range(n), 6):
        # 15 ways to pair 6 items
        # Indices 0..5 in six_points
        # Pairings:
        # (0,1)(2,3)(4,5)
        # (0,1)(2,4)(3,5)
        # ...
        # We can generate these permutations or use a known list.
        # There are 15 perfect matchings on 6 vertices.
        
        # Hardcoding the 15 matchings of indices 0-5 is safer and faster.
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
        
        for matching in matchings:
            # Get the 3 pairs of actual points
            p1 = (six_points[matching[0][0]], six_points[matching[0][1]])
            p2 = (six_points[matching[1][0]], six_points[matching[1][1]])
            p3 = (six_points[matching[2][0]], six_points[matching[2][1]])
            
            # Generate 8 triples
            # Choose one from p1, one from p2, one from p3
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
                
    # Greedy Construction
    best_triples = set()
    best_count = 0
    
    # Run multiple random restarts to find a better greedy solution
    # Since the graph is dense, order matters significantly.
    
    max_restarts = 50
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        current_triples = set()
        indices = list(range(num_triples))
        random.shuffle(indices)
        
        for idx in indices:
            # Check if adding idx creates a K(2,2,2)
            # A K is created if there is a K containing idx where the other 7 are already in current_triples
            is_conflict = False
            for k_idx in triple_to_ks[idx]:
                k_set = all_ks[k_idx]
                # Check if (k_set - {idx}) is a subset of current_triples
                # Since len(k_set) is 8, we can just check if count of k_set in current_triples is 7
                # Faster: 
                # if len(k_set & current_triples) == 7:
                #     is_conflict = True
                #     break
                # But set intersection creates a new set.
                # Since k_set is small (8), we can iterate:
                count = 0
                for t in k_set:
                    if t == idx:
                        continue
                    if t in current_triples:
                        count += 1
                        if count == 7:
                            is_conflict = True
                            break
                if is_conflict:
                    break
            
            if not is_conflict:
                current_triples.add(idx)
                
        if len(current_triples) > best_count:
            best_count = len(current_triples)
            best_triples = current_triples
            
    # Convert indices back to triples
    result = [list(all_triples[i]) for i in best_triples]
    
    return result
# EVOLVE-BLOCK-END
