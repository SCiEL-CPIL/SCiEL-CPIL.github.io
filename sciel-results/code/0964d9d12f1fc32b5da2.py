"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin before 150s kill
    
    n = 16
    # Generate all C(16,3) = 560 triples
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Map each triple to an index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # For each triple, find which K(2,2,2) configurations it participates in
    # A K(2,2,2) is defined by three disjoint pairs (a,a'), (b,b'), (c,c')
    # The 8 transversal triples are all combinations choosing one from each pair
    # A triple (x,y,z) is a transversal if {x,y,z} has exactly one element from each pair
    
    # Precompute for each triple, the list of K(2,2,2) configs it belongs to
    # First, generate all possible K(2,2,2) configurations
    # A config is determined by 3 disjoint pairs covering 6 distinct points
    # Number of ways: choose 6 points from 16, then partition into 3 ordered pairs
    # C(16,6) * 3! * 2^3 / ... actually let's think differently
    
    # For each set of 6 distinct points, there are 15 ways to partition into 3 pairs
    # (number of perfect matchings on 6 vertices is 15)
    # Total configs: C(16,6) * 15 = 8008 * 15 = 120120
    # That's too many to handle naively. Let's use a different approach.
    
    # Instead, for local search, we can check violations on the fly
    # or use a smarter representation.
    
    # Let's use a different strategy: greedy construction with local search
    # Start with a random subset, then iteratively add/remove triples to avoid K(2,2,2)
    
    # Better approach: start empty, greedily add triples that don't create a K(2,2,2)
    # Then try to improve by swapping
    
    # For checking if adding a triple creates a K(2,2,2):
    # A triple t=(a,b,c) creates a K(2,2,2) with existing triples iff there exist
    # three disjoint pairs (a,a'),(b,b'),(c,c') such that all 8 transversals are present.
    # Since we're adding one triple at a time, we need to check if the new triple
    # completes any K(2,2,2) that was previously missing exactly this one triple.
    
    # Let's precompute for each triple, the "conflict sets" - which other 7 triples
    # would complete a K(2,2,2) with this triple.
    
    # Actually, let's use a more efficient approach:
    # Represent the set of selected triples as a boolean array
    # For local search, track for each potential K(2,2,2) how many of its 8 triples are selected
    # A K(2,2,2) is violated when all 8 are selected
    
    # Given the large number of configs, let's use a sampling-based approach
    # or a different formulation.
    
    # Alternative: Use the fact that we want a large 3-uniform hypergraph on 16 vertices
    # with no K(2,2,2). This is related to extremal set theory.
    
    # Let me try a constructive approach based on known bounds.
    # ex_3(16, K(2,2,2)) - we want to maximize triples.
    
    # Strategy: Start with all triples, then remove minimal set to eliminate all K(2,2,2)
    # This is a hitting set problem on the hypergraph of K(2,2,2) configs.
    
    # For each K(2,2,2) config (8 triples), we need to remove at least 1.
    # Greedy: repeatedly find a violated config, remove a triple from it.
    
    # Let's implement this with efficient bookkeeping.
    
    # First, generate all K(2,2,2) configurations
    # A config is a set of 8 triples forming a K(2,2,2)
    # Defined by 3 disjoint pairs on 6 points
    
    configs = []  # List of frozensets of triple indices
    config_count = 0
    
    # Generate all ways to pick 6 distinct points and partition into 3 pairs
    for six_points in combinations(range(n), 6):
        # Generate all perfect matchings of 6 points
        # Number of perfect matchings on 6 points: 15
        p0, p1, p2, p3, p4, p5 = six_points
        matchings = [
            ((p0,p1),(p2,p3),(p4,p5)),
            ((p0,p1),(p2,p4),(p3,p5)),
            ((p0,p1),(p2,p5),(p3,p4)),
            ((p0,p2),(p1,p3),(p4,p5)),
            ((p0,p2),(p1,p4),(p3,p5)),
            ((p0,p2),(p1,p5),(p3,p4)),
            ((p0,p3),(p1,p2),(p4,p5)),
            ((p0,p3),(p1,p4),(p2,p5)),
            ((p0,p3),(p1,p5),(p2,p4)),
            ((p0,p4),(p1,p2),(p3,p5)),
            ((p0,p4),(p1,p3),(p2,p5)),
            ((p0,p4),(p1,p5),(p2,p3)),
            ((p0,p5),(p1,p2),(p3,p4)),
            ((p0,p5),(p1,p3),(p2,p4)),
            ((p0,p5),(p1,p4),(p2,p3)),
        ]
        for matching in matchings:
            pair1, pair2, pair3 = matching
            # Generate the 8 transversal triples
            transversals = []
            for x in pair1:
                for y in pair2:
                    for z in pair3:
                        t = tuple(sorted((x, y, z)))
                        transversals.append(t)
            # Convert to indices
            idx_set = frozenset(triple_to_idx[t] for t in transversals)
            configs.append(idx_set)
            config_count += 1
    
    # Now we have all configs. For local search, we track for each config
    # how many of its 8 triples are currently selected.
    
    # Start with all triples selected
    selected = [True] * num_triples
    
    # Count for each config how many of its triples are selected
    config_counts = [0] * len(configs)
    for i, cfg in enumerate(configs):
        count = sum(1 for idx in cfg if selected[idx])
        config_counts[i] = count
    
    # Number of violated configs (all 8 selected)
    violated = sum(1 for c in config_counts if c == 8)
    
    # Greedy removal: while there are violations, remove a triple
    # To be efficient, for each violated config, we can remove a triple that
    # appears in many other violated configs (to fix multiple at once)
    
    # Build reverse mapping: for each triple, which configs contain it
    triple_to_configs = [[] for _ in range(num_triples)]
    for i, cfg in enumerate(configs):
        for idx in cfg:
            triple_to_configs[idx].append(i)
    
    # Greedy: repeatedly pick a violated config, remove the triple in it
    # that is in the most other violated configs
    removed = set()
    
    # For efficiency, track which configs are currently violated
    violated_configs = set(i for i, c in enumerate(configs) if c == 8)
    
    while violated_configs and time.time() < deadline - 5.0:
        # Pick a random violated config
        cfg_idx = random.choice(list(violated_configs))
        cfg = configs[cfg_idx]
        
        # Find the triple in this config that appears in the most violated configs
        best_triple = None
        best_score = -1
        for t in cfg:
            if not selected[t]:
                continue
            # Count how many violated configs contain this triple
            score = sum(1 for ci in triple_to_configs[t] if ci in violated_configs)
            if score > best_score:
                best_score = score
                best_triple = t
        
        if best_triple is None:
            # All triples in this config are already removed? Shouldn't happen
            violated_configs.discard(cfg_idx)
            continue
        
        # Remove best_triple
        selected[best_triple] = False
        removed.add(best_triple)
        
        # Update config counts for all configs containing best_triple
        for ci in triple_to_configs[best_triple]:
            config_counts[ci] -= 1
            if config_counts[ci] == 8:
                violated_configs.add(ci)
            elif config_counts[ci] < 8:
                violated_configs.discard(ci)
    
    # Now try to add back some removed triples using local search
    # Try to re-add a removed triple if it doesn't create a violation
    # Or use simulated annealing / hill climbing
    
    # For each removed triple, check if adding it back keeps all configs satisfied
    # i.e., no config containing it would have all 8 selected
    
    readded = set()
    # Shuffle removed triples for randomization
    removed_list = list(removed)
    random.shuffle(removed_list)
    
    for t in removed_list:
        if time.time() > deadline - 5.0:
            break
        # Check if adding t back would violate any config
        can_add = True
        for ci in triple_to_configs[t]:
            if config_counts[ci] == 7:  # Adding would make it 8
                can_add = False
                break
        if can_add:
            selected[t] = True
            readded.add(t)
            # Update counts
            for ci in triple_to_configs[t]:
                config_counts[ci] += 1
    
    # Now do local search: try swapping one selected triple with one unselected
    # Or try removing one and adding two, etc.
    
    # Simple hill climbing: try to add an unselected triple that doesn't cause violation
    unselected = [i for i in range(num_triples) if not selected[i]]
    random.shuffle(unselected)
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for t in unselected:
            if time.time() > deadline - 5.0:
                break
            if selected[t]:
                continue
            can_add = True
            for ci in triple_to_configs[t]:
                if config_counts[ci] == 7:
                    can_add = False
                    break
            if can_add:
                selected[t] = True
                for ci in triple_to_configs[t]:
                    config_counts[ci] += 1
                improved = True
    
    # Build result
    result = [list(all_triples[i]) for i in range(num_triples) if selected[i]]
    
    return result

# EVOLVE-BLOCK-END
