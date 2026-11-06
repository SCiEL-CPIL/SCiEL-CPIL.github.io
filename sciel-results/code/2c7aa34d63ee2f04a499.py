import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # 1. Precompute all triples
    all_triples = list(combinations(range(16), 3))
    n_triples = len(all_triples)  # 560
    
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # 2. Precompute conflict graph
    # conflict[i][j] is True if triple i and triple j cannot both be in the system
    # because they would form part of a K(2,2,2) with 6 other triples.
    # Actually, the constraint is: no 6 points form a K(2,2,2).
    # Two triples conflict if there exists a set of 6 points containing their union
    # (if they are disjoint) or a superset? 
    # No, the constraint is global. 
    # A faster way: 
    # Two triples T1 and T2 are "compatible" if adding T2 to a system containing T1 
    # does not immediately force a violation? No, that's not right.
    #
    # Let's stick to the definition: A system is valid if no 6 distinct points 
    # have all 8 transversal triples present.
    #
    # We can precompute for each pair of triples (T1, T2) if they are "dangerous".
    # Actually, it's easier to check validity of a set S by checking all 6-point subsets?
    # There are C(16,6) = 8008 subsets. For each subset, check if all 8 transversals are in S.
    # This is a good way to VALIDATE, but hard to use for incremental search.
    #
    # Alternative: Precompute for each triple T, the set of other triples that 
    # "conflict" with it in the sense that if both are present, they must be 
    # checked against the rest. This is still complex.
    #
    # Let's use a different insight:
    # The number of K(2,2,2) substructures is small relative to the search space.
    # We can precompute all "forbidden sets" of 8 triples that form a K(2,2,2).
    # There are C(16,6) = 8008 choices of 6 points. For each, there is exactly 1 
    # set of 8 transversal triples.
    # So there are 8008 forbidden sets.
    #
    # A system is valid iff it does NOT contain any of these 8008 sets.
    #
    # This allows for a very fast check:
    # Maintain a count for each forbidden set: how many of its 8 triples are in the current system.
    # If any count reaches 8, it's invalid.
    
    # Precompute forbidden sets
    # forbidden_sets[k] = list of 8 triple indices
    forbidden_sets = []
    for points in combinations(range(16), 6):
        # Generate the 8 transversals for these 6 points
        # Points are partitioned into 3 pairs. 
        # Wait, the definition of K(2,2,2) implies a specific partition?
        # "no three disjoint pairs have all eight transversal triples present"
        # This implies that for ANY partition of 6 points into 3 pairs, the 8 transversals must not all be present.
        # So for a set of 6 points, there are 15 ways to partition them into 3 pairs.
        # For each partition, the 8 transversals form a forbidden set.
        
        # Let's generate all partitions of 6 points into 3 pairs.
        p = list(points)
        # p[0] pairs with p[1..5]
        # p[1] pairs with p[2..5] (remaining 4) -> 2 pairs
        # This is a standard generation.
        
        # Simpler: iterate over all permutations of the 6 points? No.
        # Let's just generate all 15 pairings.
        # A pairing is a set of 3 disjoint pairs.
        
        # We can represent a pairing by the set of 3 pairs.
        # Let's generate them recursively or via itertools.
        
        # Since 6 is small, we can hardcode or use a small generator.
        # Pairing 1: (0,1), (2,3), (4,5)
        # Pairing 2: (0,1), (2,4), (3,5)
        # ...
        
        # Let's write a helper to generate pairings of 6 elements.
        def gen_pairings(pts):
            if len(pts) == 0:
                yield []
            else:
                first = pts[0]
                for i in range(1, len(pts)):
                    second = pts[i]
                    rest = pts[1:i] + pts[i+1:]
                    for sub in gen_pairings(rest):
                        yield [(first, second)] + sub
        
        for pairing in gen_pairings(p):
            # pairing is a list of 3 tuples
            # The 8 transversals are formed by picking one from each pair
            # pair1, pair2, pair3
            p1, p2, p3 = pairing
            transversals = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        transversals.append(t)
            
            # Check if these 8 triples are distinct (they should be if points are distinct)
            if len(set(transversals)) == 8:
                # Convert to indices
                idx_set = []
                valid = True
                for t in transversals:
                    if t in triple_to_idx:
                        idx_set.append(triple_to_idx[t])
                    else:
                        valid = False
                        break
                if valid:
                    forbidden_sets.append(idx_set)
    
    n_forbidden = len(forbidden_sets)
    
    # Precompute for each triple, which forbidden sets it belongs to
    # triple_to_forbidden[i] = list of indices of forbidden sets containing triple i
    triple_to_forbidden = [[] for _ in range(n_triples)]
    for fs_idx, fs in enumerate(forbidden_sets):
        for t_idx in fs:
            triple_to_forbidden[t_idx].append(fs_idx)
    
    # 3. Local Search (Simulated Annealing)
    # State: a set of triples (boolean array of length n_triples)
    # Score: number of triples (maximize)
    # Constraint: no forbidden set is fully contained.
    
    # Initialize with a random valid set? Or start empty and add?
    # Starting empty is safe.
    
    best_system = []
    best_score = 0
    
    # Run multiple restarts
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Initialize state
        # Start with a random subset, or empty?
        # Let's start with a greedy valid set to get a good starting point.
        state = np.zeros(n_triples, dtype=bool)
        # Greedy add
        random.shuffle(all_triples)
        for t in all_triples:
            idx = triple_to_idx[t]
            # Check if adding idx causes any forbidden set to reach count 8
            # We need to maintain counts of forbidden sets.
            # This greedy is O(n * n_forbidden_per_triple)
            # Let's just use a simple check for now, or start empty and use SA to add.
            pass
        
        # Better: Start with an empty set and use SA to add triples.
        # But SA usually optimizes a fixed size or swaps.
        # Let's use a "growth" heuristic first, then SA to refine.
        
        # Growth:
        forbidden_counts = np.zeros(n_forbidden, dtype=np.int32)
        current_set = []
        
        # Order triples by how "dangerous" they are (fewer forbidden sets = safer?)
        # Or just random.
        order = list(range(n_triples))
        random.shuffle(order)
        
        for idx in order:
            # Check if adding idx violates any forbidden set
            # A violation occurs if any forbidden set containing idx already has 7 other members in current_set.
            # i.e., forbidden_counts[fs] == 7 for some fs in triple_to_forbidden[idx].
            
            is_safe = True
            for fs_idx in triple_to_forbidden[idx]:
                if forbidden_counts[fs_idx] == 7:
                    is_safe = False
                    break
            
            if is_safe:
                state[idx] = True
                current_set.append(idx)
                # Update counts
                for fs_idx in triple_to_forbidden[idx]:
                    forbidden_counts[fs_idx] += 1
        
        current_size = len(current_set)
        
        # Local Search: Try to add more triples by removing others?
        # Or just try to add any remaining triple that fits?
        # The greedy above might get stuck.
        
        # Simulated Annealing for swaps:
        # Try to swap a triple in with a triple out to increase size?
        # No, we want to maximize size. So we only want to ADD triples.
        # If we can't add, we might need to remove one to add two?
        
        # Let's try a simple hill climbing: 
        # 1. Try to add any triple that fits.
        # 2. If none fit, try to remove a triple that is part of many "blocked" sets, 
        #    then try to add more.
        
        # Actually, a better heuristic:
        # Randomly remove a triple, then greedily add as many as possible.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add any triple
            remaining = [i for i in range(n_triples) if not state[i]]
            random.shuffle(remaining)
            
            added_any = False
            for idx in remaining:
                is_safe = True
                for fs_idx in triple_to_forbidden[idx]:
                    if forbidden_counts[fs_idx] == 7:
                        is_safe = False
                        break
                
                if is_safe:
                    state[idx] = True
                    current_set.append(idx)
                    for fs_idx in triple_to_forbidden[idx]:
                        forbidden_counts[fs_idx] += 1
                    added_any = True
                    improved = True
                    # Don't break, keep adding if possible
            
            if not added_any:
                # Try a swap: remove one, add two?
                # Or just remove one and re-greedy?
                if current_size > 0:
                    # Pick a random triple to remove
                    remove_idx = random.choice(current_set)
                    # Remove it
                    state[remove_idx] = False
                    current_set.remove(remove_idx)
                    for fs_idx in triple_to_forbidden[remove_idx]:
                        forbidden_counts[fs_idx] -= 1
                    
                    # Now try to add many
                    remaining = [i for i in range(n_triples) if not state[i]]
                    random.shuffle(remaining)
                    
                    added_count = 0
                    for idx in remaining:
                        is_safe = True
                        for fs_idx in triple_to_forbidden[idx]:
                            if forbidden_counts[fs_idx] == 7:
                                is_safe = False
                                break
                        
                        if is_safe:
                            state[idx] = True
                            current_set.append(idx)
                            for fs_idx in triple_to_forbidden[idx]:
                                forbidden_counts[fs_idx] += 1
                            added_count += 1
                    
                    if added_count > 1:
                        improved = True
                    # If added_count <= 1, we didn't improve, but we changed the state.
                    # We should probably revert if we didn't improve size?
                    # For now, let's keep it if size didn't decrease.
                    # Actually, if added_count == 1, size is same. If 0, size decreased.
                    # We want to maximize size.
                    
                    if added_count <= 1:
                        # Revert? Or just accept and try again?
                        # Let's accept and continue, hoping to find a better config later.
                        pass
        
        if len(current_set) > best_score:
            best_score = len(current_set)
            best_system = [all_triples[i] for i in current_set]
            
    # Final validation (just in case)
    # Convert best_system to set of tuples for fast check
    final_set = set(best_system)
    
    # Check all forbidden sets
    for fs in forbidden_sets:
        count = 0
        for idx in fs:
            if all_triples[idx] in final_set:
                count += 1
        if count == 8:
            # Invalid! Remove one triple from this set
            # This shouldn't happen if our logic is correct, but safety first.
            # Find a triple in this set to remove
            for idx in fs:
                if all_triples[idx] in final_set:
                    final_set.remove(all_triples[idx])
                    break
    
    return [list(t) for t in final_set]
