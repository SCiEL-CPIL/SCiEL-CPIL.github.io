"""Triple systems on 16 points without K(2,2,2)."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    n = 16
    all_triples = list(combinations(range(n), 3))
    triple_index = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    deadline = time.time() + 115  # safety margin
    
    # Precompute for each triple, which K(2,2,2) configurations it participates in
    # A K(2,2,2) is defined by 3 disjoint pairs: {a,b}, {c,d}, {e,f}
    # The 8 transversal triples are: {a,c,e}, {a,c,f}, {a,d,e}, {a,d,f},
    #                                 {b,c,e}, {b,c,f}, {b,d,e}, {b,d,f}
    
    # For each 6-element subset, compute the 8 transversals
    # A triple participates in a K(2,2,2) if it's one of the 8 transversals for some partition of the 6 points into 3 pairs
    
    # Better approach: for each triple, find all K(2,2,2)s that contain it
    # A K(2,2,2) containing triple (a,b,c) means:
    # - Pick 3 more points d,e,f from the remaining 13
    # - Partition {a,b,c,d,e,f} into 3 pairs such that (a,b,c) is a transversal
    # This is complex, so let's use a different approach
    
    # For each set of 6 points, there are 15 ways to partition into 3 pairs
    # For each partition, the 8 transversals form a K(2,2,2)
    # We need to check that no K(2,2,2) has all 8 transversals present
    
    # Approach: maintain a count for each K(2,2,2) of how many of its 8 transversals are present
    # A K(2,2,2) is "complete" when its count reaches 8
    
    # Precompute all K(2,2,2) configurations
    # Each is identified by a frozenset of 8 triples
    k222_configs = []
    k222_to_triples = {}  # frozenset of 6 points -> list of 8-triple sets (one per partition)
    
    # Actually, let's be more careful. For 6 points, there are 15 perfect matchings.
    # For each matching, the 8 transversals form one K(2,2,2).
    # But different matchings can give the same set of 8 triples? No, they don't.
    # Actually, for a fixed set of 6 points, different matchings give different sets of 8 transversals.
    
    # Let me reconsider. The problem says "no three disjoint pairs have all eight transversal triples present."
    # So for any three disjoint pairs, the 8 transversals should NOT all be present.
    
    # For 6 points, the number of ways to choose 3 disjoint pairs (a perfect matching) is 15.
    # For each matching, we get 8 transversals.
    # We need to ensure that for no matching do all 8 transversals appear in our triple set.
    
    # Let's precompute all matchings for 6 points
    def get_matchings(points):
        """Get all perfect matchings of 6 points."""
        points = sorted(points)
        matchings = []
        def backtrack(remaining, current):
            if not remaining:
                matchings.append(current[:])
                return
            p0 = remaining[0]
            for i in range(1, len(remaining)):
                p1 = remaining[i]
                new_remaining = list(remaining[1:i]) + list(remaining[i+1:])
                current.append((p0, p1))
                backtrack(new_remaining, current)
                current.pop()
        backtrack(points, [])
        return matchings
    
    # Precompute for each 6-subset, the list of 8-transversal sets (one per matching)
    six_point_configs = []  # list of (frozenset_of_6, list_of_8triple_frozensets)
    for six in combinations(range(n), 6):
        six_frozen = frozenset(six)
        matchings = get_matchings(six)
        transversal_sets = []
        for matching in matchings:
            # matching is a list of 3 pairs
            # The 8 transversals: for each pair, pick one element
            triples = set()
            for p1 in matching[0]:
                for p2 in matching[1]:
                    for p3 in matching[2]:
                        triples.add(frozenset([p1, p2, p3]))
            transversal_sets.append(frozenset(triples))
        six_point_configs.append((six_frozen, transversal_sets))
    
    # Now, for each triple, precompute which (six_point_config, transversal_set) it belongs to
    triple_to_configs = {t: [] for t in all_triples}
    for six_frozen, transversal_sets in six_point_configs:
        for ts in transversal_sets:
            for triple in ts:
                if triple in triple_to_configs:
                    triple_to_configs[triple].append((six_frozen, frozenset(ts)))
    
    # For efficiency, let's use indices for the K(2,2,2) constraints
    # Each constraint is a set of 8 triples
    # We maintain a count for each constraint
    
    # Let's assign an ID to each unique 8-transversal set
    all_constraints = set()
    for six_frozen, transversal_sets in six_point_configs:
        for ts in transversal_sets:
            all_constraints.add(ts)
    
    constraint_ids = {c: i for i, c in enumerate(all_constraints)}
    num_constraints = len(all_constraints)
    
    # For each triple, list of constraint indices it belongs to
    triple_constraints = {t: [] for t in all_triples}
    for c_idx, c in enumerate(all_constraints):
        for triple in c:
            triple_constraints[triple].append(c_idx)
    
    # Greedy construction with multiple random orderings
    best_set = set()
    best_count = 0
    
    for iteration in range(200):
        if time.time() > deadline - 10:
            break
        
        random.shuffle(all_triples)
        current_set = set()
        constraint_counts = [0] * num_constraints
        
        for triple in all_triples:
            # Check if adding this triple would complete any constraint
            can_add = True
            for c_idx in triple_constraints[triple]:
                if constraint_counts[c_idx] == 7:
                    can_add = False
                    break
            if can_add:
                current_set.add(triple)
                for c_idx in triple_constraints[triple]:
                    constraint_counts[c_idx] += 1
        
        if len(current_set) > best_count:
            best_count = len(current_set)
            best_set = set(current_set)
    
    # Local search: try to add rejected triples
    # Rebuild constraint counts for best_set
    constraint_counts = [0] * num_constraints
    for triple in best_set:
        for c_idx in triple_constraints[triple]:
            constraint_counts[c_idx] += 1
    
    # Try to add any triple that was rejected
    added_any = True
    while added_any and time.time() < deadline - 5:
        added_any = False
        for triple in all_triples:
            if triple in best_set:
                continue
            can_add = True
            for c_idx in triple_constraints[triple]:
                if constraint_counts[c_idx] == 7:
                    can_add = False
                    break
            if can_add:
                best_set.add(triple)
                for c_idx in triple_constraints[triple]:
                    constraint_counts[c_idx] += 1
                added_any = True
    
    # Swap: remove one triple, try to add others
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        triples_in_set = list(best_set)
        random.shuffle(triples_in_set)
        
        for remove_triple in triples_in_set:
            if time.time() > deadline - 5:
                break
            if remove_triple not in best_set:
                continue
            best_set.remove(remove_triple)
            for c_idx in triple_constraints[remove_triple]:
                constraint_counts[c_idx] -= 1
            
            additions = 0
            for triple in all_triples:
                if triple in best_set:
                    continue
                can_add = True
                for c_idx in triple_constraints[triple]:
                    if constraint_counts[c_idx] == 7:
                        can_add = False
                        break
                if can_add:
                    best_set.add(triple)
                    for c_idx in triple_constraints[triple]:
                        constraint_counts[c_idx] += 1
                    additions += 1
            
            if additions >= 1:
                improved = True
            else:
                best_set.add(remove_triple)
                for c_idx in triple_constraints[remove_triple]:
                    constraint_counts[c_idx] += 1
    
    result = [list(t) for t in best_set]
    return result

# EVOLVE-BLOCK-END
