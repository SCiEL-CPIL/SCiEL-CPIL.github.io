"""Construct a large triple system on 16 points avoiding K(2,2,2)."""
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    
    n = 16
    points = list(range(n))
    
    # Generate all triples
    all_triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                all_triples.append((i, j, k))
    
    # Index triples
    triple_set = set(all_triples)
    triple_idx = {t: i for i, t in enumerate(all_triples)}
    
    # For each triple, precompute which other triples would form a K(2,2,2) with it
    # A K(2,2,2) has 6 points in 3 pairs. The 8 transversal triples each pick one from each pair.
    # Given triple t = (a,b,c), it participates in K(2,2,2) configs where:
    # - The 6 points are {a,b,c} plus 3 more points d,e,f
    # - The 3 pairs partition these 6 points, with a,b,c each in different pairs
    # - All 8 transversal triples must be present
    
    # For a given triple (a,b,c), to form K(2,2,2):
    # We need 3 more points d,e,f and a pairing of {a,b,c,d,e,f} into 3 pairs
    # where a,b,c are in different pairs.
    # The 8 transversal triples are all combinations picking one from each pair.
    # (a,b,c) is one of them. The other 7 must also be in our set.
    
    # Precompute conflicts: for each triple, which set of 7 other triples would complete a K(2,2,2)
    # Actually, for efficiency, let's precompute for each triple t, the list of "conflict groups"
    # where each group is a set of 7 triples such that if ALL 7 are present along with t, we have K(2,2,2)
    
    # Better approach: for each triple t, compute the set of triples that CANNOT be added 
    # because they would complete a K(2,2,2) with already-selected triples.
    
    # Let's use a different strategy: 
    # Start with star (all triples containing point 0).
    # Then greedily add non-star triples in an order that maximizes additions.
    
    # Star triples: all triples containing 0
    star_triples = set(t for t in all_triples if 0 in t)
    non_star_triples = [t for t in all_triples if 0 not in t]
    
    # For checking if adding triple t creates a K(2,2,2):
    # We need to check if there exist 5 other points and a pairing such that
    # all 8 transversal triples are in selected_set ∪ {t}
    
    # Precompute for each triple t, the list of potential K(2,2,2) completions
    # Each completion is defined by: 3 other points and a pairing
    # Store as: for triple t, a list of frozensets of 7 triples (the other 7 in the K(2,2,2))
    
    # This is expensive to precompute fully. Instead, let's do a fast check function.
    
    def has_k222_completion(triple, selected_set):
        """Check if adding 'triple' to selected_set creates any K(2,2,2)."""
        a, b, c = triple
        # The 6 points of a potential K(2,2,2) include a,b,c and 3 others.
        # The 3 others must be chosen from points not in {a,b,c}.
        other_points = [p for p in range(n) if p not in (a, b, c)]
        
        # For each choice of 3 other points d,e,f from other_points:
        for d, e, f in combinations(other_points, 3):
            six_points = {a, b, c, d, e, f}
            # We need to partition these 6 into 3 pairs such that a,b,c are in different pairs.
            # The pairs must be: {a,?}, {b,?}, {c,?} where ? are from {d,e,f}
            # So the pairing is determined by which of d,e,f pairs with a, b, c.
            # There are 3! = 6 such pairings.
            for perm in [(d,e,f), (d,f,e), (e,d,f), (e,f,d), (f,d,e), (f,e,d)]:
                pair_a = (a, perm[0])
                pair_b = (b, perm[1])
                pair_c = (c, perm[2])
                
                # The 8 transversal triples: pick one from each pair
                # Pair a: {a, perm[0]}, Pair b: {b, perm[1]}, Pair c: {c, perm[2]}
                transversal = []
                for xa in (a, perm[0]):
                    for xb in (b, perm[1]):
                        for xc in (c, perm[2]):
                            t = tuple(sorted((xa, xb, xc)))
                            transversal.append(t)
                
                # Check if all 8 are in selected_set ∪ {triple}
                all_present = True
                for t in transversal:
                    if t == triple:
                        continue
                    if t not in selected_set:
                        all_present = False
                        break
                if all_present:
                    return True
        return False
    
    # Greedy: start with star, add non-star triples one by one
    selected = set(star_triples)
    result_list = list(star_triples)
    
    # Sort non-star triples by some heuristic to get better results
    # Try multiple random orders and keep the best
    best_count = len(result_list)
    best_result = list(result_list)
    
    remaining = list(non_star_triples)
    
    start_time = time.time()
    
    # Try several random orders
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        random.shuffle(remaining)
        
        selected = set(star_triples)
        result_list = list(star_triples)
        
        for t in remaining:
            if time.time() > deadline:
                break
            if not has_k222_completion(t, selected):
                selected.add(t)
                result_list.append(t)
        
        if len(result_list) > best_count:
            best_count = len(result_list)
            best_result = list(result_list)
        
        # If we've used most of the time, stop
        if time.time() > deadline - 20:
            break
    
    return [[a, b, c] for a, b, c in best_result]
