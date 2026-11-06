"""Strategy: Maximum Independent Set via Branch and Bound on a 3-uniform hypergraph."""
import time
import itertools
import random

def construct():
    """
    Returns a nonempty subset of [1, 180] with no three distinct elements a, b, c 
    satisfying 1/a = 1/b + 1/c. Maximizes cardinality.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    N = 180
    numbers = list(range(1, N + 1))
    
    # 1. Identify all forbidden triples (a, b, c) with distinct elements
    # 1/a = 1/b + 1/c  =>  bc = a(b+c)  =>  a = bc / (b+c)
    # We iterate over pairs (b, c) with b < c to generate potential a.
    # If a is an integer, distinct from b and c, and in [1, N], it forms a triple.
    # The triple is {a, b, c}.
    
    triples = set()
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            numerator = b * c
            denominator = b + c
            if numerator % denominator == 0:
                a = numerator // denominator
                if 1 <= a <= N and a != b and a != c:
                    triple = tuple(sorted([a, b, c]))
                    triples.add(triple)
    
    # Convert to list for faster access if needed, but set is fine for lookup
    # For B&B, we need to know conflicts efficiently.
    # Let's build a conflict graph or hypergraph adjacency.
    # Since it's a 3-uniform hypergraph, we can check if adding x creates a forbidden triple
    # with any two elements already in the set.
    
    # Precompute for each element x, the list of pairs (y, z) such that {x, y, z} is a forbidden triple.
    # Actually, for B&B, we maintain a set `current`. When we consider adding `x`, we check if there exist y, z in `current` such that {x, y, z} is in `triples`.
    # To speed this up, we can precompute for each x, the set of pairs (y, z) that conflict with it.
    # Or, simpler: for each x, store the set of other elements that form a triple with x and some third element.
    # Let's use a different approach for B&B:
    # State: (candidate_set, forbidden_set, remaining_candidates)
    # This is complex. Let's try a simpler greedy with local search or a randomized heuristic first, then refine.
    
    # Given the time constraint, a robust approach is:
    # 1. Start with all numbers.
    # 2. While there is a forbidden triple, remove one element from it (e.g., the one with the highest degree in the conflict hypergraph, or random).
    # 3. Repeat until no forbidden triples remain.
    # 4. Try to add back elements that were removed, if possible.
    
    # Let's implement a randomized greedy removal + local search.
    
    def count_conflicts(s):
        """Count number of forbidden triples in set s."""
        count = 0
        # Since N=180, number of triples is small. Let's just check all triples.
        for t in triples:
            if t[0] in s and t[1] in s and t[2] in s:
                count += 1
        return count
    
    # Initial set: all numbers
    current_set = set(numbers)
    
    # Precompute degree of each element in the hypergraph of triples
    degree = {x: 0 for x in numbers}
    for t in triples:
        for x in t:
            degree[x] += 1
    
    # Greedy removal
    changed = True
    while changed:
        changed = False
        # Find a forbidden triple
        # To speed up, we can shuffle triples or pick the one involving highest degree elements
        # For now, just find any
        for t in triples:
            if t[0] in current_set and t[1] in current_set and t[2] in current_set:
                # Remove one element. Let's remove the one with the highest degree among the three.
                candidates = [t[0], t[1], t[2]]
                # Sort by degree descending
                candidates.sort(key=lambda x: degree[x], reverse=True)
                to_remove = candidates[0]
                current_set.remove(to_remove)
                changed = True
                break
    
    # Now try to add back elements
    # We maintain a list of removed elements
    removed = set(numbers) - current_set
    
    # Local search: try to add back an element if it doesn't create a conflict
    # We can do multiple passes or randomized restarts
    
    best_set = current_set.copy()
    
    # Randomized local search
    for _ in range(50):  # 50 restarts
        if time.time() > deadline:
            break
            
        # Start from best_set or a perturbed version
        current = best_set.copy()
        
        # Try to add back elements in random order
        removed_list = list(removed)
        random.shuffle(removed_list)
        
        for x in removed_list:
            if time.time() > deadline:
                break
            # Check if adding x creates a conflict
            # Conflict exists if there are y, z in current such that {x, y, z} is a triple
            conflict = False
            # Precompute: for each x, what pairs (y,z) conflict?
            # Let's just check all triples containing x
            for t in triples:
                if x in t:
                    others = [v for v in t if v != x]
                    if others[0] in current and others[1] in current:
                        conflict = True
                        break
            if not conflict:
                current.add(x)
        
        if len(current) > len(best_set):
            best_set = current
            removed = set(numbers) - best_set
            
    # Final verification
    for t in triples:
        if t[0] in best_set and t[1] in best_set and t[2] in best_set:
            # Should not happen, but just in case, remove one
            best_set.remove(t[0])
            break
            
    return list(best_set)
