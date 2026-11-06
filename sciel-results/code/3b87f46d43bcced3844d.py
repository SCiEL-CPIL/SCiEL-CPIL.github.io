"""Optimized solver for large integer sets avoiding equal-lcm triples."""

def construct():
    import time
    import math
    from itertools import combinations

    # Deadline guard: must return well before 90s wall-clock limit
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin

    N = 128
    elements = list(range(1, N + 1))

    # Precompute LCMs for all pairs
    # lcm[a][b] for a < b
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[(i, j)] = math.lcm(i, j)

    # Build bad triples: three distinct elements (a, b, c) where
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    # This means all three pairwise LCMs are equal.
    # Let's precompute which triples are "bad"
    
    # For efficiency, group pairs by their LCM value
    # For a triple (a,b,c) to be bad, all three pairs must share the same LCM
    # So: lcm(a,b) = lcm(a,c) = lcm(b,c) = L
    # This means a, b, c all divide L, and for any two, their lcm is exactly L
    
    # Better approach: for each LCM value L, find all pairs with that LCM,
    # then find triangles (triples) within those pairs
    
    # Group pairs by LCM
    from collections import defaultdict
    pairs_by_lcm = defaultdict(list)
    for (a, b), l in lcm_table.items():
        pairs_by_lcm[l].append((a, b))

    # For each LCM group, find all triples (a,b,c) where all three pairs are in the group
    # Use adjacency: for each LCM group, build a graph where edges are the pairs,
    # then find all triangles
    
    bad_triples = set()
    for L, pairs in pairs_by_lcm.items():
        if len(pairs) < 3:
            continue
        # Build adjacency for this group
        adj = defaultdict(set)
        for a, b in pairs:
            adj[a].add(b)
            adj[b].add(a)
        
        # Find all triangles
        for a, b in pairs:
            common = adj[a] & adj[b]
            for c in common:
                if c != a and c != b:
                    triple = tuple(sorted((a, b, c)))
                    bad_triples.add(triple)

    # Now we have all bad triples. We want the maximum subset with no bad triple.
    # This is a maximum independent set in a 3-uniform hypergraph.
    
    # Strategy: Start with all elements, then iteratively remove elements
    # that are part of bad triples, using a greedy removal strategy.
    # Then try to add back elements.
    
    # First, count how many bad triples each element participates in
    element_bad_count = defaultdict(int)
    for triple in bad_triples:
        for x in triple:
            element_bad_count[x] += 1

    # Greedy Remove: repeatedly remove the element that is in the most bad triples
    # among the currently active elements
    active = set(elements)
    
    # For efficient tracking, maintain for each active element the set of bad triples
    # it participates in that are still "active" (all three elements active)
    
    # Simpler approach: iterative removal
    # Remove elements one by one, prioritizing those in most remaining bad triples
    
    # Precompute: for each element, list of bad triples containing it
    elem_triples = defaultdict(list)
    for triple in bad_triples:
        for x in triple:
            elem_triples[x].append(triple)

    # Greedy removal with dynamic re-evaluation
    active = set(elements)
    removed = []
    
    # Count active bad triples per element
    def count_active_triples(elem, active_set):
        count = 0
        for t in elem_triples[elem]:
            if all(x in active_set for x in t):
                count += 1
        return count

    # This is O(n * |triples_per_elem|) per removal, which might be slow.
    # Let's use a different approach: just do multiple passes of greedy add/remove.
    
    # Approach 1: Greedy Add in various orders
    def greedy_add(order):
        selected = []
        selected_set = set()
        for x in order:
            # Check if adding x creates any bad triple with existing selected elements
            valid = True
            for a, b in combinations(selected, 2):
                triple = tuple(sorted((a, b, x)))
                if triple in bad_triples:
                    valid = False
                    break
            if valid:
                selected.append(x)
                selected_set.add(x)
        return selected

    # Try multiple orderings and keep the best
    best_set = []
    best_size = 0

    # Order 1: natural order
    result = greedy_add(elements)
    if len(result) > best_size:
        best_size = len(result)
        best_set = result

    # Order 2: reverse order
    result = greedy_add(list(reversed(elements)))
    if len(result) > best_size:
        best_size = len(result)
        best_set = result

    # Order 3: by number of bad triples (fewest first - less constrained)
    order3 = sorted(elements, key=lambda x: element_bad_count.get(x, 0))
    result = greedy_add(order3)
    if len(result) > best_size:
        best_size = len(result)
        best_set = result

    # Order 4: by number of bad triples (most first)
    order4 = sorted(elements, key=lambda x: -element_bad_count.get(x, 0))
    result = greedy_add(order4)
    if len(result) > best_size:
        best_size = len(result)
        best_set = result

    # Order 5: by element value (odd first, then even)
    odds = [x for x in elements if x % 2 == 1]
    evens = [x for x in elements if x % 2 == 0]
    result = greedy_add(odds + evens)
    if len(result) > best_size:
        best_size = len(result)
        best_set = result
    result = greedy_add(evens + odds)
    if len(result) > best_size:
        best_size = len(result)
        best_set = result

    # Now try Greedy Remove + Add Back approach
    # Start with all elements, remove those causing conflicts, then add back
    def greedy_remove_addback():
        active = set(elements)
        # Build conflict structure
        # For each element, track which other elements it conflicts with (in bad triples)
        # Actually, conflicts are hyperedges (triples), not pairwise.
        
        # Simpler: iteratively find a bad triple and remove one of its elements
        # Choose the element to remove based on some heuristic
        
        active = set(elements)
        # Precompute for fast lookup
        # For a given active set, find all active bad triples
        # This is expensive, so let's do it differently.
        
        # Use a queue-based approach
        # Start with all elements active
        # Find a bad triple, remove the "most constrained" element
        # Repeat until no bad triples remain
        
        # To find bad triples efficiently, maintain for each pair the LCM,
        # and check if there's a third element completing a bad triple.
        
        # Alternative: just do multiple random removal passes
        import random
        rng = random.Random(42)
        
        for trial in range(20):
            active = set(elements)
            # Shuffle the order of consideration
            shuffled = list(elements)
            rng.shuffle(shuffled)
            
            # Remove elements that are in bad triples
            # Process in shuffled order
            for x in shuffled:
                if x not in active:
                    continue
                # Check if x is in any active bad triple
                in_bad = False
                for t in elem_triples[x]:
                    if all(y in active for y in t):
                        in_bad = True
                        break
                if in_bad:
                    active.discard(x)
            
            # Now try to add back removed elements
            removed_elems = [x for x in elements if x not in active]
            rng.shuffle(removed_elems)
            for x in removed_elems:
                # Check if adding x to active creates a bad triple
                valid = True
                # Need to check all pairs in active with x
                active_list = list(active)
                for i in range(len(active_list)):
                    for j in range(i+1, len(active_list)):
                        a, b = active_list[i], active_list[j]
                        triple = tuple(sorted((a, b, x)))
                        if triple in bad_triples:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    active.add(x)
            
            if len(active) > best_size:
                best_size = len(active)
                best_set = list(active)
            
            if time.time() > deadline:
                break

    greedy_remove_addback()

    # Also try a local search: start from best_set, try to swap elements
    # or add elements that were excluded
    def local_search(initial_set, max_iters=100):
        current = set(initial_set)
        current_list = list(current)
        best_local = len(current)
        
        # Try adding each excluded element
        excluded = [x for x in elements if x not in current]
        rng = random.Random(123)
        rng.shuffle(excluded)
        
        for x in excluded:
            if time.time() > deadline:
                break
            # Check if x can be added
            valid = True
            for a, b in combinations(current_list, 2):
                triple = tuple(sorted((a, b, x)))
                if triple in bad_triples:
                    valid = False
                    break
            if valid:
                current.add(x)
                current_list = list(current)
        
        # Try removing one element and adding two others (swap)
        # This is complex; skip for now
        
        return current

    if best_size < N:
        improved = local_search(best_set)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = list(improved)

    # Final check: verify the set has no bad triples
    final_set = set(best_set)
    for triple in bad_triples:
        if all(x in final_set for x in triple):
            # This shouldn't happen, but if it does, remove one element
            # Find and remove
            for x in triple:
                if x in final_set:
                    final_set.discard(x)
                    break
            break

    return list(final_set) if final_set else [1]
