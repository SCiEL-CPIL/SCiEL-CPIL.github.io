"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _build_conflict_graph(n=180):
    """
    Build the conflict graph.
    Vertices: 1..n
    An edge (u, v) exists if there is some w in [1, n], w != u, w != v,
    such that {u, v, w} forms a forbidden triple 1/a = 1/b + 1/c.
    
    The forbidden condition for distinct a, b, c is:
    1/a = 1/b + 1/c  <=>  bc = a(b+c)  <=>  1/a = (b+c)/(bc)
    So for a given pair (b, c) with b != c, the third element a must satisfy:
    a = bc / (b+c)
    For a to be an integer, (b+c) must divide bc.
    
    Actually, let's re-derive carefully.
    We want to avoid three DISTINCT elements a, b, c such that 1/a = 1/b + 1/c.
    This is equivalent to: a = (b*c) / (b+c).
    So for any pair (b, c), if a = bc/(b+c) is an integer in [1, n] and a != b and a != c,
    then {a, b, c} is a forbidden triple.
    
    We need to build a graph where an edge between x and y means that x and y CANNOT both be in the solution
    because there exists a z such that {x, y, z} is a forbidden triple.
    
    Wait, the problem says "no THREE DISTINCT elements a,b,c satisfying...".
    So if we include x and y in our set, we must ensure there is no z in our set (or even just in [1,n]?)
    such that {x,y,z} is a forbidden triple.
    
    Actually, the constraint is: the subset S must not contain any three distinct elements a,b,c with 1/a=1/b+1/c.
    So if x, y in S, we must ensure that for any z in S, {x,y,z} is not a forbidden triple.
    This is a 3-uniform hypergraph independent set problem.
    
    However, we can convert this to a graph problem.
    If x and y are in S, and there exists a z in [1,n] such that {x,y,z} is a forbidden triple,
    does that mean x and y cannot both be in S?
    No, only if z is also in S.
    
    So it's not a simple graph independent set. It's a hypergraph independent set.
    But we can use a greedy approach or local search on the hypergraph.
    
    Let's first identify all forbidden triples.
    """
    n = 180
    forbidden_triples = []
    for b in range(1, n+1):
        for c in range(b+1, n+1):
            # a = bc / (b+c)
            numerator = b * c
            denominator = b + c
            if numerator % denominator == 0:
                a = numerator // denominator
                if 1 <= a <= n and a != b and a != c:
                    triple = tuple(sorted([a, b, c]))
                    forbidden_triples.append(triple)
    
    # Remove duplicates
    forbidden_triples = list(set(forbidden_triples))
    
    # Build adjacency: for each pair (x, y), does there exist a z such that {x,y,z} is forbidden?
    # If so, we create a "conflict" between x and y, but note that the conflict only matters if z is also present.
    # For local search, we can define the "cost" of adding an element x to a set S as the number of forbidden triples
    # that would be completed by adding x.
    
    return forbidden_triples

def _count_conflicts(element, current_set, forbidden_triples_by_element):
    """
    Count how many forbidden triples would be completed if `element` is added to `current_set`.
    """
    count = 0
    for triple in forbidden_triples_by_element[element]:
        # Check if the other two elements are in current_set
        other1, other2 = triple[0], triple[1]
        if other1 != element and other2 != element:
            if other1 in current_set and other2 in current_set:
                count += 1
        elif other1 == element:
            # other2 is the other element
            if other2 in current_set:
                # Need to check if there's a third element... wait, triple has 3 elements.
                # If element is in triple, the other two are the ones to check.
                pass
    # Let's re-implement more carefully.
    count = 0
    for triple in forbidden_triples_by_element[element]:
        # triple is a sorted tuple of 3 distinct elements including `element`
        others = [x for x in triple if x != element]
        if len(others) == 2:
            if others[0] in current_set and others[1] in current_set:
                count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    n = 180
    elements = list(range(1, n+1))
    
    # Precompute forbidden triples
    forbidden_triples = []
    for b in range(1, n+1):
        for c in range(b+1, n+1):
            numerator = b * c
            denominator = b + c
            if numerator % denominator == 0:
                a = numerator // denominator
                if 1 <= a <= n and a != b and a != c:
                    triple = tuple(sorted([a, b, c]))
                    forbidden_triples.append(triple)
    
    forbidden_triples = list(set(forbidden_triples))
    
    # Build mapping from element to triples containing it
    forbidden_triples_by_element = {i: [] for i in range(1, n+1)}
    for triple in forbidden_triples:
        for x in triple:
            forbidden_triples_by_element[x].append(triple)
    
    # Greedy initialization: start with all elements, then remove ones that cause conflicts
    # Or start empty and add elements with fewest conflicts.
    
    # Let's try: start with a large random set, then iteratively remove elements that are part of many conflicts.
    
    best_set = set()
    best_size = 0
    
    # Try multiple restarts
    num_restarts = 100
    random.seed(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize with a random subset
        # Heuristic: elements with fewer conflicts are "safer"
        # Let's try starting with all elements and removing problematic ones
        current_set = set(elements)
        
        # Iteratively remove elements that are in the most conflicts
        # A conflict is a forbidden triple where all three elements are in current_set
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Count conflicts for each element
            conflict_count = {x: 0 for x in current_set}
            for triple in forbidden_triples:
                if all(x in current_set for x in triple):
                    for x in triple:
                        conflict_count[x] += 1
            
            # Find element with max conflict count
            if not current_set:
                break
            
            max_conflict = max(conflict_count.values())
            if max_conflict == 0:
                break
            
            # Remove one of the elements with max conflict (random choice among them)
            candidates = [x for x in current_set if conflict_count[x] == max_conflict]
            to_remove = random.choice(candidates)
            current_set.remove(to_remove)
            improved = True
        
        # Local search: try to add back removed elements or swap
        # Simple local search: try adding each removed element if it doesn't create conflicts
        removed = set(elements) - current_set
        for x in list(removed):
            if time.time() > deadline:
                break
            # Check if adding x creates any conflict
            safe = True
            for triple in forbidden_triples_by_element[x]:
                others = [y for y in triple if y != x]
                if len(others) == 2 and others[0] in current_set and others[1] in current_set:
                    safe = False
                    break
            if safe:
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
    
    return list(best_set)

# EVOLVE-BLOCK-END
