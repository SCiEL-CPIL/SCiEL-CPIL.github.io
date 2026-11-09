"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, product

def _is_2_colorable(edges, n):
    """Check if a 5-uniform hypergraph on n vertices is 2-colorable using
    backtracking with unit propagation. Returns True if colorable, False if not."""
    # Represent each edge as a tuple for hashing
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    # For unit propagation: an edge is "satisfied" if it has at least one vertex of each color.
    # An edge is "violated" (monochromatic) if all its vertices are the same color.
    # Unit propagation: if an edge has 4 vertices of one color, the 5th must be the other.
    # If an edge has all 5 of one color, it's violated -> unsatisfiable.
    
    # We'll use a simple backtracking with unit propagation.
    # State: assignment dict vertex -> color (0 or 1), or None if unassigned.
    
    # Precompute for each edge the list of vertices
    edge_vertices = [list(e) for e in edges]
    
    # For efficiency, track for each edge how many vertices are assigned to color 0 and color 1
    # and which are unassigned.
    
    # Actually, let's use a cleaner approach:
    # For each edge, we need at least one vertex of each color.
    # If an edge has all assigned vertices the same color and there are unassigned vertices,
    # the unassigned ones must include the other color -> unit propagation.
    # If an edge has all assigned vertices the same color and no unassigned vertices -> violated.
    
    # Let's implement with recursion.
    
    # To make this efficient, we'll use a forward-checking approach.
    
    # For each edge, maintain:
    # - count0: number of vertices assigned color 0
    # - count1: number of vertices assigned color 1
    # - unassigned: list of unassigned vertices in this edge
    
    # If count0 == 0 and count1 > 0: all unassigned must be color 0? No, we need at least one of each.
    # Actually: if count0 == 0, we need at least one unassigned to be color 0.
    # If count1 == 0, we need at least one unassigned to be color 1.
    # If count0 + count1 == 5 (all assigned):
    #   if count0 == 0 or count1 == 0: violated
    # If count0 == 0 and there's exactly 1 unassigned: that vertex must be color 0.
    # If count1 == 0 and there's exactly 1 unassigned: that vertex must be color 1.
    # If count0 == 0 and count1 == 0 and 5 unassigned: no propagation yet.
    # If count0 == 4 and 1 unassigned: that vertex must be color 1.
    # If count1 == 4 and 1 unassigned: that vertex must be color 0.
    
    # More generally:
    # If count0 == 0 and num_unassigned == 1: the unassigned must be 0.
    # If count1 == 0 and num_unassigned == 1: the unassigned must be 1.
    # If count0 == 4 and num_unassigned == 1: the unassigned must be 1.
    # If count1 == 4 and num_unassigned == 1: the unassigned must be 0.
    # If count0 == 5 or count1 == 5: violated.
    
    # Let me implement a cleaner backtracking.
    
    def propagate(assignment, edge_state):
        """Apply unit propagation. Returns False if contradiction, else True.
        Modifies assignment and edge_state in place."""
        changed = True
        while changed:
            changed = False
            for i, (c0, c1, unassigned) in enumerate(edge_state):
                # Check for violation
                if c0 == 5 or c1 == 5:
                    return False
                # Check for unit propagation
                if c0 == 0 and len(unassigned) == 1:
                    v = unassigned[0]
                    if assignment[v] is not None:
                        if assignment[v] == 1:
                            return False
                        else:
                            # Already assigned 0, should have been counted
                            continue
                    assignment[v] = 0
                    # Update all edges containing v
                    for j, (c0j, c1j, unassigned_j) in enumerate(edge_state):
                        if v in unassigned_j:
                            new_unassigned = [x for x in unassigned_j if x != v]
                            edge_state[j] = (c0j + 1, c1j, new_unassigned)
                    changed = True
                    break
                elif c1 == 0 and len(unassigned) == 1:
                    v = unassigned[0]
                    if assignment[v] is not None:
                        if assignment[v] == 0:
                            return False
                        else:
                            continue
                    assignment[v] = 1
                    for j, (c0j, c1j, unassigned_j) in enumerate(edge_state):
                        if v in unassigned_j:
                            new_unassigned = [x for x in unassigned_j if x != v]
                            edge_state[j] = (c0j, c1j + 1, new_unassigned)
                    changed = True
                    break
                elif c0 == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    if assignment[v] is not None:
                        if assignment[v] == 0:
                            return False
                        else:
                            continue
                    assignment[v] = 1
                    for j, (c0j, c1j, unassigned_j) in enumerate(edge_state):
                        if v in unassigned_j:
                            new_unassigned = [x for x in unassigned_j if x != v]
                            edge_state[j] = (c0j, c1j + 1, new_unassigned)
                    changed = True
                    break
                elif c1 == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    if assignment[v] is not None:
                        if assignment[v] == 1:
                            return False
                        else:
                            continue
                    assignment[v] = 0
                    for j, (c0j, c1j, unassigned_j) in enumerate(edge_state):
                        if v in unassigned_j:
                            new_unassigned = [x for x in unassigned_j if x != v]
                            edge_state[j] = (c0j + 1, c1j, new_unassigned)
                    changed = True
                    break
        return True
    
    def backtrack(assignment, edge_state, unassigned_vertices):
        """Try to complete the assignment. Returns True if a valid 2-coloring exists."""
        if not propagate(assignment, edge_state):
            return False
        
        # Check if all assigned
        if not unassigned_vertices:
            return True
        
        # Choose a vertex to assign (heuristic: pick one involved in most constrained edges)
        # For simplicity, pick the first unassigned vertex
        v = unassigned_vertices[0]
        remaining = unassigned_vertices[1:]
        
        for color in [0, 1]:
            new_assignment = dict(assignment)
            new_assignment[v] = color
            
            # Build new edge_state
            new_edge_state = []
            for (c0, c1, unassigned) in edge_state:
                if v in unassigned:
                    new_unassigned = [x for x in unassigned if x != v]
                    if color == 0:
                        new_edge_state.append((c0 + 1, c1, new_unassigned))
                    else:
                        new_edge_state.append((c0, c1 + 1, new_unassigned))
                else:
                    new_edge_state.append((c0, c1, unassigned))
            
            new_unassigned = [x for x in unassigned_vertices if x != v]
            
            if backtrack(new_assignment, new_edge_state, new_unassigned):
                return True
        
        return False
    
    # Initialize
    assignment = {v: None for v in range(n)}
    edge_state = []
    for e in edge_vertices:
        edge_state.append((0, 0, list(e)))
    
    return backtrack(assignment, edge_state, list(range(n)))

def _try_to_find_coloring(edges, n):
    """Alias for clarity."""
    return _is_2_colorable(edges, n)

def _greedy_construct(n, target_edges, seed=42):
    """Try to construct a non-2-colorable 5-uniform hypergraph with at most target_edges edges
    on n vertices using a greedy randomized approach."""
    random.seed(seed)
    
    all_edges = list(combinations(range(n), 5))
    random.shuffle(all_edges)
    
    # Start with a random subset and try to build up
    # Strategy: start with random edges, check if non-2-colorable, if not add more
    # This is slow. Better: use a known construction or heuristic.
    
    # Let's try a different approach: start with a small set and iteratively add edges
    # that break colorings.
    
    # Actually, let's just try random subsets of increasing size.
    # For each size k, try a few random subsets of size k.
    
    for k in range(30, min(target_edges + 1, len(all_edges))):
        for attempt in range(50):
            random.seed(seed + k * 1000 + attempt)
            random.shuffle(all_edges)
            candidate = all_edges[:k]
            if not _is_2_colorable(candidate, n):
                return [list(e) for e in candidate]
    
    return None

def _known_constructions():
    """Return known non-2-colorable 5-uniform hypergraphs."""
    constructions = []
    
    # The complete 5-uniform hypergraph on 9 vertices has C(9,5)=126 edges
    # That's too many. We want fewer.
    
    # A known result: the Fano plane analog for 5-uniform?
    # Let's try to find small examples by search.
    
    return constructions

def _search_small(n, max_edges, time_limit=30):
    """Search for a non-2-colorable 5-uniform hypergraph on n vertices with at most max_edges edges."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    all_edges = list(combinations(range(n), 5))
    
    # Try random subsets
    best = None
    
    # Start with smaller sizes
    for k in range(30, max_edges + 1):
        if time.time() > deadline - 5:
            break
        
        # Try a few random subsets of size k
        num_tries = 20
        for t in range(num_tries):
            if time.time() > deadline - 2:
                break
            
            # Random subset
            random.seed(t * 1000 + k)
            random.shuffle(all_edges)
            candidate = all_edges[:k]
            
            if not _is_2_colorable(candidate, n):
                if best is None or k < len(best):
                    best = [list(e) for e in candidate]
                break  # Found one of this size, try smaller? No, we're going up.
    
    return best

def construct():
    """Construct a 5-uniform hypergraph with no proper 2-colouring, minimizing edges."""
    start_time = time.time()
    total_deadline = start_time + 140  # Leave 10s margin
    
    best_result = None
    
    # Try known small constructions first
    # The seed program uses C(9,5)=126 edges. We want fewer.
    
    # Try n=9 with fewer edges
    for n in [9, 10, 11, 12]:
        if time.time() > total_deadline - 20:
            break
        
        # Try to find a non-2-colorable hypergraph on n vertices
        # Start with a reasonable lower bound
        min_edges_to_try = 30
        max_edges_to_try = 120
        
        result = _search_small(n, max_edges_to_try, time_limit=30)
        if result is not None:
            if best_result is None or len(result) < len(best_result['edges']):
                best_result = {'n': n, 'edges': result}
    
    # If we found something, verify it
    if best_result is not None:
        if time.time() < total_deadline - 5:
            if _is_2_colorable(best_result['edges'], best_result['n']):
                # It's colorable, so invalid. Fall back to seed.
                best_result = None
    
    if best_result is not None:
        return best_result
    
    # Fallback: use the seed construction
    from itertools import combinations
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
# EVOLVE-BLOCK-END
