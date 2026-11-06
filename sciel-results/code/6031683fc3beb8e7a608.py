import time
import random
from itertools import combinations, permutations

# Deadline guard
_DEADLINE = 110.0
_start_time = None

def _time_left():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _DEADLINE - (time.time() - _start_time)

def _timeup():
    return _time_left() <= 0

# Q3 Edges in canonical form (0..7)
_Q3_EDGES = []
for i in range(8):
    for j in range(i+1, 8):
        if bin(i ^ j).count('1') == 1:
            _Q3_EDGES.append((i, j))

# Precompute unique Q3 patterns (sets of 12 local edges)
_Q3_PATTERNS = set()
for perm in permutations(range(8)):
    edges = set()
    for (i, j) in _Q3_EDGES:
        a, b = perm[i], perm[j]
        if a > b:
            a, b = b, a
        edges.add((a, b))
    _Q3_PATTERNS.add(frozenset(edges))

_Q3_PATTERNS = list(_Q3_PATTERNS)
# print(f"Patterns: {len(_Q3_PATTERNS)}") # Should be 840

def _contains_q3_in_subset(subset, edge_set):
    """
    Check if the subgraph induced by `subset` (tuple of 8 vertices) contains Q3.
    `edge_set` is a set of (u, v) with u < v.
    """
    # Map global vertices to local 0..7
    v_map = {v: i for i, v in enumerate(subset)}
    
    # Build set of local edges present
    local_edges = set()
    for i in range(8):
        for j in range(i+1, 8):
            u, v = subset[i], subset[j]
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                local_edges.add((i, j))
                
    # If fewer than 12 edges, cannot contain Q3
    if len(local_edges) < 12:
        return False
        
    # Check against patterns
    for pattern in _Q3_PATTERNS:
        if pattern.issubset(local_edges):
            return True
            
    return False

def _find_q3(edge_set):
    """
    Find a Q3 subgraph in the graph defined by edge_set.
    Returns (subset, q3_edges_in_subset) or (None, None).
    """
    vertices = range(16)
    # Randomize order to avoid bias
    subsets = list(combinations(vertices, 8))
    random.shuffle(subsets)
    
    for subset in subsets:
        if _timeup():
            return None, None
        if _contains_q3_in_subset(subset, edge_set):
            # Find which specific Q3 edges are present
            # We can just return the subset, and pick a random edge from it later
            # But we need to know which edges to remove.
            # Let's find one valid pattern.
            v_map = {v: i for i, v in enumerate(subset)}
            local_edges = set()
            for i in range(8):
                for j in range(i+1, 8):
                    u, v = subset[i], subset[j]
                    if u > v:
                        u, v = v, u
                    if (u, v) in edge_set:
                        local_edges.add((i, j))
                        
            for pattern in _Q3_PATTERNS:
                if pattern.issubset(local_edges):
                    # Map local edges back to global
                    global_q3_edges = []
                    for (i, j) in pattern:
                        u, v = subset[i], subset[j]
                        if u > v:
                            u, v = v, u
                        global_q3_edges.append((u, v))
                    return subset, global_q3_edges
                    
    return None, None

def _add_edge_creates_q3(edge_set, u, v):
    """
    Check if adding edge (u, v) creates a Q3.
    Only check 8-subsets containing u and v.
    """
    if u > v:
        u, v = v, u
    if (u, v) in edge_set:
        return False
        
    new_edge_set = edge_set | {(u, v)}
    
    other_vertices = [x for x in range(16) if x != u and x != v]
    
    # C(14, 6) = 3003 subsets
    for combo6 in combinations(other_vertices, 6):
        subset = tuple(sorted([u, v] + list(combo6)))
        if _contains_q3_in_subset(subset, new_edge_set):
            return True
            
    return False

def construct():
    global _start_time
    _start_time = time.time()
    random.seed(42)
    
    # Start with K16
    edge_set = set()
    for i in range(16):
        for j in range(i+1, 16):
            edge_set.add((i, j))
            
    removed_edges = []
    
    # Phase 1: Remove edges to destroy all Q3s
    # We use a loop that finds a Q3 and removes an edge.
    # To be efficient, we find a Q3, remove an edge, and continue.
    
    while not _timeup():
        subset, q3_edges = _find_q3(edge_set)
        if subset is None:
            break
            
        # Pick an edge to remove
        # Heuristic: Pick the edge in the Q3 that appears in the most other Q3s?
        # Too expensive to calculate. Just pick randomly.
        edge_to_remove = random.choice(q3_edges)
        edge_set.discard(edge_to_remove)
        removed_edges.append(edge_to_remove)
        
    # Phase 2: Try to add back removed edges
    # We try to add edges back in random order.
    # We repeat this process a few times to catch edges that became addable
    # after other edges were removed.
    
    best_edge_set = set(edge_set)
    best_count = len(edge_set)
    
    for _pass in range(10):
        if _timeup():
            break
            
        improved = False
        # Shuffle removed edges
        random.shuffle(removed_edges)
        
        for e in removed_edges:
            if _timeup():
                break
            u, v = e
            if (u, v) in edge_set:
                continue # already added back
            
            if not _add_edge_creates_q3(edge_set, u, v):
                edge_set.add((u, v))
                removed_edges.remove(e)
                improved = True
                
        if not improved:
            break
            
    if len(edge_set) > best_count:
        best_edge_set = set(edge_set)
        best_count = len(edge_set)
        
    # Final check and cleanup
    # Ensure no Q3 exists (safety)
    subset, _ = _find_q3(best_edge_set)
    if subset is not None:
        # This shouldn't happen if _add_edge_creates_q3 is correct,
        # but if it does, we need to remove an edge.
        # Re-run the removal loop for the best set
        while not _timeup():
            subset, q3_edges = _find_q3(best_edge_set)
            if subset is None:
                break
            edge_to_remove = random.choice(q3_edges)
            best_edge_set.discard(edge_to_remove)

    return [[u, v] for (u, v) in best_edge_set]
