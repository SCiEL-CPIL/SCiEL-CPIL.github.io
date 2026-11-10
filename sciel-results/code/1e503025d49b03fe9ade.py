"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _check_non_2colorable(n, edges):
    """
    Check if the hypergraph is NOT 2-colorable using backtracking with unit propagation.
    Returns True if non-2-colorable (no valid 2-coloring exists).
    Returns False if 2-colorable (a valid coloring exists).
    """
    edge_set = list(set(tuple(sorted(e)) for e in edges))
    m = len(edge_set)
    
    # Build incidence: vertex -> list of edge indices
    vert_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_set):
        for v in e:
            vert_to_edges[v].append(i)
    
    # We'll use a simple backtracking solver
    # assignment[v] = -1 (unassigned), 0, or 1
    assignment = [-1] * n
    
    # For each edge, track how many of each color are assigned
    # edge_counts[i] = [count0, count1, count_unknown]
    
    def is_dead_edge(e_idx):
        """Check if edge is already monochromatic (all 5 same color)."""
        c0 = c1 = 0
        for v in edge_set[e_idx]:
            if assignment[v] == 0:
                c0 += 1
            elif assignment[v] == 1:
                c1 += 1
        if c0 == 5 or c1 == 5:
            return True
        return False
    
    def propagate():
        """
        Unit propagation: if an edge has 4 vertices of one color and 1 unassigned,
        the unassigned must be the other color.
        Returns False if contradiction (edge becomes monochromatic), True otherwise.
        """
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edge_set):
                c0 = c1 = 0
                unknown = None
                for v in e:
                    if assignment[v] == 0:
                        c0 += 1
                    elif assignment[v] == 1:
                        c1 += 1
                    else:
                        unknown = v
                
                if c0 == 5 or c1 == 5:
                    return False  # monochromatic edge
                
                if unknown is not None:
                    if c0 == 4:
                        # Must be color 1
                        if assignment[unknown] == 0:
                            return False
                        if assignment[unknown] == -1:
                            assignment[unknown] = 1
                            changed = True
                    elif c1 == 4:
                        # Must be color 0
                        if assignment[unknown] == 1:
                            return False
                        if assignment[unknown] == -1:
                            assignment[unknown] = 0
                            changed = True
        return True
    
    def backtrack():
        # Check for immediate contradictions
        if not propagate():
            return False
        
        # Find unassigned vertex with most constraints (MRV-like)
        unassigned = [v for v in range(n) if assignment[v] == -1]
        if not unassigned:
            # All assigned, verify no monochromatic edges
            for e in edge_set:
                c0 = sum(1 for v in e if assignment[v] == 0)
                c1 = sum(1 for v in e if assignment[v] == 1)
                if c0 == 5 or c1 == 5:
                    return False
            return True  # Valid coloring found
        
        # Pick vertex with most incident edges
        v = max(unassigned, key=lambda x: len(vert_to_edges[x]))
        
        for color in [0, 1]:
            old_val = assignment[v]
            assignment[v] = color
            if backtrack():
                return True
            assignment[v] = old_val
        
        return False
    
    return not backtrack()

def _try_construct(n, target_edges, seed=42, time_limit=10):
    """
    Try to find a non-2-colorable hypergraph on n vertices with ~target_edges edges.
    Uses greedy random addition with backtracking check.
    """
    import random
    rng = random.Random(seed)
    
    all_edges = list(combinations(range(n), 5))
    rng.shuffle(all_edges)
    
    edges = []
    start_time = time.time()
    
    for e in all_edges:
        if time.time() - start_time > time_limit:
            break
        if len(edges) >= target_edges:
            break
        edges.append(list(e))
    
    if len(edges) < target_edges:
        return None
    
    # Check if non-2-colorable
    if _check_non_2colorable(n, edges):
        return edges
    
    return None

def _try_systematic(n, time_limit=10):
    """
    Try systematic constructions for small n.
    """
    start_time = time.time()
    
    # For n=9, complete hypergraph is 126 edges - too many
    # Try to find smaller
    
    # Strategy: start with a base set and add edges
    # One known approach: use the fact that we need to block all 2-colorings
    
    # Try: take a random subset of edges and check
    import random
    rng = random.Random(123)
    
    all_edges = list(combinations(range(n), 5))
    rng.shuffle(all_edges)
    
    # Try different sizes
    for size in range(29, min(60, len(all_edges) + 1)):
        if time.time() - start_time > time_limit:
            break
        edges = [list(e) for e in all_edges[:size]]
        if _check_non_2colorable(n, edges):
            return edges
    
    return None

def construct():
    """
    Return a non-2-colorable 5-uniform hypergraph with minimal edges.
    """
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Fallback: complete hypergraph on 9 vertices (126 edges)
    fallback = {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    best = None
    best_edges_count = 126  # fallback has 126 edges
    
    # Try various n and strategies
    for n in range(10, 17):
        if time.time() > deadline - 30:
            break
        
        # Try systematic approach
        result = _try_systematic(n, time_limit=min(15, max(5, deadline - time.time() - 30)))
        if result is not None:
            if len(result) < best_edges_count:
                best_edges_count = len(result)
                best = {'n': n, 'edges': result}
        
        if time.time() > deadline - 30:
            break
    
    # Try random search with more iterations
    for n in range(10, 15):
        if time.time() > deadline - 20:
            break
        
        for seed in range(100):
            if time.time() > deadline - 20:
                break
            
            target = best_edges_count - 1 if best else 50
            target = max(29, min(target, 55))
            
            result = _try_construct(n, target, seed=seed, time_limit=min(5, max(2, deadline - time.time() - 20)))
            if result is not None:
                if len(result) < best_edges_count:
                    best_edges_count = len(result)
                    best = {'n': n, 'edges': result}
    
    if best is not None:
        return best
    
    return fallback
# EVOLVE-BLOCK-END
