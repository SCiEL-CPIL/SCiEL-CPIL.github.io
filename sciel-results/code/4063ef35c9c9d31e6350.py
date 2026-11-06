"""
Search for a small 5-uniform hypergraph with no proper 2-colouring.
Strategy:
1. Generate candidate edge sets via greedy randomized construction.
2. Verify non-2-colourability with backtracking + unit propagation.
3. Minimise edge count via greedy removal and local search.
4. Deadline guard to stay within 150s.
"""

import time
import random
from itertools import combinations
from collections import defaultdict

DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # seconds before deadline to stop

def is_2_colorable(n, edges, timeout=2.0):
    """
    Check if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    start_time = time.time()
    
    # Build adjacency: for each vertex, list of edges containing it
    vertex_edges = defaultdict(list)
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    # State: assignment dict, conflicts set
    # We use a recursive backtracking with unit propagation
    
    def propagate(assignment, edge_satisfied):
        """
        Unit propagation: if an edge has 4 vertices of same color, the 5th must be different.
        Also, if an edge is fully assigned and monochromatic, conflict.
        Returns (assignment, edge_satisfied, conflict)
        """
        changed = True
        while changed:
            changed = False
            for v in list(assignment.keys()):
                c = assignment[v]
                for ei in vertex_edges[v]:
                    if ei in edge_satisfied:
                        continue
                    e = edges[ei]
                    # Count colors in this edge
                    same = 0
                    diff = 0
                    unassigned = None
                    for u in e:
                        if u not in assignment:
                            unassigned = u
                        elif assignment[u] == c:
                            same += 1
                        else:
                            diff += 1
                    # If 4 same and 1 unassigned, force the unassigned to be different
                    if same == 4 and unassigned is not None and unassigned not in assignment:
                        assignment[unassigned] = 1 - c
                        edge_satisfied.add(ei)  # mark as processed (will be checked)
                        changed = True
                    # Check if edge is now monochromatic (conflict)
                    if all(u in assignment for u in e):
                        colors = set(assignment[u] for u in e)
                        if len(colors) == 1:
                            return None, None, True
                        edge_satisfied.add(ei)
        return assignment, edge_satisfied, False
    
    # Recursive backtracking
    def backtrack(assignment, edge_satisfied):
        if time.time() - start_time > timeout:
            return "timeout"
        
        # Check for conflicts
        for ei in list(edge_satisfied):
            e = edges[ei]
            if all(u in assignment for u in e):
                colors = set(assignment[u] for u in e)
                if len(colors) == 1:
                    return False
        
        # Try to find an unassigned vertex
        unassigned = [v for v in range(n) if v not in assignment]
        if not unassigned:
            # All assigned, check all edges
            for e in edges:
                colors = set(assignment[u] for u in e)
                if len(colors) == 1:
                    return False
            return True
        
        # Choose vertex with most constraints
        v = max(unassigned, key=lambda x: len(vertex_edges[x]))
        
        for c in [0, 1]:
            new_assignment = dict(assignment)
            new_assignment[v] = c
            new_edge_sat = set(edge_satisfied)
            
            # Propagate
            result = propagate(new_assignment, new_edge_sat)
            if result[2]:  # conflict
                continue
            new_assignment, new_edge_sat, _ = result
            
            # Check for immediate conflicts
            conflict = False
            for ei in new_edge_sat:
                e = edges[ei]
                if all(u in new_assignment for u in e):
                    colors = set(new_assignment[u] for u in e)
                    if len(colors) == 1:
                        conflict = True
                        break
            if conflict:
                continue
            
            res = backtrack(new_assignment, new_edge_sat)
            if res == "timeout":
                return "timeout"
            if res:
                return True
        
        return False
    
    return backtrack({}, set())

def try_remove_edge(n, edges, edge_idx):
    """Try removing edge at index and check if still non-2-colorable."""
    new_edges = [e for i, e in enumerate(edges) if i != edge_idx]
    if len(new_edges) == len(edges):
        return False
    return not is_2_colorable(n, new_edges, timeout=1.0)

def greedy_construct(n, num_edges, seed=None):
    """
    Greedy construction: start with random edges, try to add edges that break 2-colorability.
    """
    if seed is not None:
        random.seed(seed)
    
    # Start with a base set of edges
    all_edges = list(combinations(range(n), 5))
    random.shuffle(all_edges)
    
    # Start with a subset
    edges = [list(e) for e in all_edges[:min(num_edges, len(all_edges))]]
    
    # Try to add edges to make it non-2-colorable
    remaining = [list(e) for e in all_edges[num_edges:]]
    random.shuffle(remaining)
    
    for e in remaining:
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        new_edges = edges + [e]
        if not is_2_colorable(n, new_edges, timeout=0.5):
            edges = new_edges
            if len(edges) >= num_edges:
                break
    
    return edges

def local_search(n, edges, max_iterations=100):
    """
    Local search: try removing edges one by one, keep if still non-2-colorable.
    """
    edges = list(edges)
    improved = True
    while improved and time.time() < DEADLINE - SAFETY_MARGIN:
        improved = False
        for i in range(len(edges)):
            if time.time() > DEADLINE - SAFETY_MARGIN:
                break
            if try_remove_edge(n, edges, i):
                edges.pop(i)
                improved = True
                break
    return edges

def construct():
    start_time = time.time()
    best = None
    best_count = float('inf')
    
    # Known good starting point: complete 5-uniform on 9 vertices has 126 edges
    # We want to find smaller ones
    
    # Try different n values
    for n in [9, 10, 11, 12, 13]:
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        # Try multiple random seeds
        for seed in range(50):
            if time.time() > DEADLINE - SAFETY_MARGIN:
                break
            
            # Greedy construction
            edges = greedy_construct(n, 30 + seed % 20, seed=seed)
            
            if not edges:
                continue
            
            # Verify
            if not is_2_colorable(n, edges, timeout=1.0):
                # It's non-2-colorable, try to minimize
                minimized = local_search(n, edges)
                if len(minimized) < best_count:
                    best_count = len(minimized)
                    best = {'n': n, 'edges': minimized}
                    print(f"Found: n={n}, edges={best_count}")
    
    # If no better found, return the complete 5-uniform on 9 vertices
    if best is None:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    return best
