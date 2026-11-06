# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_not_2_colorable_fast(n, edges):
    """
    Check if hypergraph is NOT 2-colorable.
    Returns True if NOT 2-colorable (i.e., no valid coloring exists).
    Uses backtracking with unit propagation and bitmasks.
    """
    if not edges:
        return False
        
    edge_set = [tuple(sorted(e)) for e in edges]
    
    # Map vertices to bit positions
    # We only care about vertices that appear in edges
    active_vertices = set()
    for e in edge_set:
        active_vertices.update(e)
    
    if not active_vertices:
        return False

    # Compress vertex indices to 0..m-1
    v_map = {v: i for i, v in enumerate(sorted(active_vertices))}
    m = len(v_map)
    
    # Convert edges to list of compressed indices
    comp_edges = []
    for e in edge_set:
        comp_edges.append([v_map[v] for v in e])
        
    # Build incidence: vertex -> list of edge indices
    vert_edges = [[] for _ in range(m)]
    for i, e in enumerate(comp_edges):
        for v in e:
            vert_edges[v].append(i)
            
    # State: assignment[v] = -1 (unassigned), 0 or 1
    # We will use a list of size m
    
    def unit_propagate(assignment):
        """
        Apply unit propagation.
        Returns False if contradiction, True otherwise.
        Modifies assignment in place.
        """
        changed = True
        while changed:
            changed = False
            for e in comp_edges:
                c0 = 0
                c1 = 0
                unassigned = []
                for v in e:
                    a = assignment[v]
                    if a == 0:
                        c0 += 1
                    elif a == 1:
                        c1 += 1
                    else:
                        unassigned.append(v)
                
                # Contradiction: all same color
                if c0 == 5 or c1 == 5:
                    return False
                
                # Unit clause: 4 of one color, 1 unassigned
                if c0 == 4 and len(unassigned) == 1:
                    assignment[unassigned[0]] = 1
                    changed = True
                elif c1 == 4 and len(unassigned) == 1:
                    assignment[unassigned[0]] = 0
                    changed = True
                    
        return True

    def backtrack(assignment):
        """
        Returns True if a valid 2-coloring exists.
        """
        # Find first unassigned vertex with highest degree (most constraints)
        idx = -1
        max_degree = -1
        for i in range(m):
            if assignment[i] == -1:
                deg = len(vert_edges[i])
                if deg > max_degree:
                    max_degree = deg
                    idx = i
                    
        if idx == -1:
            # All assigned, check if any edge is monochromatic
            # (Should be caught by propagation, but double check)
            for e in comp_edges:
                c0 = sum(1 for v in e if assignment[v] == 0)
                c1 = sum(1 for v in e if assignment[v] == 1)
                if c0 == 5 or c1 == 5:
                    return False
            return True
            
        for color in [0, 1]:
            new_assignment = assignment[:]
            new_assignment[idx] = color
            
            if unit_propagate(new_assignment):
                if backtrack(new_assignment):
                    return True
                    
        return False

    initial_assignment = [-1] * m
    is_colorable = backtrack(initial_assignment)
    
    return not is_colorable

def _try_to_make_non_2_colorable(n, target_edges, time_limit):
    """
    Randomly generate edges until the hypergraph is non-2-colorable.
    """
    start_time = time.time()
    all_edges = list(combinations(range(n), 5))
    random.shuffle(all_edges)
    
    edges = []
    edge_set = set()
    
    # Add edges one by one
    for e in all_edges:
        if time.time() - start_time > time_limit:
            break
        if len(edges) >= target_edges:
            break
            
        edges.append(e)
        edge_set.add(e)
        
        # Check periodically to save time
        if len(edges) % 5 == 0:
            if _is_not_2_colorable_fast(n, edges):
                return edges
                
    # Final check
    if edges and _is_not_2_colorable_fast(n, edges):
        return edges
        
    return None

def _minimize_edges(n, edges, time_limit):
    """
    Greedy edge removal: try to remove each edge if the hypergraph remains non-2-colorable.
    """
    start_time = time.time()
    current_edges = [list(e) for e in edges]
    
    # Shuffle edges to avoid bias
    indices = list(range(len(current_edges)))
    
    improved = True
    while improved:
        improved = False
        random.shuffle(indices)
        
        for i in indices:
            if time.time() - start_time > time_limit:
                return current_edges
                
            if i >= len(current_edges):
                continue
                
            # Try removing edge i
            edge_to_remove = current_edges[i]
            new_edges = current_edges[:i] + current_edges[i+1:]
            
            # Check if new hypergraph is non-2-colorable
            if _is_not_2_colorable_fast(n, new_edges):
                # Keep it removed
                current_edges = new_edges
                improved = True
                # Note: indices are now shifted, but we are in a while loop 
                # that reshuffles, so this is okay. 
                # However, modifying the list while iterating over indices 
                # based on the old length is tricky. 
                # The `if i >= len(current_edges)` check helps, but 
                # subsequent indices refer to the NEW list.
                # It's safer to restart the loop or handle carefully.
                # Let's just break and restart the while loop to be safe.
                break
                
    return current_edges

def construct():
    """
    Find a small non-2-colorable 5-uniform hypergraph.
    """
    start_time = time.time()
    deadline = 130  # Safety margin
    
    best_edges = None
    best_n = None
    
    # Try different n values
    # Focus on n=9 and n=10 first as they are likely to have smaller edge counts
    # n=9 has 126 total edges, n=10 has 252
    for n in [9, 10, 11, 12, 13, 14]:
        if time.time() > deadline:
            break
            
        # Estimate target edges: start high to ensure non-2-colorability, then minimize
        # For n=9, max edges is 126.
        # For n=10, max edges is 252.
        
        # We want to find a non-2-colorable one.
        # Randomly picking edges might take a while.
        # Let's try to build one.
        
        target_edges = 80  # Start with a decent number
        
        # Try a few times to find an initial non-2-colorable hypergraph
        initial_found = None
        for _ in range(15):
            if time.time() > deadline:
                break
            edges = _try_to_make_non_2_colorable(n, target_edges, 3)
            if edges:
                initial_found = edges
                break
                
        if not initial_found:
            continue
            
        # Minimize
        # Allocate more time for minimization if we have budget
        time_left = deadline - time.time()
        min_time = min(15, max(5, time_left * 0.5))
        minimized = _minimize_edges(n, initial_found, min_time)
        
        if len(minimized) < (len(best_edges) if best_edges else float('inf')):
            best_edges = minimized
            best_n = n
            
        # If we found something very good, we might stop early?
        # The task says best known is 51. If we get < 51, great.
        # If we get 51, we can try to improve or stop if time is tight.
        if best_edges and len(best_edges) <= 52:
            # We have a very good result, try to squeeze a bit more time for minimization
            # or stop if we are very close to the deadline
            if time.time() > deadline - 10:
                break

    if best_edges is None:
        # Fallback: Complete hypergraph on 9 vertices (126 edges)
        # This is guaranteed non-2-colorable?
        # Actually, complete hypergraph on 9 vertices:
        # Is it non-2-colorable?
        # If we color vertices, any edge is monochromatic if all 5 are same.
        # With 9 vertices, can we avoid monochromatic edges?
        # This is equivalent to finding a 2-coloring of K_9^5 with no monochromatic edge.
        # This is related to Ramsey numbers.
        # R(5,5;5) is huge.
        # Actually, for n=9, it is likely 2-colorable?
        # Let's check: If we color 4 vertices 0 and 5 vertices 1.
        # An edge with 5 vertices from the 5-vertex set is mono 1.
        # So we need to distribute colors such that no 5-subset is mono.
        # Max size of a 0-set is 4, max size of a 1-set is 4? No, 4+4=8 < 9.
        # So one color must have at least 5 vertices.
        # If a color has 5 vertices, the edge formed by them is monochromatic.
        # Thus, K_9^5 is NOT 2-colorable.
        # So 126 edges is a valid fallback.
        
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': [list(e) for e in best_edges]}
# EVOLVE-BLOCK-END
