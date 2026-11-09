import time
import random
from itertools import combinations

START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin for 120s limit

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def check_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable (has Property B).
    Returns False if it is NOT 2-colorable (no Property B).
    Uses DPLL with unit propagation.
    """
    if not edges:
        return True
    
    # Normalize edges
    unique_edges = []
    seen = set()
    for e in edges:
        e_sorted = tuple(sorted(e))
        if e_sorted not in seen:
            seen.add(e_sorted)
            unique_edges.append(list(e_sorted))
    edges = unique_edges
    m = len(edges)
    if m == 0:
        return True

    # colors[v]: -1 (unassigned), 0, 1
    colors = [-1] * n
    
    # Precompute edge indices for faster access?
    # For now, simple iteration.
    
    def propagate():
        """
        Apply unit propagation.
        If an edge has 4 vertices of color c, the 5th must be 1-c.
        If an edge has 5 vertices of color c, contradiction.
        If an edge has both 0 and 1, it is satisfied (cannot be monochromatic).
        """
        changed = True
        while changed:
            changed = False
            for e in edges:
                counts = [0, 0]
                unassigned = []
                for v in e:
                    if colors[v] == -1:
                        unassigned.append(v)
                    else:
                        counts[colors[v]] += 1
                
                if len(unassigned) == 0:
                    # Check for contradiction
                    if counts[0] == 5 or counts[1] == 5:
                        return False
                    # If counts are like 3-2, it's fine (not monochromatic)
                
                # Unit propagation
                if counts[0] == 4 and len(unassigned) == 1:
                    colors[unassigned[0]] = 1
                    changed = True
                elif counts[1] == 4 and len(unassigned) == 1:
                    colors[unassigned[0]] = 0
                    changed = True
        return True

    def select_variable():
        """
        Heuristic: Select the unassigned vertex that appears in the most 'threatening' edges.
        Threatening edge: An edge with 3 or 4 assigned vertices.
        """
        best_v = -1
        max_score = -1
        
        # Count appearances in edges with high assignment density
        scores = [0] * n
        for e in edges:
            assigned_count = sum(1 for v in e if colors[v] != -1)
            if assigned_count >= 3:
                for v in e:
                    if colors[v] == -1:
                        scores[v] += (5 - assigned_count) # Higher score for more constrained
                        # Actually, just count edges where it's the only or one of few unassigned
                        if assigned_count == 4:
                            scores[v] += 10 # Very high priority
                        elif assigned_count == 3:
                            scores[v] += 2
        
        for v in range(n):
            if colors[v] == -1 and scores[v] > max_score:
                max_score = scores[v]
                best_v = v
        
        if best_v == -1:
            # Fallback: first unassigned
            for v in range(n):
                if colors[v] == -1:
                    return v
            return -1 # All assigned
        
        return best_v

    def backtrack():
        v = select_variable()
        if v == -1:
            # All assigned. Check if any edge is monochromatic.
            # Propagation should have caught contradictions, but double check.
            for e in edges:
                c = colors[e[0]]
                for u in e[1:]:
                    if colors[u] != c:
                        break
                else:
                    return False # Monochromatic edge found -> Not a valid 2-coloring
            return True # Valid coloring found -> 2-colorable

        # Try 0
        colors_backup = colors.copy()
        colors[v] = 0
        if propagate():
            if backtrack():
                return True
        colors[:] = colors_backup

        # Try 1
        colors[v] = 1
        if propagate():
            if backtrack():
                return True
        colors[:] = colors_backup

        return False

    # Symmetry breaking: Fix vertex 0 to 0
    colors[0] = 0
    if not propagate():
        return False # Contradiction immediately? If so, not 2-colorable? 
        # Wait. If fixing v0=0 leads to contradiction, it means no coloring with v0=0 exists.
        # But there might be a coloring with v0=1.
        # However, the problem is symmetric under color swap.
        # If no coloring with v0=0 exists, then no coloring with v0=1 exists either?
        # Yes, because if (c0, ..., cn-1) is a solution, then (1-c0, ..., 1-cn-1) is also a solution.
        # So if no solution has c0=0, then no solution has c0=1.
        # Thus, if propagate fails with c0=0, the graph is NOT 2-colorable.
        # So return False.
    
    return backtrack()

def minimize_edges(n, edges):
    """
    Try to remove edges from the hypergraph while keeping it non-2-colorable.
    """
    current_edges = edges[:]
    improved = True
    while improved and time_left() > 5:
        improved = False
        # Shuffle indices to try different removals
        idxs = list(range(len(current_edges)))
        random.shuffle(idxs)
        for i in idxs:
            if time_left() < 5:
                break
            # Try removing edge i
            test_edges = current_edges[:i] + current_edges[i+1:]
            # Check if still non-2-colorable
            if not check_2_colorable(n, test_edges):
                current_edges = test_edges
                improved = True
                break # Restart loop with new set
    return current_edges

def search_small_hypergraph():
    best_edges = None
    best_n = None
    
    # Try different n values
    # Known bounds suggest n around 10-15
    for n in [10, 11, 12, 13, 14, 15]:
        if time_left() < 10:
            break
            
        # Try to find a non-2-colorable hypergraph
        # Random search
        for trial in range(500):
            if time_left() < 5:
                break
            
            # Number of edges to try
            # Start with a range likely to be non-2-colorable
            # For n=10, C(10,5)=252. 
            # Let's try 30 to 60 edges.
            m = random.randint(30, 60)
            
            all_possible = list(combinations(range(n), 5))
            random.shuffle(all_possible)
            edges = [list(e) for e in all_possible[:m]]
            
            # Check
            if not check_2_colorable(n, edges):
                # Found non-2-colorable
                # Minimize
                minimized = minimize_edges(n, edges)
                
                if best_edges is None or len(minimized) < len(best_edges):
                    best_edges = minimized
                    best_n = n
                    # If we get very low, stop early
                    if len(minimized) <= 30:
                        break
                # Continue searching for potentially smaller ones in this trial?
                # No, we already minimized. Try next trial or next n.
                break # Break trial loop for this n, try next n? 
                # Actually, different random instances might yield different minimal sets.
                # But minimizing a random instance is a good heuristic.
                # Let's try a few more trials for this n if time permits.
                # But we break to save time for other n's.
                
        # End trial loop
        
    if best_edges is not None:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback: Complete graph on 9 vertices
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}

def construct():
    try:
        result = search_small_hypergraph()
        return result
    except Exception:
        # Fallback to original
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
