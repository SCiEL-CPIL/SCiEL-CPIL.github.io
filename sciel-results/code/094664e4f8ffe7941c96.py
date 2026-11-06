# EVOLVE-BLOCK-START
import itertools
import time

def _is_2_colorable(n, edges):
    """
    Checks if the hypergraph is 2-colorable (i.e., there exists a coloring 
    with no monochromatic edges).
    Returns True if 2-colorable, False otherwise.
    """
    # Preprocess edges for fast lookup
    # For each vertex, store list of edges containing it
    # For unit propagation, we need to check edges quickly.
    
    # State: color of each vertex: 0 (unassigned), 1 (Red), 2 (Blue)
    colors = [0] * n
    
    # Edges as tuples for hashing
    edge_set = [tuple(e) for e in edges]
    
    # For efficient propagation, we can just iterate all edges in each step?
    # n is small (<= 12), edges <= 200.
    # Iterating all edges is fine.
    
    def propagate():
        """
        Applies unit propagation.
        Returns False if a conflict is found (monochromatic edge).
        Modifies colors.
        """
        changed = True
        while changed:
            changed = False
            for e in edge_set:
                # Count colors in this edge
                c0 = c1 = c2 = 0
                unassigned = None
                for v in e:
                    c = colors[v]
                    if c == 0:
                        unassigned = v
                    elif c == 1:
                        c1 += 1
                    else:
                        c2 += 1
                
                if c1 == 5 or c2 == 5:
                    return False # Monochromatic edge found
                
                # Unit propagation
                # If 4 are Red, the unassigned must be Blue
                if c1 == 4 and unassigned is not None and colors[unassigned] == 0:
                    colors[unassigned] = 2
                    changed = True
                elif c2 == 4 and unassigned is not None and colors[unassigned] == 0:
                    colors[unassigned] = 1
                    changed = True
        return True

    def backtrack(idx):
        """
        Backtracking search.
        """
        # Find first unassigned vertex
        # To speed up, pick vertex involved in most constraints?
        # For small n, just pick first unassigned.
        
        # Check if all assigned
        # If all assigned, we need to verify no monochromatic edge.
        # But propagation ensures no monochromatic edge IF it completes successfully?
        # Propagation only triggers on 4-1 or 4-2.
        # If we assign the last vertex, we might create a 5-0 or 0-5?
        # So we must check edges at the end or during propagation.
        
        # Actually, if propagation returns True, it means no immediate conflict.
        # But if all vertices are assigned, we must ensure no edge is monochromatic.
        
        if all(c != 0 for c in colors):
             # Verify
             for e in edge_set:
                 c1 = sum(1 for v in e if colors[v] == 1)
                 if c1 == 5 or c1 == 0:
                     return False
             return True

        # Find first unassigned
        v = -1
        for i in range(n):
            if colors[i] == 0:
                v = i
                break
        
        if v == -1:
            # Should be handled by all() check
            return True

        # Try color 1
        colors[v] = 1
        if propagate():
            if backtrack(idx + 1):
                return True
        colors[v] = 0
        
        # Try color 2
        colors[v] = 2
        if propagate():
            if backtrack(idx + 1):
                return True
        colors[v] = 0
        
        return False

    # Start backtracking
    # We can add a simple heuristic: pick vertex with highest degree?
    # For now, simple index order.
    
    # Optimization: If we find a valid coloring, return True immediately.
    # The function returns True if 2-colorable.
    
    # We need to reset colors for each call?
    # Yes, colors is local.
    
    # Initial propagation?
    if not propagate():
        return False
    
    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Try n=10 first, as n=9 was 126.
    # If n=10 yields too many edges or takes too long, try n=11?
    # Actually, smaller n is usually denser.
    # Let's stick to n=10.
    
    n = 10
    all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
    
    # Greedy removal
    # We maintain a list of edges.
    # To avoid O(N^2) checks with list removal, we can use a set of edge tuples.
    # But we need to iterate.
    
    current_edges = set(tuple(e) for e in all_edges)
    
    # To speed up removal, we can iterate multiple times.
    # Or just one pass? One pass might miss opportunities if removing one edge enables removing another?
    # No, if removing e1 makes it 2-colorable, e1 is "essential".
    # If removing e2 (with e1 present) makes it 2-colorable, e2 is "essential" in that context.
    # If we remove e1 (if possible), the status of e2 might change.
    # So multiple passes are needed for a *minimal* hypergraph.
    # But we just want *few* edges. Greedy one-pass is often good enough for a significant reduction.
    # Let's do a few passes until no change or time limit.
    
    changed = True
    passes = 0
    while changed and time.time() < deadline:
        changed = False
        passes += 1
        # Iterate over a copy of the list
        edges_list = list(current_edges)
        # Shuffle to avoid bias?
        # import random
        # random.shuffle(edges_list)
        
        for e in edges_list:
            if time.time() > deadline:
                break
            if e not in current_edges:
                continue
            
            # Try removing e
            current_edges.remove(e)
            
            # Check if still NOT 2-colorable
            # is_2_colorable returns True if 2-colorable.
            # We want it to be False.
            if _is_2_colorable(n, [list(x) for x in current_edges]):
                # It became 2-colorable, so e was needed.
                current_edges.add(e)
            else:
                # Still not 2-colorable, keep it removed.
                changed = True
        
        if time.time() > deadline:
            break

    return {'n': n, 'edges': [list(e) for e in current_edges]}
# EVOLVE-BLOCK-END
