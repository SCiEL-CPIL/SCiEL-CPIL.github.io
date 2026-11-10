import time
import random
from itertools import combinations

def _is_non_2colorable(n, edges):
    """Fast backtracking solver."""
    if not edges:
        return True # Empty graph is 2-colorable? No, empty is trivially 2-colorable.
                   # Wait, "no proper 2-colouring".
                   # Empty graph has a proper 2-colouring. So it IS 2-colorable.
                   # We want NON-2-colorable.
                   # So empty graph returns False.
    
    edge_set = [tuple(sorted(e)) for e in edges]
    # Remove duplicates
    edge_set = list(set(edge_set))
    
    vert_to_edges = {v: [] for v in range(n)}
    for i, e in enumerate(edge_set):
        for v in e:
            vert_to_edges[v].append(i)
            
    # colors: -1 unassigned, 0, 1
    colors = [-1] * n
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edge_set):
                c = [colors[v] for v in e]
                zeros = c.count(0)
                ones = c.count(1)
                unknowns = c.count(-1)
                
                if zeros + ones == 5:
                    if zeros == 5 or ones == 5:
                        return False # Conflict
                
                if unknowns == 1:
                    if zeros == 4:
                        # 5th must be 1
                        for v in e:
                            if colors[v] == -1:
                                if colors[v] != 1: # It is -1
                                    colors[v] = 1
                                    changed = True
                    elif ones == 4:
                        # 5th must be 0
                        for v in e:
                            if colors[v] == -1:
                                colors[v] = 0
                                changed = True
        return True

    def backtrack(idx):
        if idx == n:
            # Check if any edge is monochromatic
            # Actually, propagation should have caught conflicts if we assign all.
            # But we need to ensure no monochromatic edge.
            # If we assigned all vertices, and no conflict was found during propagation,
            # is it guaranteed?
            # Yes, if an edge became mono, propagation would return False.
            return True
        
        v = idx
        # Heuristic: Pick vertex with most constraints?
        # For speed, just pick first unassigned?
        # Better: MRV.
        # But here we are iterating 0..n-1.
        # Let's stick to simple order for now, or swap to MRV.
        
        # Try 0
        colors[v] = 0
        if propagate():
            if backtrack(idx + 1):
                return True
        
        # Try 1
        colors[v] = 1
        if propagate():
            if backtrack(idx + 1):
                return True
        
        colors[v] = -1
        return False

    # Initialize propagation
    if not propagate():
        return True # Non-2-colorable (conflict immediately?) No, empty assignment no conflict.
    
    # Find first unassigned
    # Simple backtrack
    return not backtrack(0)

def construct():
    deadline = time.time() + 145
    best_edges = None
    best_n = 0
    
    # Fallback: Complete K9
    n_fb = 9
    edges_fb = [list(e) for e in combinations(range(n_fb), 5)]
    if _is_non_2colorable(n_fb, edges_fb):
        best_edges = edges_fb
        best_n = n_fb
    else:
        # This shouldn't happen
        return {'n': 9, 'edges': edges_fb}

    # Search for better
    # Strategy: Start with dense random, then minimize.
    
    for n in range(10, 16):
        if time.time() > deadline - 5:
            break
            
        # Try different densities
        # High density likely non-2-colorable
        all_edges = list(combinations(range(n), 5))
        
        for trial in range(20):
            if time.time() > deadline - 5:
                break
                
            # Start with random subset
            # Size: maybe 50% of all edges?
            # For n=10, 252 edges. 50% = 126.
            # For n=11, 462 edges. 50% = 231.
            # We want to end up with small E.
            # Start with full or near-full, then prune.
            
            random.shuffle(all_edges)
            current_edges = [list(e) for e in all_edges]
            
            if not _is_non_2colorable(n, current_edges):
                continue
                
            # Prune
            pruned = list(current_edges)
            # To speed up pruning, we can iterate edges and try to remove.
            # Random order for pruning?
            
            i = 0
            while i < len(pruned):
                if time.time() > deadline - 5:
                    break
                    
                e_to_try = pruned[i]
                # Remove e_to_try
                test_edges = pruned[:i] + pruned[i+1:]
                
                if _is_non_2colorable(n, test_edges):
                    # Can remove
                    pruned = test_edges
                    # Do not increment i, because list shifted
                else:
                    i += 1
            
            if len(pruned) < len(best_edges):
                best_edges = pruned
                best_n = n
                print(f"Found non-2-colorable with n={n}, edges={len(pruned)}")
                
                # Try to prune more? Or restart with smaller start?
                # If we found one with < 50 edges, we are doing well.
                
    return {'n': best_n, 'edges': best_edges}
