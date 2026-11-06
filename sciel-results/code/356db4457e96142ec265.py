"""Direct construction of a small 5-uniform hypergraph without Property B."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # safety margin under 150s

def is_2_colorable_fast(n, edges):
    """
    Check if a 5-uniform hypergraph is 2-colorable.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    # Convert edges to bitmasks for fast checking
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # For small n, brute force is fastest
    if n <= 14:
        for color_mask in range(1 << n):
            valid = True
            for em in edge_masks:
                # Check if em is monochromatic in color_mask
                if (em & color_mask) == 0 or (em & color_mask) == em:
                    valid = False
                    break
            if valid:
                return True
        return False
    else:
        # For larger n, use backtracking with unit propagation
        color = [-1] * n
        c0 = [0] * len(edges)
        c1 = [0] * len(edges)
        
        # Precompute vertex to edges
        vertex_edges = [[] for _ in range(n)]
        for i, em in enumerate(edge_masks):
            for v in range(n):
                if em & (1 << v):
                    vertex_edges[v].append(i)
        
        def propagate():
            changed = True
            while changed:
                changed = False
                for i in range(len(edges)):
                    if c0[i] == 5 or c1[i] == 5:
                        return False
                    if c0[i] == 4 and c1[i] == 0:
                        for v in range(n):
                            if color[v] == -1 and (edge_masks[i] & (1 << v)):
                                color[v] = 1
                                c1[i] += 1
                                for other_e in vertex_edges[v]:
                                    c1[other_e] += 1
                                changed = True
                                if c1[i] == 5:
                                    return False
                                break
                    elif c1[i] == 4 and c0[i] == 0:
                        for v in range(n):
                            if color[v] == -1 and (edge_masks[i] & (1 << v)):
                                color[v] = 0
                                c0[i] += 1
                                for other_e in vertex_edges[v]:
                                    c0[other_e] += 1
                                changed = True
                                if c0[i] == 5:
                                    return False
                                break
            return True
        
        import sys
        sys.setrecursionlimit(10000)
        
        def backtrack():
            if time.time() - _START_TIME > _TIME_LIMIT:
                return True
            
            if not propagate():
                return False
            
            # Find most constrained edge
            best_v = -1
            best_edge = -1
            best_unassigned = 6
            best_constraint = 0
            
            for i in range(len(edges)):
                if c0[i] == 5 or c1[i] == 5:
                    return False
                unassigned = 5 - c0[i] - c1[i]
                constraint = max(c0[i], c1[i])
                if unassigned < best_unassigned or (unassigned == best_unassigned and constraint > best_constraint):
                    best_unassigned = unassigned
                    best_constraint = constraint
                    best_edge = i
                    if unassigned == 0:
                        return False
            
            if best_edge == -1:
                return True
            
            candidates = []
            for v in range(n):
                if color[v] == -1 and (edge_masks[best_edge] & (1 << v)):
                    candidates.append(v)
            
            if not candidates:
                return True
            
            for v in candidates:
                # Try 0
                saved_c0 = c0[:]
                saved_c1 = c1[:]
                color[v] = 0
                ok = True
                for e_idx in vertex_edges[v]:
                    c0[e_idx] += 1
                    if c0[e_idx] == 5:
                        ok = False
                        break
                if ok and propagate():
                    if backtrack():
                        return True
                color[v] = -1
                c0[:] = saved_c0
                c1[:] = saved_c1
                
                # Try 1
                saved_c0 = c0[:]
                saved_c1 = c1[:]
                color[v] = 1
                ok = True
                for e_idx in vertex_edges[v]:
                    c1[e_idx] += 1
                    if c1[e_idx] == 5:
                        ok = False
                        break
                if ok and propagate():
                    if backtrack():
                        return True
                color[v] = -1
                c0[:] = saved_c0
                c1[:] = saved_c1
            
            return False
        
        return backtrack()

def construct():
    """
    Return a 5-uniform hypergraph on n vertices (5 <= n <= 26, <=200 edges)
    that has no proper 2-colouring, minimizing the number of edges.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_edges = None
    best_n = 9
    best_m = 10**9
    
    # Try different n values
    # The 51-edge construction likely exists for n=11 or 12
    for n in [9, 10, 11, 12, 13]:
        if time.time() > deadline - 20:
            break
        
        all_possible = list(combinations(range(n), 5))
        
        # Start with complete hypergraph
        current_edges = [list(e) for e in all_possible]
        
        # Check if complete is non-2-colorable
        if is_2_colorable_fast(n, current_edges):
            continue
        
        # Greedy removal
        # We will do multiple passes with random shuffles
        for attempt in range(20):
            if time.time() > deadline - 20:
                break
            
            edge_indices = list(range(len(current_edges)))
            random.shuffle(edge_indices)
            
            removed_any = True
            while removed_any and time.time() < deadline - 20:
                removed_any = False
                for i in edge_indices:
                    if time.time() > deadline - 20:
                        break
                    if i >= len(current_edges):
                        continue
                    
                    # Try removing edge i
                    new_edges = current_edges[:i] + current_edges[i+1:]
                    
                    # Check if still non-2-colorable
                    if not is_2_colorable_fast(n, new_edges):
                        current_edges = new_edges
                        removed_any = True
                        break
            
            if len(current_edges) < best_m:
                best_m = len(current_edges)
                best_n = n
                best_edges = current_edges[:]
        
        # If we already found a very good solution, stop
        if best_m <= 55:
            break
    
    if best_edges is None:
        # Fallback: return complete 5-graph on 9 vertices
        n = 9
        all_edges = [list(e) for e in combinations(range(n), 5)]
        return {'n': n, 'edges': all_edges}
    
    # Verify the result is valid
    assert 5 <= best_n <= 26
    assert len(best_edges) <= 200
    assert len(set(tuple(sorted(e)) for e in best_edges)) == len(best_edges)
    for e in best_edges:
        assert len(e) == 5
        assert len(set(e)) == 5
        assert all(0 <= v < best_n for v in e)
    
    # Final verification that it's NOT 2-colourable
    if is_2_colorable_fast(best_n, best_edges):
        # Fallback: return the complete graph
        n = 9
        all_edges = [list(e) for e in combinations(range(n), 5)]
        return {'n': n, 'edges': all_edges}
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
