"""Direct construction of a small 5-uniform hypergraph without Property B."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # safety margin under 150s

def is_2_colorable(n, edges):
    """
    Check if a 5-uniform hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    num_edges = len(edges)
    
    # Convert edges to bitmasks for fast membership tests
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # Precompute for each vertex, which edges contain it
    vertex_edges = [[] for _ in range(n)]
    for i, em in enumerate(edge_masks):
        for v in range(n):
            if em & (1 << v):
                vertex_edges[v].append(i)
    
    # State: for each edge, count of 0s and 1s assigned
    c0 = [0] * num_edges
    c1 = [0] * num_edges
    color = [-1] * n
    
    def propagate():
        """Returns False on conflict, True if no conflict."""
        changed = True
        while changed:
            changed = False
            for i in range(num_edges):
                if c0[i] == 5 or c1[i] == 5:
                    return False
                if c0[i] == 4 and c1[i] == 0:
                    # Need one more vertex to be color 1
                    mask = edge_masks[i]
                    for v in range(n):
                        if color[v] == -1 and (mask & (1 << v)):
                            color[v] = 1
                            c1[i] += 1
                            for other_e in vertex_edges[v]:
                                c1[other_e] += 1
                            changed = True
                            if c1[i] == 5:
                                return False
                            break
                elif c1[i] == 4 and c0[i] == 0:
                    # Need one more vertex to be color 0
                    mask = edge_masks[i]
                    for v in range(n):
                        if color[v] == -1 and (mask & (1 << v)):
                            color[v] = 0
                            c0[i] += 1
                            for other_e in vertex_edges[v]:
                                c0[other_e] += 1
                            changed = True
                            if c0[i] == 5:
                                return False
                            break
        return True
    
    def backtrack():
        if time.time() - _START_TIME > _TIME_LIMIT:
            return True  # Assume colorable to avoid timeout
        
        # Find unassigned vertex in the most constrained edge
        best_v = -1
        best_edge = -1
        best_unassigned = 6
        best_constraint = 0
        
        for i in range(num_edges):
            if c0[i] == 5 or c1[i] == 5:
                return False
            unassigned = 5 - c0[i] - c1[i]
            constraint = max(c0[i], c1[i])
            if unassigned < best_unassigned or (unassigned == best_unassigned and constraint > best_constraint):
                best_unassigned = unassigned
                best_constraint = constraint
                best_edge = i
                if unassigned == 0:
                    # All vertices in this edge are assigned, check if valid
                    if c0[i] == 0 or c1[i] == 0:
                        return False
        
        if best_edge == -1:
            # All edges satisfied
            for i in range(num_edges):
                if c0[i] == 5 or c1[i] == 5:
                    return False
            return True
        
        # Find an unassigned vertex in best_edge
        mask = edge_masks[best_edge]
        candidates = []
        for v in range(n):
            if color[v] == -1 and (mask & (1 << v)):
                candidates.append(v)
        
        if not candidates:
            return True  # Edge is satisfied
        
        # Try each candidate with both colors
        for v in candidates:
            # Try color 0
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
            
            # Try color 1
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


def is_2_colorable_bruteforce(n, edges):
    """Brute force for small n."""
    if not edges:
        return True
    
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    if n <= 14:
        for mask in range(1 << n):
            valid = True
            for em in edge_masks:
                if (mask & em) == 0 or (mask & em) == em:
                    valid = False
                    break
            if valid:
                return True
        return False
    else:
        return is_2_colorable(n, edges)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_edges = None
    best_n = 9
    best_m = 10**9
    
    # Try multiple n values, starting from known good ones
    for n in range(9, 17):
        if time.time() > deadline - 30:
            break
        
        all_possible = list(combinations(range(n), 5))
        num_total = len(all_possible)
        
        # Check if complete is non-2-colorable
        complete_edges = [list(e) for e in all_possible]
        if is_2_colorable_bruteforce(n, complete_edges):
            continue
        
        # Update best if smaller
        if len(complete_edges) < best_m:
            best_edges = complete_edges[:]
            best_n = n
            best_m = len(complete_edges)
        
        current_edges = complete_edges[:]
        current_m = len(current_edges)
        
        # Multiple greedy passes with random orderings
        for attempt in range(15):
            if time.time() > deadline - 30:
                break
            
            # Shuffle edge order for removal
            edge_indices = list(range(len(current_edges)))
            random.shuffle(edge_indices)
            
            removed_any = True
            while removed_any and time.time() < deadline - 30:
                removed_any = False
                for i in edge_indices:
                    if time.time() > deadline - 30:
                        break
                    if i >= len(current_edges):
                        continue
                    
                    # Try removing edge i
                    new_edges = current_edges[:i] + current_edges[i+1:]
                    if not is_2_colorable_bruteforce(n, new_edges):
                        current_edges = new_edges
                        current_m = len(current_edges)
                        if current_m < best_m:
                            best_edges = current_edges[:]
                            best_n = n
                            best_m = current_m
                        edge_indices = list(range(len(current_edges)))
                        random.shuffle(edge_indices)
                        removed_any = True
        
        # After greedy, try more aggressive removal with different orderings
        for attempt in range(10):
            if time.time() > deadline - 20:
                break
            
            edge_indices = list(range(len(current_edges)))
            random.shuffle(edge_indices)
            
            for i in edge_indices:
                if time.time() > deadline - 20:
                    break
                if i >= len(current_edges):
                    continue
                
                new_edges = current_edges[:i] + current_edges[i+1:]
                if not is_2_colorable_bruteforce(n, new_edges):
                    current_edges = new_edges
                    current_m = len(current_edges)
                    if current_m < best_m:
                        best_edges = current_edges[:]
                        best_n = n
                        best_m = current_m
    
    # If we didn't find anything better, use the complete 9-vertex hypergraph
    if best_edges is None:
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n
    
    # Final verification
    n = best_n
    edges = best_edges
    assert 5 <= n <= 26
    assert len(edges) <= 200
    assert len(set(tuple(sorted(e)) for e in edges)) == len(edges)
    for e in edges:
        assert len(e) == 5
        assert len(set(e)) == 5
        assert all(0 <= v < n for v in e)
    
    # Verify it's NOT 2-colorable
    if is_2_colorable_bruteforce(n, edges):
        # Fallback: return the complete graph
        n = 9
        edges = [list(e) for e in combinations(range(n), 5)]
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
