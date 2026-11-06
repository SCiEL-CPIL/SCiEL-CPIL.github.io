import time
import random
from itertools import combinations
import sys

def is_2_colorable(n, edges):
    """
    Check if a 5-uniform hypergraph is 2-colorable.
    Returns True if 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
    
    # Convert edges to bitmasks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # Precompute vertex to edges mapping
    vertex_edges = [[] for _ in range(n)]
    for i, em in enumerate(edge_masks):
        # Iterate bits
        m = em
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            vertex_edges[v].append(i)
            m ^= lsb

    color = [-1] * n
    # c0[i] = number of vertices in edge i colored 0
    # c1[i] = number of vertices in edge i colored 1
    c0 = [0] * len(edges)
    c1 = [0] * len(edges)

    def propagate():
        """
        Perform unit propagation.
        Returns False if contradiction, True otherwise.
        """
        changed = True
        while changed:
            changed = False
            for i in range(len(edges)):
                if c0[i] == 5 or c1[i] == 5:
                    return False
                # If 4 are 0, the 5th must be 1
                if c0[i] == 4 and c1[i] == 0:
                    # Find the uncolored vertex in this edge
                    em = edge_masks[i]
                    # Iterate bits of em
                    m = em
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        if color[v] == -1:
                            # Assign 1
                            color[v] = 1
                            c1[i] += 1
                            for other_e in vertex_edges[v]:
                                c1[other_e] += 1
                            changed = True
                            if c1[i] == 5:
                                return False
                            break
                        m ^= lsb
                # If 4 are 1, the 5th must be 0
                elif c1[i] == 4 and c0[i] == 0:
                    em = edge_masks[i]
                    m = em
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        if color[v] == -1:
                            # Assign 0
                            color[v] = 0
                            c0[i] += 1
                            for other_e in vertex_edges[v]:
                                c0[other_e] += 1
                            changed = True
                            if c0[i] == 5:
                                return False
                            break
                        m ^= lsb
        return True

    def backtrack():
        # Check for immediate contradiction
        if not propagate():
            return False
            
        # Select an edge with the most constrained uncolored vertices
        # Heuristic: minimize uncolored, maximize colored count (MRV)
        best_edge = -1
        min_unassigned = 6
        max_colored = -1
        
        for i in range(len(edges)):
            if c0[i] == 5 or c1[i] == 5:
                return False
            unassigned = 5 - c0[i] - c1[i]
            colored = c0[i] + c1[i]
            if unassigned < min_unassigned or (unassigned == min_unassigned and colored > max_colored):
                min_unassigned = unassigned
                max_colored = colored
                best_edge = i
        
        if best_edge == -1:
            # All edges are satisfied (no unassigned vertices in any edge? No, just no edges left to constrain?)
            # If we reach here, all edges have been checked for propagation.
            # If no edge has unassigned vertices, we are done?
            # Actually, if an edge has 0,1,2,3 unassigned, it's not forced yet.
            # But if we pick the "best" edge, and it has 0 unassigned, it means it's fully colored and not monochromatic.
            # If all edges are fully colored and valid, we found a coloring.
            return True
        
        # Get uncolored vertices in best_edge
        em = edge_masks[best_edge]
        candidates = []
        m = em
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if color[v] == -1:
                candidates.append(v)
            m ^= lsb
        
        if not candidates:
            return True

        # Try assigning 0 to the first candidate
        # We can just pick the first candidate to reduce branching factor
        # Or try all? Branching on one vertex is standard.
        v = candidates[0]
        
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
        if ok and backtrack():
            return True
        # Restore
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
        if ok and backtrack():
            return True
        
        return False

    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_edges = None
    best_n = 9
    best_m = 10**9
    
    # Prioritize n values where minimal constructions are likely
    # Known: 51 edges.
    # Try n from 9 to 15
    for n in [11, 12, 10, 9, 13, 14]:
        if time.time() > deadline - 10:
            break
        
        all_possible = list(combinations(range(n), 5))
        
        # Check if complete hypergraph is non-2-colorable
        # For n>=5, K_n^5 is 2-colorable if n is small?
        # Actually, we just need to find *some* non-2-colorable subgraph.
        # Start with the complete graph. If it's 2-colorable, skip this n?
        # K_5^5 has 1 edge, 2-colorable.
        # K_6^5 has 6 edges.
        # It is known that m(5) >= 29.
        # So for small n, the complete graph might be 2-colorable.
        # We should check.
        if is_2_colorable(n, all_possible):
            continue
        
        # Greedy removal with multiple restarts
        # Increase attempts to explore more of the solution space
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Restart from complete graph
            curr = [list(e) for e in all_possible]
            
            edge_indices = list(range(len(curr)))
            random.shuffle(edge_indices)
            
            removed_any = True
            while removed_any and time.time() < deadline - 5:
                removed_any = False
                for i in edge_indices:
                    if time.time() > deadline - 5:
                        break
                    if i >= len(curr):
                        continue
                    
                    # Try removing edge i
                    new_edges = curr[:i] + curr[i+1:]
                    if not is_2_colorable(n, new_edges):
                        curr = new_edges
                        # Re-shuffle to break symmetry
                        edge_indices = list(range(len(curr)))
                        random.shuffle(edge_indices)
                        removed_any = True
                        break
            
            if len(curr) < best_m:
                best_m = len(curr)
                best_edges = curr[:]
                best_n = n
                if best_m <= 51:
                    # If we hit the known bound, we might stop early or try to beat it
                    # But 51 is the best known. Beating it is unlikely but possible.
                    # We will continue to see if we can find < 51.
                    pass
        
        if best_m <= 50:
            break

    if best_edges is None:
        # Fallback: find any non-2-colorable hypergraph
        # Use n=11 complete graph and remove edges until non-2-colorable?
        # Or just return a known valid one if we have one hardcoded?
        # Better to construct one quickly.
        n = 11
        all_possible = list(combinations(range(n), 5))
        # Randomly pick a subset?
        # Just use the complete graph if nothing else works, though it has many edges.
        # The evaluator minimizes edges, so we want few.
        # If search fails, we return the best found. If none found, we return complete graph of small n?
        # K_11^5 has 462 edges.
        # Let's try a quick greedy on n=11 if we haven't found anything.
        if time.time() < deadline - 5:
             curr = [list(e) for e in all_possible]
             edge_indices = list(range(len(curr)))
             random.shuffle(edge_indices)
             removed_any = True
             while removed_any and time.time() < deadline - 5:
                 removed_any = False
                 for i in edge_indices:
                     if time.time() > deadline - 5:
                         break
                     if i >= len(curr):
                         continue
                     new_edges = curr[:i] + curr[i+1:]
                     if not is_2_colorable(n, new_edges):
                         curr = new_edges
                         edge_indices = list(range(len(curr)))
                         random.shuffle(edge_indices)
                         removed_any = True
                         break
             if len(curr) < best_m:
                 best_m = len(curr)
                 best_edges = curr[:]
                 best_n = n
                 
    if best_edges is None:
        # Absolute fallback: 51-edge construction if we can't find better?
        # We don't have it hardcoded.
        # Return a valid hypergraph with more edges.
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n

    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
