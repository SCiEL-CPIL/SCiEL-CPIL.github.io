import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Checks if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if there exists a 2-coloring with no monochromatic edge.
    """
    if not edges:
        return True
    
    m = len(edges)
    # Precompute: for each vertex, which edges contain it
    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vert_edges[v].append(i)
    
    # State:
    # assign: -1 unassigned, 0 or 1 assigned
    # c0, c1: count of 0s and 1s in each edge
    # cu: count of unassigned in each edge

    def propagate(assign, c0, c1, cu):
        # Returns False if contradiction, True otherwise.
        # Modifies 'assign', c0, c1, cu in place.
        changed = True
        while changed:
            changed = False
            for i in range(m):
                if c0[i] == 5 or c1[i] == 5:
                    return False
                if cu[i] == 0:
                    continue
                if c0[i] == 4:
                    # Must have at least one 1. If cu[i]==1, the last one is 1.
                    if cu[i] == 1:
                        for v in edges[i]:
                            if assign[v] == -1:
                                assign[v] = 1
                                cu[i] -= 1
                                c1[i] += 1
                                for j in vert_edges[v]:
                                    if j != i:
                                        cu[j] -= 1
                                        c1[j] += 1
                                changed = True
                                break
                elif c1[i] == 4:
                    # Must have at least one 0. If cu[i]==1, the last one is 0.
                    if cu[i] == 1:
                        for v in edges[i]:
                            if assign[v] == -1:
                                assign[v] = 0
                                cu[i] -= 1
                                c0[i] += 1
                                for j in vert_edges[v]:
                                    if j != i:
                                        cu[j] -= 1
                                        c0[j] += 1
                                changed = True
                                break
        return True

    def solve(assign, c0, c1, cu):
        # Find next unassigned vertex
        v = -1
        best_crit = -1
        # Heuristic: pick vertex in an edge with most assigned vertices
        for i in range(m):
            if cu[i] == 0:
                continue
            # Criticality: prefer edges close to contradiction
            # If c0[i]+c1[i] is high, it's critical
            crit = c0[i] + c1[i]
            if crit > best_crit:
                best_crit = crit
                # Pick first unassigned vertex in this edge
                for vv in edges[i]:
                    if assign[vv] == -1:
                        v = vv
                        break
        if v == -1:
            return True
        
        # Try 0
        na0 = assign[:]
        nc00 = c0[:]
        nc10 = c1[:]
        ncu0 = cu[:]
        na0[v] = 0
        for j in vert_edges[v]:
            ncu0[j] -= 1
            nc00[j] += 1
        if propagate(na0, nc00, nc10, ncu0):
            if solve(na0, nc00, nc10, ncu0):
                return True
        
        # Try 1
        na1 = assign[:]
        nc01 = c0[:]
        nc11 = c1[:]
        ncu1 = cu[:]
        na1[v] = 1
        for j in vert_edges[v]:
            ncu1[j] -= 1
            nc11[j] += 1
        if propagate(na1, nc01, nc11, ncu1):
            if solve(na1, nc01, nc11, ncu1):
                return True
        
        return False

    assign = [-1] * n
    c0 = [0] * m
    c1 = [0] * m
    cu = [5] * m
    
    if not propagate(assign, c0, c1, cu):
        return False
    
    return solve(assign, c0, c1, cu)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    # Known small non-2-colorable 5-uniform hypergraphs.
    # We will try to find one with fewer edges.
    # Start with a known valid one if we have one, or generate a seed.
    
    # A simple seed: Complete hypergraph on 9 vertices has 126 edges.
    # We can try to prune it.
    
    def greedy_prune(n, edges, time_limit):
        """
        Try to remove edges one by one while maintaining non-2-colorability.
        """
        current_edges = set(tuple(sorted(e)) for e in edges)
        improved = True
        while improved:
            improved = False
            if time.time() > time_limit:
                break
            # Try removing each edge
            for e in list(current_edges):
                if time.time() > time_limit:
                    break
                test_edges = current_edges - {e}
                if not is_2_colorable(n, [list(x) for x in test_edges]):
                    current_edges = test_edges
                    improved = True
                    break
        return [list(e) for e in current_edges]

    best_edges = None
    best_n = 0
    
    # Strategy 1: Start with a dense graph on small n and prune.
    # n=9 is the smallest possible.
    # Let's try n=9 first.
    # The complete hypergraph K_9^5 has 126 edges.
    
    n = 9
    all_edges = list(combinations(range(n), 5))
    current_edges = all_edges[:]
    
    # Prune
    pruned = greedy_prune(n, current_edges, deadline - 10)
    if len(pruned) < 100: # Just a sanity check
        best_edges = pruned
        best_n = n
    
    # Strategy 2: Try other n values with random starts and pruning.
    # It's possible that a sparse graph on a larger n has fewer edges.
    # But usually for fixed uniformity, minimal non-colorable graphs are on small n.
    
    # Let's also try a randomized approach:
    # Start with a random subset, check if non-colorable, if not add edges, then prune.
    # Or start with a known structure.
    
    # Since we have time, let's try to improve the pruned graph from n=9 further with a different heuristic.
    # Or try n=10.
    
    for test_n in [10, 11]:
        if time.time() > deadline - 10:
            break
        
        # Start with a random subset of edges of size ~50-60
        # and try to make it non-colorable by adding edges, then prune.
        # This might find a smaller solution.
        
        all_possible = list(combinations(range(test_n), 5))
        random.shuffle(all_possible)
        
        # Start with a random subset
        k = 60
        start_edges = [list(e) for e in all_possible[:k]]
        
        # Check if it's already non-colorable
        if not is_2_colorable(test_n, start_edges):
            # Great, prune it
            pruned = greedy_prune(test_n, start_edges, deadline - 10)
            if best_edges is None or len(pruned) < len(best_edges):
                best_edges = pruned
                best_n = test_n
        else:
            # It is colorable. Add random edges until it becomes non-colorable.
            current_set = set(tuple(e) for e in start_edges)
            # Add edges from the remaining pool
            remaining = [e for e in all_possible[k:]]
            random.shuffle(remaining)
            
            for e in remaining:
                if time.time() > deadline - 10:
                    break
                start_edges.append(list(e))
                current_set.add(tuple(e))
                # Check periodically to save time? Or just check when we think it might be close.
                # Actually, checking every step is expensive.
                # Let's check every few edges.
                if len(start_edges) % 5 == 0:
                    if not is_2_colorable(test_n, start_edges):
                        break
            
            if not is_2_colorable(test_n, start_edges):
                pruned = greedy_prune(test_n, start_edges, deadline - 10)
                if best_edges is None or len(pruned) < len(best_edges):
                    best_edges = pruned
                    best_n = test_n
    
    # If we still don't have a good one, fall back to the pruned n=9
    if best_edges is None:
        best_edges = pruned
        best_n = 9
        
    # Final verification
    if is_2_colorable(best_n, best_edges):
        # Fallback to a known valid one (complete graph on 9 vertices)
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_n = 9
    
    return {'n': best_n, 'edges': best_edges}
