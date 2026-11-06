# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Deterministic seed for reproducibility
random.seed(42)

def _is_2_colorable(edges, n):
    """Check if hypergraph is 2-colorable (properly: no monochromatic edge).
    Uses backtracking with unit propagation. Returns True if 2-colorable."""
    # Build incidence: for each vertex, list of edges containing it
    v_edges = [[] for _ in range(n)]
    for ei, e in enumerate(edges):
        for v in e:
            v_edges[v].append(ei)
    
    # Edge adjacency: for each edge, list of other edges it shares a vertex with
    e_adj = [[] for _ in range(len(edges))]
    for ei in range(len(edges)):
        seen = set(edges[ei])
        for ej in range(len(edges)):
            if ej == ei:
                continue
            for v in edges[ej]:
                if v in seen:
                    e_adj[ei].append(ej)
                    break
    
    # For unit propagation: for each edge, if all but one vertex are same color, force the last
    m = len(edges)
    # state: -1 uncolored, 0 color A, 1 color B
    color = [-1] * n
    
    # Counters: for each edge, count of A-colored and B-colored vertices
    cntA = [0] * m
    cntB = [0] * m
    
    def propagate(color, cntA, cntB, queue):
        # Process queue of (edge, forced_vertex, forced_color) or just edges to recheck
        # We'll use a worklist of vertices whose color changed
        while queue:
            v = queue.pop()
            cv = color[v]
            if cv == -1:
                continue
            for ei in v_edges[v]:
                # Check if this edge is now forced
                if cntB[ei] == 0:
                    # All colored so far are A; if 4 are A, force last to B
                    if cntA[ei] == 4:
                        # Find the uncolored vertex in this edge
                        for u in edges[ei]:
                            if color[u] == -1:
                                if cv == 0:
                                    # would be monochromatic A -> contradiction
                                    return False
                                color[u] = 1
                                cntB[ei] += 1
                                queue.append(u)
                                break
                    elif cntA[ei] < 4:
                        pass
                if cntA[ei] == 0:
                    if cntB[ei] == 4:
                        for u in edges[ei]:
                            if color[u] == -1:
                                if cv == 1:
                                    return False
                                color[u] = 0
                                cntA[ei] += 1
                                queue.append(u)
                                break
        return True
    
    def backtrack(color, cntA, cntB):
        # Find uncolored vertex with smallest domain (MRV heuristic)
        best_v = -1
        best_dom = 5
        for v in range(n):
            if color[v] == -1:
                # domain: which colors are still possible
                dom = 0
                for c in (0, 1):
                    ok = True
                    for ei in v_edges[v]:
                        if c == 0 and cntB[ei] == 4:
                            ok = False
                            break
                        if c == 1 and cntA[ei] == 4:
                            ok = False
                            break
                    if ok:
                        dom += 1
                if dom < best_dom:
                    best_dom = dom
                    best_v = v
                    if dom <= 1:
                        break
        if best_v == -1:
            return True
        if best_dom == 0:
            return False
        # Try each color
        for c in (0, 1):
            # Check if c is allowed
            ok = True
            for ei in v_edges[best_v]:
                if c == 0 and cntB[ei] == 4:
                    ok = False
                    break
                if c == 1 and cntA[ei] == 4:
                    ok = False
                    break
            if not ok:
                continue
            # Assign and propagate
            color[best_v] = c
            cntA2 = cntA[:]
            cntB2 = cntB[:]
            if c == 0:
                cntA2[ei] += 1
            else:
                cntB2[ei] += 1
            # Recompute properly
            cntA2 = cntA[:]
            cntB2 = cntB[:]
            for ei in v_edges[best_v]:
                if c == 0:
                    cntA2[ei] += 1
                else:
                    cntB2[ei] += 1
            queue = [best_v]
            if propagate(color, cntA2, cntB2, queue):
                if backtrack(color, cntA2, cntB2):
                    return True
            # Undo
            color[best_v] = -1
        return False
    
    # Initialize
    return backtrack(color, cntA, cntB)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start from a known non-2-colorable hypergraph: K9^5 (126 edges)
    # But we want to minimize edges. Start from a smaller structure.
    
    # Strategy: Start from K9^5 and try to remove/swap edges while maintaining non-2-colorability.
    # Better: Start from a random set and add edges until non-2-colorable, then minimize.
    
    # Approach: 
    # 1. Start with a base non-2-colorable hypergraph (e.g., K9^5 or a known construction)
    # 2. Use local search to minimize edges
    
    # Let's start with K9^5 on 9 vertices
    n = 9
    base_edges = [list(e) for e in combinations(range(n), 5)]
    
    # Verify it's non-2-colorable
    if _is_2_colorable(base_edges, n):
        # Shouldn't happen, but handle it
        return {'n': n, 'edges': base_edges}
    
    best_edges = base_edges[:]
    best_count = len(best_edges)
    
    # Local search: try to remove edges one at a time, keep if still non-2-colorable
    # Then try swaps: remove one edge, add another
    
    def is_non_2_colorable(edges, n):
        return not _is_2_colorable(edges, n)
    
    # Phase 1: Greedy removal
    changed = True
    while changed and time.time() < deadline:
        changed = False
        current = best_edges[:]
        for i in range(len(current)):
            if time.time() > deadline:
                break
            trial = current[:i] + current[i+1:]
            if is_non_2_colorable(trial, n):
                best_edges = trial
                best_count = len(best_edges)
                changed = True
                break
    
    # Phase 2: Local search with swaps
    # Try removing edge i and adding a random edge not in the set
    max_swaps = 5000
    for swap_iter in range(max_swaps):
        if time.time() > deadline:
            break
        if len(best_edges) <= 29:
            break
        
        # Try to remove a random edge and add a different random edge
        i = random.randint(0, len(best_edges) - 1)
        # Generate a random 5-element subset of range(n)
        new_edge = list(random.sample(range(n), 5))
        new_edge_set = frozenset(new_edge)
        
        # Skip if same edge
        if frozenset(best_edges[i]) == new_edge_set:
            continue
        # Skip if new edge already in set
        if new_edge_set in [frozenset(e) for e in best_edges]:
            continue
        
        trial = best_edges[:i] + best_edges[i+1:] + [new_edge]
        if is_non_2_colorable(trial, n):
            best_edges = trial
            best_count = len(best_edges)
    
    # Phase 3: More aggressive removal after swaps
    changed = True
    while changed and time.time() < deadline:
        changed = False
        current = best_edges[:]
        random.shuffle(current)
        for i in range(len(current)):
            if time.time() > deadline:
                break
            trial = current[:i] + current[i+1:]
            if is_non_2_colorable(trial, n):
                best_edges = trial
                best_count = len(best_edges)
                changed = True
                break
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
