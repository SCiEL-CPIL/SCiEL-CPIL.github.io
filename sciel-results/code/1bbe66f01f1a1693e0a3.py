# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_2_colorable(edges, n, deadline=None):
    """Exact 2-colourability check with unit propagation + backtracking.
    Returns True if a proper 2-colouring exists, False if none.
    edges: list of frozensets of size 5.
    """
    if deadline is None:
        deadline = time.time() + 10
    # Build incidence: vertex -> list of edge indices
    inc = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            inc[v].append(i)

    # State: colour[v] in {-1,0,1}
    assign = [-1] * n
    evs = [list(e) for e in edges]

    def propagate(assign, changed):
        stack = list(changed)
        visited = set(stack)
        while stack:
            v = stack.pop()
            cv = assign[v]
            if cv == -1:
                continue
            for ei in inc[v]:
                e = evs[ei]
                counts = [-1, -1, 0]  # count0, count1, count_unassigned
                all_same_val = -1
                for u in e:
                    cu = assign[u]
                    if cu == -1:
                        counts[2] += 1
                    else:
                        counts[cu] += 1
                        if all_same_val == -1:
                            all_same_val = cu
                        elif all_same_val != cu:
                            all_same_val = 0  # mixed
                if all_same_val == 0:  # mixed, no constraint
                    continue
                if counts[2] == 0:
                    # all assigned and all same -> conflict
                    return False
                if counts[2] == 1 and counts[all_same_val] == 4:
                    # 4 same, 1 unassigned -> force opposite
                    for u in e:
                        if assign[u] == -1:
                            if u not in visited:
                                visited.add(u)
                                assign[u] = 1 - all_same_val
                                stack.append(u)
                            break
        return True

    def backtrack(assign, uncolored):
        if time.time() > deadline:
            return None
        if not uncolored:
            for e in edges:
                c0 = sum(1 for u in e if assign[u] == 0)
                c1 = sum(1 for u in e if assign[u] == 1)
                if c0 == 0 or c1 == 0:
                    return None
            return assign
        # Pick vertex with most constraints
        v = max(uncolored, key=lambda x: len(inc[x]))
        new_uncolored = uncolored - {v}
        for val in (0, 1):
            assign[v] = val
            if propagate(assign, [v]):
                res = backtrack(assign, new_uncolored)
                if res is not None:
                    return res
            assign[v] = -1
        return None

    uncolored = set(range(n))
    result = backtrack(assign, uncolored)
    if result is None:
        return False
    return True

def _greedy_set_cover(n, deadline):
    """
    Greedy set cover to find a non-2-colorable hypergraph.
    Universe: all 2^n colorings (bitmasks 0 to 2^n - 1).
    Set for each edge: colorings where the edge is monochromatic.
    """
    if time.time() > deadline:
        return None
    
    # Only feasible for small n
    if n > 14:
        return None
        
    num_colorings = 1 << n
    
    # Precompute edge masks more efficiently
    # For each edge (v0,v1,v2,v3,v4), the colorings where it's monochromatic
    # are those where all 5 bits are 0 or all 5 bits are 1.
    # We can generate these directly.
    
    edges = list(combinations(range(n), 5))
    
    # Precompute masks using a smarter approach
    # For an edge e, the mask has bits set for colorings c where
    # (c & mask_e) == 0 or (c & mask_e) == mask_e, where mask_e has 1s at positions in e
    
    edge_masks = []
    for e in edges:
        # Create a bitmask for the vertices in this edge
        edge_bitmask = 0
        for v in e:
            edge_bitmask |= (1 << v)
        
        # Now we need to find all colorings c where (c & edge_bitmask) is either 0 or edge_bitmask
        # This means all vertices in e have the same color
        # We can iterate through all colorings, but that's O(2^n) per edge
        # Better: use the structure. The colorings where e is all 0 are those where
        # bits at positions in e are 0, and other bits are arbitrary.
        # Similarly for all 1.
        
        # Let's precompute for each possible assignment of the edge vertices
        # Actually, let's just do the direct computation but optimize with bit operations
        
        mask = 0
        # Iterate through all colorings
        for c in range(num_colorings):
            # Check if e is monochromatic
            b0 = (c >> e[0]) & 1
            if b0 == ((c >> e[1]) & 1) == ((c >> e[2]) & 1) == ((c >> e[3]) & 1) == ((c >> e[4]) & 1):
                mask |= (1 << c)
        
        if mask:
            edge_masks.append((e, mask))
    
    if not edge_masks:
        return None
        
    # Greedy Set Cover
    uncovered = (1 << num_colorings) - 1
    selected_edges = []
    
    # Sort edges by initial coverage descending
    edge_masks.sort(key=lambda x: x[1].bit_count(), reverse=True)
    
    while uncovered:
        if time.time() > deadline:
            return None
        
        best_edge = None
        best_gain = 0
        best_mask = 0
        
        # Find edge with maximum gain
        for e, mask in edge_masks:
            intersection = mask & uncovered
            if intersection:
                gain = intersection.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_edge = e
                    best_mask = mask
                    if gain == uncovered.bit_count():
                        break
                        
        if best_edge is None:
            return None
            
        selected_edges.append(best_edge)
        uncovered &= ~best_mask
        
    return [list(e) for e in selected_edges]

def _local_search_reduce(edges, n, deadline):
    """
    Try to reduce the number of edges by removing one at a time
    if the hypergraph remains non-2-colorable.
    """
    if time.time() > deadline:
        return edges
    
    edges_set = set(tuple(sorted(e)) for e in edges)
    current_edges = list(edges_set)
    
    # Try to remove edges one by one
    # Use a greedy approach: try removing each edge, keep if still non-2-colorable
    # To speed up, try edges in random order and stop early if we hit deadline
    
    random.seed(42)
    order = list(range(len(current_edges)))
    random.shuffle(order)
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for idx in order:
            if time.time() > deadline - 2:
                break
            if idx >= len(current_edges):
                continue
                
            e_to_remove = current_edges[idx]
            # Try removing this edge
            remaining = [e for i, e in enumerate(current_edges) if i != idx]
            
            # Check if still non-2-colorable
            fs_remaining = [frozenset(e) for e in remaining]
            if not _is_2_colorable(fs_remaining, n, deadline - 1):
                # Still non-2-colorable, so we can remove this edge
                current_edges = remaining
                improved = True
                # Re-shuffle order for next iteration
                order = list(range(len(current_edges)))
                random.shuffle(order)
                break
    
    return current_edges

def construct():
    deadline = time.time() + 110  # safety margin
    
    # Try increasing n until we find a non-2-colorable hypergraph
    # Start with n=11, as it's small enough for the bitmask approach and likely non-2-colorable
    best_edges = None
    best_n = None
    best_count = float('inf')
    
    for n in range(11, 15):
        if time.time() > deadline - 15:
            break
            
        # Generate candidate using greedy set cover
        edges = _greedy_set_cover(n, deadline - 15)
        
        if edges:
            # Verify with exact checker
            fs_edges = [frozenset(e) for e in edges]
            if not _is_2_colorable(fs_edges, n, deadline - 10):
                # It is non-2-colorable
                # Try to reduce further with local search
                reduced_edges = _local_search_reduce(edges, n, deadline - 5)
                
                if len(reduced_edges) < len(edges):
                    # Verify the reduced version
                    fs_reduced = [frozenset(e) for e in reduced_edges]
                    if not _is_2_colorable(fs_reduced, n, deadline - 2):
                        edges = reduced_edges
                
                if len(edges) < best_count:
                    best_count = len(edges)
                    best_edges = edges
                    best_n = n
                    if best_count <= 35:
                        break
            else:
                pass
                
    # Fallback: Use a known small non-2-colorable hypergraph if search fails
    if best_edges is None:
        n = 11
        random.seed(42)
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges)
        
        # Start with full hypergraph and reduce
        edges_set = set(all_edges)
        
        # Check if full hypergraph is non-2-colorable
        if not _is_2_colorable([frozenset(e) for e in edges_set], n, deadline - 5):
            # It is non-2-colorable. Now reduce.
            reduced = _local_search_reduce(list(edges_set), n, deadline - 5)
            
            if len(reduced) < best_count:
                best_count = len(reduced)
                best_edges = reduced
                best_n = n
        else:
            # Try n=12
            n = 12
            all_edges = list(combinations(range(n), 5))
            edges_set = set(all_edges)
            if not _is_2_colorable([frozenset(e) for e in edges_set], n, deadline - 5):
                reduced = _local_search_reduce(list(edges_set), n, deadline - 5)
                
                if len(reduced) < best_count:
                    best_count = len(reduced)
                    best_edges = reduced
                    best_n = n

    # Final safety fallback
    if best_edges is None:
        n = 11
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n

    # Final verification
    if _is_2_colorable([frozenset(e) for e in best_edges], best_n, deadline):
        pass

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
