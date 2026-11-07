"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _validate(bicliques, adj, all_edges_set):
    """Check that bicliques form a valid partition."""
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        if set(left).intersection(right):
            return False
        for u in left:
            for v in right:
                if v not in adj[u]:
                    return False
                e = (min(u, v), max(u, v))
                if e in covered:
                    return False
                covered.add(e)
    if covered != all_edges_set:
        return False
    return True

def _expand_to_maximal(left, right, adj, n, remaining_edges=None):
    """Expand (left, right) to a maximal biclique. If remaining_edges is given,
    only expand using vertices that have at least one remaining edge to the opposite side."""
    left = set(left)
    right = set(right)
    changed = True
    while changed:
        changed = False
        for u in range(n):
            if u in left or u in right:
                continue
            # u can join left if all of right are neighbors of u
            if right.issubset(adj[u]):
                if remaining_edges is None:
                    left.add(u)
                    changed = True
                else:
                    # u must have at least one remaining edge to right
                    for v in right:
                        e = (min(u, v), max(u, v))
                        if e in remaining_edges:
                            left.add(u)
                            changed = True
                            break
        for v in range(n):
            if v in left or v in right:
                continue
            if left.issubset(adj[v]):
                if remaining_edges is None:
                    right.add(v)
                    changed = True
                else:
                    for u in left:
                        e = (min(u, v), max(u, v))
                        if e in remaining_edges:
                            right.add(v)
                            changed = True
                            break
    # Ensure disjointness
    right -= left
    left -= right
    if not left or not right:
        return None
    return (left, right)

def _biclique_edges(left, right, adj):
    """Return the set of edges covered by biclique (left, right)."""
    edges = set()
    for u in left:
        for v in right:
            if v in adj[u]:
                edges.add((min(u, v), max(u, v)))
    return edges

def _generate_maximal_pool(adj, n, all_edges_set, deadline):
    """Generate a pool of maximal bicliques by expanding every edge in both orientations,
    plus random starts."""
    pool = {}  # frozenset(left) | frozenset(right) -> (left, right, edges)
    
    # Expand every edge in both orientations
    for u, v in all_edges_set:
        if time.time() > deadline:
            break
        for l, r in [({u}, {v}), ({v}, {u})]:
            bc = _expand_to_maximal(l, r, adj, n, remaining_edges=None)
            if bc:
                key = (frozenset(bc[0]), frozenset(bc[1]))
                if key not in pool:
                    edges = _biclique_edges(bc[0], bc[1], adj)
                    if edges:
                        pool[key] = (bc[0], bc[1], edges)
    
    # Random starts for diversity
    random.seed(12345)
    for _ in range(500):
        if time.time() > deadline:
            break
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v and v in adj[u]:
            bc = _expand_to_maximal({u}, {v}, adj, n, remaining_edges=None)
            if bc:
                key = (frozenset(bc[0]), frozenset(bc[1]))
                if key not in pool:
                    edges = _biclique_edges(bc[0], bc[1], adj)
                    if edges:
                        pool[key] = (bc[0], bc[1], edges)
    
    return list(pool.values())

def _weighted_greedy_set_cover(pool, all_edges_set, deadline):
    """Solve set cover: repeatedly pick the biclique that covers the most uncovered edges."""
    remaining = set(all_edges_set)
    selected = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        best_idx = -1
        best_count = 0
        for i, (left, right, edges) in enumerate(pool):
            overlap = len(edges & remaining)
            if overlap > best_count:
                best_count = overlap
                best_idx = i
        
        if best_idx == -1 or best_count == 0:
            # Fallback: pick a single edge
            e = next(iter(remaining))
            selected.append(({e[0]}, {e[1]}))
            remaining.discard(e)
        else:
            left, right, edges = pool[best_idx]
            selected.append((left, right))
            remaining -= edges
    
    return selected

def _local_search_swap(selection, pool, all_edges_set, adj, deadline, n):
    """Try to improve the selection by swapping individual bicliques with others from the pool.
    A swap is valid if the new biclique covers all edges of the old one and doesn't overlap
    with other selected bicliques' edges."""
    selection = list(selection)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(selection)):
            if time.time() > deadline:
                break
            left_i, right_i = selection[i]
            edges_i = _biclique_edges(left_i, right_i, adj)
            
            # Compute edges covered by other selected bicliques
            other_edges = set()
            for j in range(len(selection)):
                if j == i:
                    continue
                left_j, right_j = selection[j]
                for u in left_j:
                    for v in right_j:
                        other_edges.add((min(u, v), max(u, v)))
            
            # Try to find a pool biclique that covers edges_i and doesn't overlap with other_edges
            # and is strictly "better" (e.g., covers more edges, or same edges but smaller biclique)
            for left_k, right_k, edges_k in pool:
                # Must cover all of edges_i
                if not edges_i.issubset(edges_k):
                    continue
                # Must not overlap with other selected bicliques
                if edges_k & other_edges:
                    continue
                # Must be valid (disjoint sides, all edges exist)
                if left_k & right_k:
                    continue
                valid = True
                for u in left_k:
                    for v in right_k:
                        if v not in adj[u]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Check if this swap is beneficial:
                # The new biclique covers edges_i (which it must) and possibly more.
                # But we must ensure that the "extra" edges it covers are not already covered by others.
                # Actually, if edges_k & other_edges is empty, then all edges in edges_k are uncovered
                # except those in edges_i (which are being replaced). So edges_k - edges_i are newly covered.
                # This is always beneficial or neutral in terms of coverage, but we want to reduce count.
                # Wait, a swap keeps the count the same. To reduce count, we need a "merge" or "absorb" move.
                # However, a swap might enable a future merge. For now, let's stick to swaps that
                # at least don't increase the count, and hope for local minima improvement.
                # Actually, to reduce the count, we need a move that replaces 1 biclique with 1 biclique
                # that allows another biclique to be removed. That's complex.
                # Let's try a different local search: try to remove a biclique if its edges are covered by others.
                # But in a partition, no edge is covered by others. So removal is not possible.
                # The only way to reduce count is to merge two bicliques into one.
                # So the "swap" move is not directly reducing count.
                # Let's implement a "merge" move instead: try to merge two bicliques in the selection
                # into one biclique from the pool.
                pass
            
            # Let's try a merge move: for each pair (i, j), check if there is a pool biclique
            # that covers edges_i ∪ edges_j and doesn't overlap with others.
            for j in range(i + 1, len(selection)):
                if time.time() > deadline:
                    break
                left_j, right_j = selection[j]
                edges_j = _biclique_edges(left_j, right_j, adj)
                target_edges = edges_i | edges_j
                
                other_edges = set()
                for k in range(len(selection)):
                    if k == i or k == j:
                        continue
                    left_k, right_k = selection[k]
                    for u in left_k:
                        for v in right_k:
                            other_edges.add((min(u, v), max(u, v)))
                
                for left_k, right_k, edges_k in pool:
                    # Must cover target_edges
                    if not target_edges.issubset(edges_k):
                        continue
                    # Must not overlap with other selected bicliques
                    if edges_k & other_edges:
                        continue
                    # Must be valid
                    if left_k & right_k:
                        continue
                    valid = True
                    for u in left_k:
                        for v in right_k:
                            if v not in adj[u]:
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    # Perform the merge
                    new_selection = []
                    for k in range(len(selection)):
                        if k != i and k != j:
                            new_selection.append(selection[k])
                    new_selection.append((left_k, right_k))
                    
                    # Validate the new selection
                    if _validate(new_selection, adj, all_edges_set):
                        selection = new_selection
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return selection

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    n = P['n']
    adj = _build_adjacency()
    all_edges_set = set((min(u, v), max(u, v)) for u, v in P['edges'])
    
    # Step 1: Generate a pool of maximal bicliques
    pool = _generate_maximal_pool(adj, n, all_edges_set, deadline)
    
    # Step 2: Solve set cover using weighted greedy
    selection = _weighted_greedy_set_cover(pool, all_edges_set, deadline)
    
    # Step 3: Local search to improve the selection
    if time.time() < deadline:
        improved_selection = _local_search_swap(selection, pool, all_edges_set, adj, deadline, n)
        if _validate(improved_selection, adj, all_edges_set):
            selection = improved_selection
    
    # Fallback if something went wrong
    if not selection or not _validate(selection, adj, all_edges_set):
        # Simple fallback: each edge as its own biclique
        selection = [({u}, {v}) for u, v in all_edges_set]
    
    # Convert to proper types
    result = []
    for left, right in selection:
        result.append([list(left), list(right)])
    
    return result
# EVOLVE-BLOCK-END
