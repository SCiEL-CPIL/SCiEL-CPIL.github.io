import time
import random
from collections import deque

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _ek(e):
    a, b = e
    return (a, b) if a < b else (b, a)

def _adj(n, edges):
    A = [set() for _ in range(n)]
    for u, v in edges:
        A[u].add(v)
        A[v].add(u)
    return A

def _is_biclique(l, r, adj, n):
    if not l or not r:
        return False
    if l & r:
        return False
    for u in l:
        for v in r:
            if v not in adj[u]:
                return False
    return True

def _grow_maximal_biclique(seed_l, seed_r, adj, n, edge_set):
    left = set(seed_l)
    right = set(seed_r)
    if not left or not right or (left & right):
        return None
    # Verify initial
    for u in left:
        for v in right:
            if _ek((u, v)) not in edge_set:
                return None
    changed = True
    while changed:
        changed = False
        # Add to left
        candidates_l = [w for w in range(n) if w not in left and w not in right]
        for w in candidates_l:
            if right.issubset(adj[w]):
                left.add(w)
                changed = True
        # Add to right
        candidates_r = [w for w in range(n) if w not in left and w not in right]
        for w in candidates_r:
            if left.issubset(adj[w]):
                right.add(w)
                changed = True
    return (frozenset(left), frozenset(right))

def _biclique_edges(l, r):
    return frozenset(_ek((u, v)) for u in l for v in r)

def _greedy_partition(edges, adj, n):
    remaining = set(_ek(e) for e in edges)
    partition = []
    while remaining:
        # Pick an edge
        u, v = next(iter(remaining))
        # Grow maximal biclique
        bc = _grow_maximal_biclique({u}, {v}, adj, n, remaining)
        if bc is None:
            # Fallback: single edge
            bc = (frozenset({u}), frozenset({v}))
        l, r = bc
        e_bc = _biclique_edges(l, r)
        # Ensure it only covers remaining edges
        # It might cover edges not in remaining? No, we grew using 'remaining' as edge_set?
        # _grow_maximal_biclique uses edge_set for verification.
        # So e_bc is subset of remaining.
        partition.append((l, r))
        remaining -= e_bc
    return partition

def _try_merge(partition, adj, n):
    improved = True
    while improved:
        improved = False
        new_partition = []
        i = 0
        while i < len(partition):
            l_i, r_i = partition[i]
            merged = False
            for j in range(i + 1, len(partition)):
                if merged: break
                l_j, r_j = partition[j]
                # Try merging (l_i, r_i) and (l_j, r_j)
                # Possible merges:
                # 1. L = L_i U L_j, R = R_i U R_j
                # 2. L = L_i U R_j, R = R_i U L_j
                # 3. L = R_i U L_j, R = L_i U R_j
                # 4. L = R_i U R_j, R = L_i U L_j
                
                candidates = [
                    (l_i | l_j, r_i | r_j),
                    (l_i | r_j, r_i | l_j),
                    (r_i | l_j, l_i | r_j),
                    (r_i | r_j, l_i | l_j)
                ]
                
                for nl, nr in candidates:
                    if _is_biclique(nl, nr, adj, n):
                        # Check if edges are disjoint?
                        # The partition property requires edge disjointness.
                        # If we merge, the new biclique covers edges of both.
                        # We must ensure that the union of edges is exactly the union of the two edge sets.
                        # Since they were a partition, they are disjoint.
                        # The new biclique must not introduce new edges that were not in the two.
                        # _is_biclique checks if all cross pairs are edges.
                        # It does NOT check if the biclique is "minimal" or if it contains extra edges.
                        # However, if (nl, nr) is a biclique in G, it contains ALL edges between nl and nr.
                        # If nl x nr contains an edge that was in a THIRD biclique, then the partition is broken.
                        # So we must check: Edges(nl, nr) == Edges(l_i, r_i) U Edges(l_j, r_j).
                        e_new = _biclique_edges(nl, nr)
                        e_old = _biclique_edges(l_i, r_i) | _biclique_edges(l_j, r_j)
                        if e_new == e_old:
                            # Valid merge
                            partition.pop(j)
                            partition[i] = (nl, nr)
                            merged = True
                            improved = True
                            break
            if not merged:
                new_partition.append(partition[i])
                i += 1
        partition = new_partition
    return partition

def _local_search(partition, adj, n, deadline):
    # Moves:
    # 1. Merge (done in _try_merge)
    # 2. Split: Take a biclique (L, R). Move a vertex v from L to R (if valid) or to a new biclique.
    #    Actually, simpler: Try to move an edge from one biclique to another?
    #    Or, try to "shrink" a biclique to make it mergeable.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # 1. Merge
        partition = _try_merge(partition, adj, n)
        if len(partition) == 0: break
        
        # 2. Vertex Swap / Move
        # For each biclique i, try moving a vertex from L to R or vice versa, or to another biclique.
        # This is complex. Let's try a simpler "Edge Move" or "Biclique Refinement".
        
        # Try to split a biclique into two smaller ones if it allows a better global arrangement?
        # Or, try to replace a biclique with two others.
        
        # Let's try: For each biclique B_i, pick a vertex v in L_i.
        # Remove v from L_i. The edges incident to v are now uncovered.
        # Try to form a new biclique with v and some neighbors.
        
        for i in range(len(partition)):
            if time.time() > deadline: break
            l_i, r_i = partition[i]
            # Try moving a vertex from L_i to R_i
            for v in list(l_i):
                nl = l_i - {v}
                nr = r_i | {v}
                if nl and nr and not (nl & nr):
                    if _is_biclique(nl, nr, adj, n):
                        # Check if this is a valid partition step?
                        # The edges of (nl, nr) must be the same as (l_i, r_i)?
                        # No, moving v changes the edge set.
                        # Edges(l_i, r_i) = Edges(nl, r_i) U Edges({v}, r_i).
                        # Edges(nl, nr) = Edges(nl, r_i) U Edges(nl, {v}).
                        # These are different. So we can't just swap v in place without changing the covered edges.
                        # This move is only valid if Edges(nl, {v}) is a subset of Edges({v}, r_i) ? No.
                        # This approach is tricky for partitions.
                        
                        # Better Move: **Biclique Exchange**
                        # Swap a vertex between two bicliques.
                        pass
        
        # 3. Try to find a biclique in the "residual" graph if we remove one? No.
        
        # Let's use a heuristic: **Random Restart with Local Search**
        # The current partition might be local optimum.
        # We will rely on the outer loop for restarts.
        
        # One more move: **Remove Redundant**
        # If a biclique's edges are covered by others, remove it.
        # But in a partition, edges are disjoint, so this is impossible unless a biclique is empty.
        
        # If no improvement, break.
        if not improved:
            break
            
    return partition

def construct():
    start_time = time.time()
    deadline = start_time + 55 # Safety margin
    
    n = P['n']
    edges = P['edges']
    adj = _adj(n, edges)
    all_edge_set = frozenset(_ek(e) for e in edges)
    
    best_partition = None
    best_count = float('inf')
    
    # Multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Randomize the order of edges for greedy
        shuffled_edges = edges[:]
        random.shuffle(shuffled_edges)
        
        # Greedy Partition
        partition = _greedy_partition(shuffled_edges, adj, n)
        
        # Local Search
        partition = _local_search(partition, adj, n, deadline)
        
        # Validation
        covered = set()
        valid = True
        for l, r in partition:
            if not l or not r or (l & r):
                valid = False
                break
            e = _biclique_edges(l, r)
            if not e.issubset(all_edge_set):
                valid = False
                break
            if e & covered:
                valid = False
                break
            covered |= e
        if covered != all_edge_set:
            valid = False
            
        if valid:
            count = len(partition)
            if count < best_count:
                best_count = count
                best_partition = [tuple(p) for p in partition]
                
        if best_count <= 10: # Good enough?
            break

    if best_partition is None:
        # Fallback: each edge is a biclique
        best_partition = [({u}, {v}) for u, v in edges]
        
    # Convert to output format
    result = []
    for l, r in best_partition:
        result.append([list(l), list(r)])
        
    return result
