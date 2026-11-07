"""Biclique partition with greedy + local search."""
import time
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_biclique(left, right, adj):
    """Check if all cross pairs are edges."""
    for u in left:
        for v in right:
            if v not in adj[u]:
                return False
    return True

def _biclique_edges(left, right):
    """Return set of edge tuples for a biclique."""
    return set((min(u,v), max(u,v)) for u in left for v in right)

def _greedy_partition(adj, deadline):
    """Greedily find large bicliques to cover all edges."""
    n = P['n']
    all_edges = set()
    for u, v in P['edges']:
        all_edges.add((min(u,v), max(u,v)))
    
    remaining = set(all_edges)
    bicliques = []
    
    # Vertex degrees for heuristic ordering
    deg = [len(adj[i]) for i in range(n)]
    
    while remaining and time.time() < deadline:
        # Pick the edge with max combined degree (heuristic for larger bicliques)
        best_edge = None
        best_score = -1
        for e in remaining:
            u, v = e
            score = deg[u] + deg[v]
            if score > best_score:
                best_score = score
                best_edge = e
        
        if best_edge is None:
            break
        
        u, v = best_edge
        # Start biclique with {u} on left, {v} on right
        left = {u}
        right = {v}
        
        # Expand: try to add vertices to left or right
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Try adding to left: vertex w can join left if w connects to all of right
            for w in range(n):
                if w in left or w in right:
                    continue
                if all(v2 in adj[w] for v2 in right):
                    # Check w has at least one edge to right in remaining
                    if any((min(w,v2), max(w,v2)) in remaining for v2 in right):
                        left.add(w)
                        changed = True
                        break
            if changed:
                continue
            # Try adding to right: vertex w can join right if w connects to all of left
            for w in range(n):
                if w in left or w in right:
                    continue
                if all(u2 in adj[w] for u2 in left):
                    if any((min(u2,w), max(u2,w)) in remaining for u2 in left):
                        right.add(w)
                        changed = True
                        break
        
        # Only keep edges that are in remaining
        bl = sorted(left)
        br = sorted(right)
        # Filter to only include edges that are actually in remaining
        valid_edges = set()
        for a in bl:
            for b in br:
                e = (min(a,b), max(a,b))
                if e in remaining:
                    valid_edges.add(e)
        
        if valid_edges:
            bicliques.append([bl, br])
            remaining -= valid_edges
    
    # Cover any leftover edges individually
    for e in remaining:
        u, v = e
        bicliques.append([[u], [v]])
    
    return bicliques

def _try_merge(bicliques, adj, deadline):
    """Try to merge pairs of bicliques. Return new list if improved."""
    n = P['n']
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(bicliques)):
            for j in range(i+1, len(bicliques)):
                if time.time() >= deadline:
                    return bicliques
                li, ri = bicliques[i]
                lj, rj = bicliques[j]
                
                # Try merging: left = li ∪ lj, right = ri ∪ rj
                # Or left = li ∪ rj, right = ri ∪ lj (swap)
                # Check if merged forms valid biclique with all cross edges being host edges
                # And the edge sets partition correctly
                
                # Option 1: merge as (li∪lj, ri∪rj)
                merged_l1 = set(li) | set(lj)
                merged_r1 = set(ri) | set(rj)
                # Check all cross pairs are edges
                valid1 = True
                for a in merged_l1:
                    for b in merged_r1:
                        if b not in adj[a]:
                            valid1 = False
                            break
                    if not valid1:
                        break
                
                if valid1:
                    # Check that the merged biclique's edges are exactly the union of the two
                    edges_i = _biclique_edges(li, ri)
                    edges_j = _biclique_edges(lj, rj)
                    edges_merged = _biclique_edges(sorted(merged_l1), sorted(merged_r1))
                    if edges_merged == edges_i | edges_j:
                        # Valid merge
                        bicliques[i] = [sorted(merged_l1), sorted(merged_r1)]
                        del bicliques[j]
                        improved = True
                        break
            if improved:
                break
        
        # Option 2: swap merge (li∪rj, ri∪lj)
        if not improved:
            for i in range(len(bicliques)):
                for j in range(i+1, len(bicliques)):
                    if time.time() >= deadline:
                        return bicliques
                    li, ri = bicliques[i]
                    lj, rj = bicliques[j]
                    
                    merged_l2 = set(li) | set(rj)
                    merged_r2 = set(ri) | set(lj)
                    valid2 = True
                    for a in merged_l2:
                        for b in merged_r2:
                            if b not in adj[a]:
                                valid2 = False
                                break
                        if not valid2:
                            break
                    
                    if valid2:
                        edges_i = _biclique_edges(li, ri)
                        edges_j = _biclique_edges(lj, rj)
                        edges_merged = _biclique_edges(sorted(merged_l2), sorted(merged_r2))
                        if edges_merged == edges_i | edges_j:
                            bicliques[i] = [sorted(merged_l2), sorted(merged_r2)]
                            del bicliques[j]
                            improved = True
                            break
                if improved:
                    break
    
    return bicliques

def _try_split_move(bicliques, adj, deadline):
    """Try moving a vertex from one side of a biclique to another to enable merges."""
    # This is complex; skip for now and rely on merge
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    adj = _build_adj()
    
    # Get initial greedy partition
    bicliques = _greedy_partition(adj, deadline)
    
    # Local search: try merging
    bicliques = _try_merge(bicliques, adj, deadline)
    
    # Try multiple greedy restarts with different orderings
    best = bicliques
    best_count = len(bicliques)
    
    # Try a few more greedy strategies
    for attempt in range(5):
        if time.time() >= deadline:
            break
        
        # Shuffle edge order by using different heuristics
        all_edges = set()
        for u, v in P['edges']:
            all_edges.add((min(u,v), max(u,v)))
        remaining = set(all_edges)
        bicliques2 = []
        
        while remaining and time.time() < deadline:
            # Pick edge with random-ish tie-breaking
            best_edge = None
            best_score = -1
            for e in remaining:
                u, v = e
                score = len(adj[u]) + len(adj[v]) + (attempt * 7 + u * 13 + v * 17) % 100
                if score > best_score:
                    best_score = score
                    best_edge = e
            
            if best_edge is None:
                break
            
            u, v = best_edge
            left = {u}
            right = {v}
            
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for w in range(P['n']):
                    if w in left or w in right:
                        continue
                    if all(v2 in adj[w] for v2 in right):
                        if any((min(w,v2), max(w,v2)) in remaining for v2 in right):
                            left.add(w)
                            changed = True
                            break
                if changed:
                    continue
                for w in range(P['n']):
                    if w in left or w in right:
                        continue
                    if all(u2 in adj[w] for u2 in left):
                        if any((min(u2,w), max(u2,w)) in remaining for u2 in left):
                            right.add(w)
                            changed = True
                            break
            
            bl = sorted(left)
            br = sorted(right)
            valid_edges = set()
            for a in bl:
                for b in br:
                    e = (min(a,b), max(a,b))
                    if e in remaining:
                        valid_edges.add(e)
            
            if valid_edges:
                bicliques2.append([bl, br])
                remaining -= valid_edges
        
        for e in remaining:
            u, v = e
            bicliques2.append([[u], [v]])
        
        bicliques2 = _try_merge(bicliques2, adj, deadline)
        
        if len(bicliques2) < best_count:
            best = bicliques2
            best_count = len(bicliques2)
    
    # Final validation and format
    result = []
    for bl, br in best:
        result.append([list(bl), list(br)])
    
    return result
