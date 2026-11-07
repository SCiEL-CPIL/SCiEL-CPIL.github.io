"""Improved program for small exact biclique partitions of a random graph."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _validate_bicliques(bicliques, edges_set):
    """Validate that bicliques form a valid partition."""
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        for u in left:
            for v in right:
                if u == v:
                    return False
                edge = tuple(sorted((u, v)))
                if edge not in edges_set:
                    return False
                if edge in covered:
                    return False
                covered.add(edge)
    return covered == edges_set

def _biclique_edges(left, right):
    """Return the set of edges covered by a biclique."""
    edges = set()
    for u in left:
        for v in right:
            if u != v:
                edges.add(tuple(sorted((u, v))))
    return edges

def _find_max_biclique(adj, remaining_edges, n):
    """Find a large biclique in the subgraph defined by remaining_edges using multiple strategies."""
    if not remaining_edges:
        return None, None
    
    # Count how many remaining edges each vertex is incident to
    deg = {}
    for u, v in remaining_edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    
    if not deg:
        return None, None
    
    best_left, best_right = None, None
    best_size = 0
    
    # Try multiple anchor vertices
    anchors = sorted(deg.keys(), key=lambda x: -deg[x])[:min(10, len(anchors))]
    
    for anchor in anchors:
        # Its neighbors in the remaining edge set
        candidates = set()
        for u, v in remaining_edges:
            if u == anchor:
                candidates.add(v)
            elif v == anchor:
                candidates.add(u)
        
        if not candidates:
            continue
        
        # Start with anchor on left, candidates on right
        left = {anchor}
        right = set(candidates)
        
        # Iteratively refine
        changed = True
        while changed:
            changed = False
            # Filter right: must be connected to all in left via remaining edges
            new_right = set()
            for v in right:
                valid = True
                for u in left:
                    if u == v:
                        continue
                    edge = tuple(sorted((u, v)))
                    if edge not in remaining_edges:
                        valid = False
                        break
                if valid:
                    new_right.add(v)
            if new_right != right:
                right = new_right
                changed = True
            
            if not right:
                break
            
            # Filter left: must be connected to all in right via remaining edges
            new_left = set()
            for u in left:
                valid = True
                for v in right:
                    if u == v:
                        continue
                    edge = tuple(sorted((u, v)))
                    if edge not in remaining_edges:
                        valid = False
                        break
                if valid:
                    new_left.add(u)
            if new_left != left:
                left = new_left
                changed = True
        
        if not left or not right:
            continue
        
        # Try to expand left by adding vertices connected to all of right via remaining edges
        for u in deg:
            if u not in left:
                valid = True
                for v in right:
                    if u == v:
                        continue
                    edge = tuple(sorted((u, v)))
                    if edge not in remaining_edges:
                        valid = False
                        break
                if valid:
                    left.add(u)
        
        # Try to expand right by adding vertices connected to all of left via remaining edges
        for v in deg:
            if v not in right:
                valid = True
                for u in left:
                    if u == v:
                        continue
                    edge = tuple(sorted((u, v)))
                    if edge not in remaining_edges:
                        valid = False
                        break
                if valid:
                    right.add(v)
        
        # Final refinement
        changed = True
        while changed:
            changed = False
            new_right = set()
            for v in right:
                valid = True
                for u in left:
                    if u == v:
                        continue
                    edge = tuple(sorted((u, v)))
                    if edge not in remaining_edges:
                        valid = False
                        break
                if valid:
                    new_right.add(v)
            if new_right != right:
                right = new_right
                changed = True
            
            if not right:
                break
            
            new_left = set()
            for u in left:
                valid = True
                for v in right:
                    if u == v:
                        continue
                    edge = tuple(sorted((u, v)))
                    if edge not in remaining_edges:
                        valid = False
                        break
                if valid:
                    new_left.add(u)
            if new_left != left:
                left = new_left
                changed = True
        
        size = len(left) * len(right)
        if size > best_size:
            best_size = size
            best_left = left
            best_right = right
    
    if best_left is None or best_right is None:
        return None, None
    
    return list(best_left), list(best_right)

def _greedy_partition(adj, edges_set, n, deadline):
    """Greedy biclique partition."""
    remaining = set(edges_set)
    bicliques = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        left, right = _find_max_biclique(adj, remaining, n)
        
        if left is None or right is None:
            u, v = next(iter(remaining))
            bicliques.append([[u], [v]])
            remaining.discard((u, v))
            continue
        
        # Verify this biclique only uses remaining edges
        valid = True
        for u in left:
            for v in right:
                if u == v:
                    continue
                edge = tuple(sorted((u, v)))
                if edge not in remaining:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Shrink to only remaining edges
            new_left = []
            for u in left:
                if all((u == v) or (tuple(sorted((u, v))) in remaining) for v in right):
                    new_left.append(u)
            new_right = []
            for v in right:
                if all((u == v) or (tuple(sorted((u, v))) in remaining) for u in new_left):
                    new_right.append(v)
            left, right = new_left, new_right
        
        if not left or not right:
            u, v = next(iter(remaining))
            bicliques.append([[u], [v]])
            remaining.discard((u, v))
            continue
        
        # Remove covered edges
        for u in left:
            for v in right:
                if u != v:
                    edge = tuple(sorted((u, v)))
                    remaining.discard(edge)
        
        bicliques.append([left, right])
    
    # Handle leftover edges
    for edge in remaining:
        u, v = edge
        bicliques.append([[u], [v]])
    
    return bicliques

def _try_merge(bicliques, i, j, edges_set):
    """Try to merge bicliques i and j. Returns new biclique list if successful, else None."""
    left1, right1 = bicliques[i]
    left2, right2 = bicliques[j]
    
    edges1 = _biclique_edges(left1, right1)
    edges2 = _biclique_edges(left2, right2)
    
    # Try all four merge combinations
    for merge_type in range(4):
        if merge_type == 0:
            new_left = list(set(left1) | set(left2))
            new_right = list(set(right1) & set(right2))
        elif merge_type == 1:
            new_left = list(set(left1) & set(left2))
            new_right = list(set(right1) | set(right2))
        elif merge_type == 2:
            new_left = list(set(left1) | set(left2))
            new_right = list(set(right1) | set(right2))
        else:
            new_left = list(set(left1) & set(left2))
            new_right = list(set(right1) & set(right2))
        
        if not new_left or not new_right:
            continue
        
        new_edges = _biclique_edges(new_left, new_right)
        
        # Check all new edges are in the graph
        if not new_edges.issubset(edges_set):
            continue
        
        # Check new edges are exactly the union of edges from bicliques i and j
        if new_edges != (edges1 | edges2):
            continue
        
        # Build new biclique list
        new_bicliques = [bicliques[k] for k in range(len(bicliques)) if k != i and k != j]
        new_bicliques.append([new_left, new_right])
        
        if _validate_bicliques(new_bicliques, edges_set):
            return new_bicliques
    
    return None

def _try_absorb(bicliques, i, j, edges_set):
    """Try to absorb biclique j into biclique i by expanding i's sides."""
    left_i, right_i = bicliques[i]
    left_j, right_j = bicliques[j]
    
    edges_i = _biclique_edges(left_i, right_i)
    edges_j = _biclique_edges(left_j, right_j)
    
    # Try adding vertices from j to i's sides
    best_new_left, best_new_right = None, None
    best_gain = 0
    
    # Try expanding right_i with vertices from left_j
    for v in left_j:
        if v in left_i:
            continue
        # Check if v is connected to all in left_i
        valid = True
        for u in left_i:
            if u == v:
                continue
            edge = tuple(sorted((u, v)))
            if edge not in edges_set:
                valid = False
                break
        if valid:
            candidate_left = list(left_i)
            candidate_right = list(right_i) + [v]
            new_edges = _biclique_edges(candidate_left, candidate_right)
            # Check that new edges are subset of edges_i | edges_j
            if new_edges.issubset(edges_i | edges_j):
                gain = len(new_edges) - len(edges_i)
                if gain > best_gain:
                    best_gain = gain
                    best_new_left = candidate_left
                    best_new_right = candidate_right
    
    # Try expanding left_i with vertices from right_j
    for u in right_j:
        if u in right_i:
            continue
        # Check if u is connected to all in right_i
        valid = True
        for v in right_i:
            if u == v:
                continue
            edge = tuple(sorted((u, v)))
            if edge not in edges_set:
                valid = False
                break
        if valid:
            candidate_left = list(left_i) + [u]
            candidate_right = list(right_i)
            new_edges = _biclique_edges(candidate_left, candidate_right)
            if new_edges.issubset(edges_i | edges_j):
                gain = len(new_edges) - len(edges_i)
                if gain > best_gain:
                    best_gain = gain
                    best_new_left = candidate_left
                    best_new_right = candidate_right
    
    if best_new_left is None:
        return None
    
    # Check that the expanded biclique covers all of edges_j
    new_edges = _biclique_edges(best_new_left, best_new_right)
    if not edges_j.issubset(new_edges):
        return None
    
    # Build new biclique list
    new_bicliques = [bicliques[k] for k in range(len(bicliques)) if k != i and k != j]
    new_bicliques.append([best_new_left, best_new_right])
    
    if _validate_bicliques(new_bicliques, edges_set):
        return new_bicliques
    
    return None

def _local_search(bicliques, adj, edges_set, deadline):
    """Local search to reduce number of bicliques."""
    current = [list(bc) for bc in bicliques]
    best_count = len(current)
    
    while time.time() < deadline:
        improved = False
        
        # Try merging pairs
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                if len(current) <= best_count:
                    break
                result = _try_merge(current, i, j, edges_set)
                if result is not None:
                    current = result
                    best_count = len(current)
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try absorbing
        for i in range(len(current)):
            for j in range(len(current)):
                if i == j:
                    continue
                if len(current) <= best_count:
                    break
                result = _try_absorb(current, i, j, edges_set)
                if result is not None:
                    current = result
                    best_count = len(current)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s limit
    
    n = P['n']
    adj = _build_adjacency()
    edges_set = set(tuple(sorted(e)) for e in P['edges'])
    
    # Fallback: star decomposition
    def star_partition():
        result = []
        for u in range(n):
            neighbors = [v for v in adj[u]]
            if neighbors:
                result.append([list(neighbors), [u]])
        return result
    
    fallback = star_partition()
    best = fallback
    
    # Greedy biclique partition
    if time.time() < deadline:
        greedy = _greedy_partition(adj, edges_set, n, deadline)
        if _validate_bicliques(greedy, edges_set) and len(greedy) < len(best):
            best = greedy
    
    # Local search to try to improve
    if time.time() < deadline:
        improved = _local_search(best, adj, edges_set, deadline)
        if _validate_bicliques(improved, edges_set) and len(improved) < len(best):
            best = improved
    
    # Ensure proper types
    result = []
    for left, right in best:
        result.append([[int(v) for v in left], [int(v) for v in right]])
    
    return result

# EVOLVE-BLOCK-END
