import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    trees = instance_trees(20, 611)
    n = 20
    
    # Build adjacency for K20
    # K20 edge set
    K20_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            K20_edges.add((i, j))
    
    # Process trees in order: larger trees first (more constrained)
    # Actually, let's try processing in reverse order (T20 first) since it's most constrained
    order = list(range(n - 1, -1, -1))  # T20, T19, ..., T1
    
    used_edges = set()
    result = [None] * n
    
    def try_embed(tree_idx, tree_edges, k, used_edges_set, host_map, depth):
        """
        Try to embed tree with k vertices (indices 0..k-1) into K20.
        tree_edges: list of edges in the tree
        used_edges_set: set of edges already used in K20
        Returns: list of k host vertices (image of tree vertex i) or None
        """
        if time.time() > deadline:
            return None
        
        if depth == k:
            return list(host_map)
        
        v = depth  # tree vertex to assign
        # Find neighbors of v in the tree that are already assigned
        assigned_neighbors = []
        for (a, b) in tree_edges:
            if a == v and b < depth:
                assigned_neighbors.append((b, host_map[b]))
            elif b == v and a < depth:
                assigned_neighbors.append((a, host_map[a]))
        
        # Candidate host vertices for v
        candidates = list(range(n))
        
        # Filter: if v has assigned neighbors, host must be adjacent to their hosts
        if assigned_neighbors:
            filtered = []
            for h in candidates:
                ok = True
                for (nb_tree, nb_host) in assigned_neighbors:
                    # Check if (h, nb_host) is an edge in K20
                    edge = (min(h, nb_host), max(h, nb_host))
                    if edge not in K20_edges:
                        ok = False
                        break
                    # Check if edge is already used
                    if edge in used_edges_set:
                        ok = False
                        break
                if ok:
                    filtered.append(h)
            candidates = filtered
        
        # Additional constraint: if v has multiple assigned neighbors, 
        # the edges to those neighbors must all be valid and unused
        # Already handled above
        
        # Try candidates in order (could use heuristic)
        for h in candidates:
            # Check all edges from v to assigned neighbors
            valid = True
            new_edges = []
            for (a, b) in tree_edges:
                if a == v and b < depth:
                    nb_host = host_map[b]
                    edge = (min(h, nb_host), max(h, nb_host))
                    if edge in used_edges_set:
                        valid = False
                        break
                    new_edges.append(edge)
                elif b == v and a < depth:
                    nb_host = host_map[a]
                    edge = (min(h, nb_host), max(h, nb_host))
                    if edge in used_edges_set:
                        valid = False
                        break
                    new_edges.append(edge)
            
            if not valid:
                continue
            
            # Check that host h is not already used by another tree vertex
            # (all tree vertices map to distinct host vertices)
            if h in host_map[:depth]:
                continue
            
            # Assign
            host_map[depth] = h
            used_edges_set.update(new_edges)
            
            res = try_embed(tree_idx, tree_edges, k, used_edges_set, host_map, depth + 1)
            if res is not None:
                return res
            
            # Undo
            host_map[depth] = -1
            for e in new_edges:
                used_edges_set.discard(e)
        
        return None
    
    # For each tree in order, try to embed
    for tree_idx in order:
        if time.time() > deadline:
            break
        
        k = tree_idx + 1
        tree_edges = trees[tree_idx]
        
        if k == 1:
            # Single vertex, any host works
            result[tree_idx] = [0]
            continue
        
        if k == 2:
            # Edge (0,1), need any unused edge
            found = False
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) not in used_edges:
                        result[tree_idx] = [i, j]
                        used_edges.add((i, j))
                        found = True
                        break
                if found:
                    break
            if not found:
                result[tree_idx] = None
            continue
        
        # Backtracking for k >= 3
        host_map = [-1] * k
        res = try_embed(tree_idx, tree_edges, k, used_edges, host_map, 0)
        
        if res is not None:
            result[tree_idx] = res
            # Add all edges of this tree to used_edges
            for (a, b) in tree_edges:
                ha, hb = res[a], res[b]
                edge = (min(ha, hb), max(ha, hb))
                used_edges.add(edge)
        else:
            result[tree_idx] = None
    
    # Verify and count
    total_edges = 0
    for i in range(n):
        if result[i] is not None:
            k = i + 1
            if len(result[i]) != k:
                result[i] = None
                continue
            if len(set(result[i])) != k:
                result[i] = None
                continue
            if any(v < 0 or v >= n for v in result[i]):
                result[i] = None
                continue
            total_edges += len(trees[i])
    
    return result
