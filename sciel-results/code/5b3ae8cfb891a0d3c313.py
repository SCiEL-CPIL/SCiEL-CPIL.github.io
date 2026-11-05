"""Path decomposition of fixed connected graphs via iterative path merging."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _max_matching_bipartite(left_nodes, right_nodes, adj):
    """
    Find maximum matching in bipartite graph using Hopcroft-Karp.
    left_nodes: list of left vertex IDs
    right_nodes: list of right vertex IDs
    adj: dict mapping left node -> list of right nodes
    Returns list of (left, right) pairs.
    """
    if not left_nodes or not right_nodes:
        return []
    
    # Map to indices
    left_idx = {v: i for i, v in enumerate(left_nodes)}
    right_idx = {v: i for i, v in enumerate(right_nodes)}
    n_left = len(left_nodes)
    n_right = len(right_nodes)
    
    # Build adjacency in index form
    adj_idx = [[] for _ in range(n_left)]
    for u in left_nodes:
        for v in adj.get(u, []):
            if v in right_idx:
                adj_idx[left_idx[u]].append(right_idx[v])
    
    # Hopcroft-Karp
    pair_u = [-1] * n_left
    pair_v = [-1] * n_right
    dist = [0] * n_left
    
    from collections import deque
    
    def bfs():
        queue = deque()
        for u in range(n_left):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = -1
        found = False
        while queue:
            u = queue.popleft()
            for v in adj_idx[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == -1:
                    dist[pu] = dist[u] + 1
                    queue.append(pu)
        return found
    
    def dfs(u):
        for v in adj_idx[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False
    
    matching = []
    while bfs():
        for u in range(n_left):
            if pair_u[u] == -1:
                if dfs(u):
                    matching.append((left_nodes[u], right_nodes[pair_u[u]]))
    
    return matching

def _decompose_graph(edges, deadline):
    """
    Decompose edges into minimum number of simple paths.
    Strategy: start with each edge as a path, then iteratively merge
    paths using maximum matching on an auxiliary bipartite graph.
    """
    if not edges:
        return []
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Start with each edge as a path fragment
    # Represent each path as (start, end, edge_set, vertex_set)
    # Initially: one path per edge
    paths = []
    for u, v in edges:
        paths.append({'start': u, 'end': v, 'edges': {tuple(sorted((u, v)))}, 'verts': {u, v}})
    
    # Iteratively try to merge paths
    max_iters = 50
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Find all possible merges
        # A merge connects end of path A to start of path B if:
        # 1. There's an edge between A.end and B.start
        # 2. B.start is not already in A's vertex set (simplicity)
        # 3. A.end is not already in B's vertex set (simplicity)
        # 4. The connecting edge is not already used
        
        # Build auxiliary bipartite graph:
        # Left side: paths that can be extended (their end vertex)
        # Right side: paths that can be extended from (their start vertex)
        # Edge if they can be merged
        
        left_nodes = []
        right_nodes = []
        adj_aux = defaultdict(list)
        
        # For each path, check if it can merge with others
        # We need to be careful: each path can only be merged once per iteration
        # Left: path i (as the "first" path), Right: path j (as the "second" path)
        # Merge: i.end -> j.start via an edge
        
        n_paths = len(paths)
        for i in range(n_paths):
            for j in range(n_paths):
                if i == j:
                    continue
                pi = paths[i]
                pj = paths[j]
                # Check if pi.end is adjacent to pj.start
                if pj['start'] in adj.get(pi['end'], []):
                    # Check simplicity: pi.end not in pj's verts, pj.start not in pi's verts
                    if pi['end'] not in pj['verts'] and pj['start'] not in pi['verts']:
                        # Check edge not used
                        edge_key = tuple(sorted((pi['end'], pj['start'])))
                        if edge_key not in pi['edges'] and edge_key not in pj['edges']:
                            if i not in left_nodes:
                                left_nodes.append(i)
                            if j not in right_nodes:
                                right_nodes.append(j)
                            adj_aux[i].append(j)
        
        if not left_nodes:
            break
        
        # Find maximum matching
        matching = _max_matching_bipartite(left_nodes, right_nodes, adj_aux)
        
        if not matching:
            break
        
        # Apply merges
        merged = set()
        new_paths = []
        for i, j in matching:
            pi = paths[i]
            pj = paths[j]
            # Create new path: pi's start to pj's end
            new_start = pi['start']
            new_end = pj['end']
            new_edges = pi['edges'] | pj['edges'] | {tuple(sorted((pi['end'], pj['start'])))}
            new_verts = pi['verts'] | pj['verts']
            merged.add(i)
            merged.add(j)
            new_paths.append({'start': new_start, 'end': new_end, 'edges': new_edges, 'verts': new_verts})
        
        # Add unmerged paths
        for k in range(n_paths):
            if k not in merged:
                new_paths.append(paths[k])
        
        paths = new_paths
    
    # Convert to vertex lists
    result = []
    for p in paths:
        # Reconstruct path from start to end
        # We need to find the actual path. Since we only merged adjacent paths,
        # we can reconstruct by tracking.
        # Actually, let's store the full vertex sequence.
        result.append(p['verts'])  # This is wrong, we need ordered vertices
    
    # Let me redo this with proper vertex sequences
    # Restart with proper representation
    return None

def _decompose_graph_v2(edges, deadline):
    """
    Decompose edges into minimum number of simple paths using iterative merging.
    Each path is represented as an ordered list of vertices.
    """
    if not edges:
        return []
    
    # Build adjacency set for quick lookup
    edge_set = set()
    adj = defaultdict(set)
    for u, v in edges:
        edge_set.add(tuple(sorted((u, v))))
        adj[u].add(v)
        adj[v].add(u)
    
    # Start with each edge as a path [u, v]
    paths = [[u, v] for u, v in edges]
    
    max_iters = 100
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        n = len(paths)
        if n <= 1:
            break
        
        # Build auxiliary bipartite graph for merging
        # Left: path index i, Right: path index j
        # Can merge if paths[i][-1] -- paths[j][0] is an edge,
        # and no vertex in paths[i] is in paths[j] (except the connecting vertex),
        # and the edge is not already in either path's edge set.
        
        adj_aux = defaultdict(list)
        valid_left = set()
        valid_right = set()
        
        # For efficiency, precompute vertex sets and edge sets
        vert_sets = [set(p) for p in paths]
        edge_sets = [set(tuple(sorted((p[k], p[k+1]))) for k in range(len(p)-1)) for p in paths]
        
        for i in range(n):
            end_i = paths[i][-1]
            for j in range(n):
                if i == j:
                    continue
                start_j = paths[j][0]
                if start_j in adj.get(end_i, set()):
                    # Check edge not used
                    ek = tuple(sorted((end_i, start_j)))
                    if ek in edge_sets[i] or ek in edge_sets[j]:
                        continue
                    # Check simplicity: only end_i and start_j may overlap
                    # Actually, end_i is in vert_sets[i], start_j is in vert_sets[j]
                    # After merge, we need: (vert_sets[i] | vert_sets[j]) - {end_i, start_j} has no intersection
                    # i.e., vert_sets[i] - {end_i} and vert_sets[j] - {start_j} are disjoint
                    overlap = (vert_sets[i] - {end_i}) & (vert_sets[j] - {start_j})
                    if not overlap:
                        adj_aux[i].append(j)
                        valid_left.add(i)
                        valid_right.add(j)
        
        if not valid_left:
            break
        
        left_list = list(valid_left)
        right_list = list(valid_right)
        matching = _max_matching_bipartite(left_list, right_list, adj_aux)
        
        if not matching:
            break
        
        # Apply merges
        merged = set()
        new_paths = []
        for i, j in matching:
            pi = paths[i]
            pj = paths[j]
            new_path = pi + pj  # Concatenate vertex lists
            merged.add(i)
            merged.add(j)
            new_paths.append(new_path)
        
        for k in range(n):
            if k not in merged:
                new_paths.append(paths[k])
        
        paths = new_paths
    
    return paths

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for spec in specs:
        edges = _instance(*spec)
        paths = _decompose_graph_v2(edges, deadline)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
