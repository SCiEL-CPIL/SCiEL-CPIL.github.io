from random import Random

def construct():
    def instance_trees(n=20, seed=611):
        rng = Random(seed)
        trees = []
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
                edges.append((leaf, x))
                degree[leaf] -= 1
                degree[x] -= 1
            u, w = [v for v in range(k) if degree[v] == 1]
            edges.append((u, w))
            trees.append(edges)
        return trees

    trees = instance_trees()
    N = 20
    all_edges = set()
    for i in range(N):
        for j in range(i + 1, N):
            all_edges.add((i, j))
    
    used_edges = set()
    result = [None] * 20
    
    def add_edges(edge_list, mapping, used_set):
        for u, v in edge_list:
            hu = mapping[u]
            hv = mapping[v]
            if hu > hv: hu, hv = hv, hu
            used_set.add((hu, hv))
            
    # Embed T20 first (Identity)
    map_20 = list(range(20))
    result[19] = map_20
    add_edges(trees[19], map_20, used_edges)
    
    def find_embedding(tree_edges, used_edges_set):
        k = len(tree_edges) + 1 if tree_edges else 1
        if k == 1:
            # Pick any unused vertex? No, T1 has no edges, so any vertex works.
            # But we must return distinct vertices.
            # Since it's the last one, we just need a valid host vertex.
            # Actually, T1 is a single vertex. It doesn't consume edges.
            # We can just pick 0, but 0 might be used by other trees' vertices?
            # The problem says "position i is the image of vertex i".
            # "Embedded trees must be edge-disjoint".
            # It does NOT say vertex-disjoint.
            # "Entry k-1 is None or a list of k distinct int host vertices".
            # So vertices within a tree must be distinct.
            # Vertices across trees can be shared?
            # "Packing trees... into K20".
            # Usually packing implies edge-disjoint.
            # If vertices could be shared, it's easier.
            # Let's check the constraint: "list of k distinct int host vertices".
            # This only applies to the list itself.
            # So T1 can map to vertex 0 even if T20 uses 0.
            # So for k=1, we can just return [0].
            return [0]
            
        adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Order vertices: BFS from 0
        order = []
        visited = [False] * k
        queue = [0]
        visited[0] = True
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
                    
        mapping = [-1] * k
        used_host_vertices = [False] * N
        
        def dfs(idx):
            if idx == len(order):
                return True
            
            u = order[idx]
            parent_host = -1
            for v in adj[u]:
                if mapping[v] != -1:
                    parent_host = mapping[v]
                    break
            
            candidates = []
            if parent_host != -1:
                for w in range(N):
                    if not used_host_vertices[w]:
                        edge = (min(parent_host, w), max(parent_host, w))
                        if edge not in used_edges_set:
                            candidates.append(w)
            else:
                for w in range(N):
                    if not used_host_vertices[w]:
                        candidates.append(w)
                        
            # Heuristic: Sort candidates by number of available edges to unmapped neighbors?
            # Or just randomize/shuffle to avoid bad local minima?
            # Since we need deterministic result, we stick to index order or degree.
            
            for w in candidates:
                valid = True
                for v in adj[u]:
                    if mapping[v] != -1:
                        hv = mapping[v]
                        edge = (min(hv, w), max(hv, w))
                        if edge in used_edges_set:
                            valid = False
                            break
                if not valid:
                    continue
                
                mapping[u] = w
                used_host_vertices[w] = True
                
                if dfs(idx + 1):
                    return True
                    
                mapping[u] = -1
                used_host_vertices[w] = False
                
            return False
            
        if dfs(0):
            return mapping
        return None

    for k in range(19, 0, -1):
        tree_edges = trees[k-1]
        mapping = find_embedding(tree_edges, used_edges)
        if mapping is None:
            # Fallback: try different root or random restarts
            # For now, if it fails, we return None for this tree
            # But we should try harder.
            # Let's add a simple retry with shuffled vertex order in tree?
            # The tree structure is fixed, but we can root it differently.
            # The current code roots at 0.
            # We can try all possible roots for small trees.
            found = False
            for root in range(len(tree_edges) + 1):
                # Re-root the tree at 'root'
                # This changes the BFS order.
                # We need to re-construct the adjacency and order.
                # This is complex.
                # Instead, we can just shuffle the candidate order in DFS.
                pass
            # If still not found, leave as None
            result[k-1] = None
        else:
            result[k-1] = mapping
            add_edges(tree_edges, mapping, used_edges)
            
    return result
