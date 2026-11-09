import random

def construct():
    # 1. Initialize with 8 disjoint K5s
    n_clusters = 8
    cluster_size = 5
    n_vertices = 40
    
    edges = set()
    adj = [[] for _ in range(n_vertices)]
    
    # Add intra-cluster edges
    for k in range(n_clusters):
        base = k * cluster_size
        for i in range(cluster_size):
            for j in range(i + 1, cluster_size):
                u = base + i
                v = base + j
                edges.add((u, v))
                edges.add((v, u))
                adj[u].append(v)
                adj[v].append(u)
                
    # 2. Generate candidate inter-cluster edges
    candidates = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            base_i = i * cluster_size
            base_j = j * cluster_size
            for a in range(cluster_size):
                for b in range(cluster_size):
                    u = base_i + a
                    v = base_j + b
                    candidates.append((u, v))
                    
    random.shuffle(candidates)
    
    # 3. Greedy addition with C6 check
    # Helper: get N2(u) - set of (node, prev_node) pairs reachable in 2 steps
    def get_n2(u):
        # Returns list of (v, w) such that u-w-v is a path
        # We need distinct u, w, v
        res = []
        for w in adj[u]:
            for v in adj[w]:
                if v != u:
                    res.append((v, w))
        return res

    for u, v in candidates:
        if (u, v) in edges:
            continue
            
        # Check if adding (u,v) creates a C6
        # C6: u - a - b - c - d - v
        # Split: u-a-b (len 2) and v-d-c (len 2) with edge (b,c)
        n2_u = get_n2(u)
        n2_v = get_n2(v)
        
        conflict = False
        # Optimization: If u and v are in same cluster, we already have edges?
        # No, candidates are inter-cluster only.
        
        # We need to check if there exists b, c, a, d such that:
        # (b,a) in n2_u, (c,d) in n2_v
        # (b,c) is an edge
        # All vertices u,v,a,b,c,d are distinct.
        
        # To speed up, we can pre-filter.
        # But N is small.
        
        for b, a in n2_u:
            if b == v or a == v:
                continue
            for c, d in n2_v:
                if c == u or d == u:
                    continue
                if b == c:
                    continue
                if b == d or a == c or a == d:
                    continue
                # Check if (b,c) is an edge
                # We can check adjacency list or set
                # Using set for O(1)
                if (b, c) in edges:
                    conflict = True
                    break
            if conflict:
                break
                
        if not conflict:
            # Add edge
            edges.add((u, v))
            edges.add((v, u))
            adj[u].append(v)
            adj[v].append(u)
            
    # Convert to list of lists
    result = []
    for u, v in edges:
        if u < v:
            result.append([u, v])
            
    return result
