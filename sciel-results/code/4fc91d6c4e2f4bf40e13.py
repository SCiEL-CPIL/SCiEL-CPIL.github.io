from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    trees = instance_trees(20, 611)
    n = 20
    
    # Order trees by size descending
    order = list(range(19, -1, -1))
    
    used_edges = set()
    assignment = [None] * n
    
    # Pre-shuffled host lists for each tree index to diversify search
    # Use a fixed seed for reproducibility
    rng_search = Random(12345)
    host_orders = {}
    for idx in order:
        hosts = list(range(n))
        rng_search.shuffle(hosts)
        host_orders[idx] = hosts
        
    # For the first tree (largest), we fix vertex 0 to host 0 to break symmetry
    # But we should still try other hosts if that fails? 
    # Due to symmetry, if a solution exists with tree vertex 0 at host h, 
    # there is a symmetric solution with tree vertex 0 at host 0.
    # So fixing to 0 is safe for the FIRST tree.
    
    def backtrack(idx):
        if idx == len(order):
            return True
            
        k_idx = order[idx]
        k = k_idx + 1
        tree_edges = trees[k_idx]
        
        if k == 1:
            assignment[k_idx] = [0]
            if backtrack(idx + 1):
                return True
            assignment[k_idx] = None
            return False

        # Build adjacency
        adj = {i: [] for i in range(k)}
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        parent = {i: -1 for i in range(k)}
        parent[0] = -2
        
        # Candidates for vertex 0
        if idx == 0:
            candidates = [0]
        else:
            candidates = host_orders[k_idx]
            
        for host_0 in candidates:
            mapping = {0: host_0}
            used_hosts = {host_0}
            
            def assign_tree(vertex):
                if vertex == -1:
                    return True
                
                # Try hosts in shuffled order
                for host in host_orders[k_idx]:
                    if host in used_hosts:
                        continue
                    
                    # Check edges to assigned neighbors
                    valid = True
                    for nb in adj[vertex]:
                        if nb in mapping:
                            edge = (min(host, mapping[nb]), max(host, mapping[nb]))
                            if edge in used_edges:
                                valid = False
                                break
                    
                    if not valid:
                        continue
                    
                    mapping[vertex] = host
                    used_hosts.add(host)
                    
                    success = True
                    for nb in adj[vertex]:
                        if parent[nb] == vertex:
                            if not assign_tree(nb):
                                success = False
                                break
                    
                    if success:
                        return True
                    
                    del mapping[vertex]
                    used_hosts.discard(host)
                
                return False
            
            success = True
            for nb in adj[0]:
                parent[nb] = 0
                if not assign_tree(nb):
                    success = False
                    break
            
            if success:
                for u, v in tree_edges:
                    edge = (min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
                    used_edges.add(edge)
                
                assignment[k_idx] = [mapping[i] for i in range(k)]
                
                if backtrack(idx + 1):
                    return True
                
                for u, v in tree_edges:
                    edge = (min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
                    used_edges.discard(edge)
                assignment[k_idx] = None
        
        return False

    if backtrack(0):
        return assignment
    
    return [None] * n

