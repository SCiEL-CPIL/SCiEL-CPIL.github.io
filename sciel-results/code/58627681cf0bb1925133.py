import time
import sys
from random import Random

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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n = 20
    trees = instance_trees(n, 611)
    
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # Build adjacency for each tree
    tree_adj = []
    for t in trees:
        k = len(t) + 1 if t else 1
        adj = [[] for _ in range(k)]
        for u, v in t:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    # Order: largest first
    order = list(range(n - 1, -1, -1))
    
    available = set(all_edges)
    solution = [None] * n
    
    def check_time():
        return time.time() < deadline
    
    def embed_tree_backtrack(tree_idx, avail):
        """Backtracking with constraint propagation."""
        k = tree_idx + 1
        if k == 1:
            return [0]
        
        adj = tree_adj[tree_idx]
        edges = trees[tree_idx]
        
        # Find a good root: highest degree vertex
        root_t = max(range(k), key=lambda v: len(adj[v]))
        
        # Build BFS/DFS order from root
        parent = [-1] * k
        order_list = [root_t]
        visited = [False] * k
        visited[root_t] = True
        for node in order_list:
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    parent[nb] = node
                    order_list.append(nb)
        
        # Try different graph vertices for root
        # Sort candidates by degree in available graph
        def graph_degree(g):
            count = 0
            for nb in range(n):
                if nb == g:
                    continue
                e = (min(g, nb), max(g, nb))
                if e in avail:
                    count += 1
            return count
        
        root_candidates = sorted(range(n), key=graph_degree, reverse=True)
        
        for root_g in root_candidates:
            if not check_time():
                return None
            
            mapping = [-1] * k
            mapping[root_t] = root_g
            mapped_set = {root_g}
            
            def dfs(idx):
                if not check_time():
                    return False
                
                if idx == len(order_list):
                    return True
                
                t_node = order_list[idx]
                p_node = parent[t_node]
                if p_node == -1:
                    # Shouldn't happen for non-root
                    return False
                
                p_g = mapping[p_node]
                
                # Candidates for t_node: graph vertices connected to p_g via available edge
                # and not already used
                candidates = []
                for g in range(n):
                    if g in mapped_set:
                        continue
                    e = (min(p_g, g), max(p_g, g))
                    if e in avail:
                        candidates.append(g)
                
                # Heuristic: prefer vertices with more available connections to other unmapped tree neighbors
                def score_candidate(g):
                    # Count how many of t_node's other neighbors (in tree) could potentially connect
                    s = 0
                    for nb in adj[t_node]:
                        if nb == p_node:
                            continue
                        if mapping[nb] != -1:
                            e2 = (min(g, mapping[nb]), max(g, mapping[nb]))
                            if e2 in avail:
                                s += 1
                    # Also count available edges from g
                    for nb2 in range(n):
                        if nb2 == g or nb2 in mapped_set:
                            continue
                        e3 = (min(g, nb2), max(g, nb2))
                        if e3 in avail:
                            s += 0.1
                    return s
                
                candidates.sort(key=score_candidate, reverse=True)
                
                for g in candidates:
                    mapping[t_node] = g
                    mapped_set.add(g)
                    if dfs(idx + 1):
                        return True
                    mapped_set.remove(g)
                    mapping[t_node] = -1
                
                return False
            
            if dfs(1):  # Start from index 1 since root is at index 0
                return mapping[:]
        
        return None
    
    def embed_tree_matching(tree_idx, avail):
        """
        Use a matching-based approach:
        For each tree edge, find a host edge assignment.
        This is modeled as: assign each tree edge (u,v) to a host edge (g_u, g_v)
        such that:
        - The mapping is consistent (same tree vertex maps to same graph vertex)
        - Host edges are distinct
        - Vertices in the tree map to distinct graph vertices
        
        Simplified: use iterative deepening with constraint propagation.
        """
        k = tree_idx + 1
        if k == 1:
            return [0]
        
        adj = tree_adj[tree_idx]
        edges = trees[tree_idx]
        
        # Find root with max degree
        root_t = max(range(k), key=lambda v: len(adj[v]))
        
        # BFS order
        parent = [-1] * k
        bfs_order = [root_t]
        visited = [False] * k
        visited[root_t] = True
        for node in bfs_order:
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    parent[nb] = node
                    bfs_order.append(nb)
        
        # Try root assignments, sorted by available degree
        def avail_deg(g):
            c = 0
            for nb in range(n):
                if nb != g and (min(g, nb), max(g, nb)) in avail:
                    c += 1
            return c
        
        root_cands = sorted(range(n), key=avail_deg, reverse=True)
        
        for root_g in root_cands:
            if not check_time():
                return None
            
            mapping = [-1] * k
            mapping[root_t] = root_g
            used_g = {root_g}
            
            # Process in BFS order
            success = True
            for i in range(1, len(bfs_order)):
                t_node = bfs_order[i]
                p_node = parent[t_node]
                p_g = mapping[p_node]
                
                # Find candidates
                cands = []
                for g in range(n):
                    if g in used_g:
                        continue
                    e = (min(p_g, g), max(p_g, g))
                    if e in avail:
                        # Check consistency with already-mapped neighbors
                        consistent = True
                        for nb in adj[t_node]:
                            if nb != p_node and mapping[nb] != -1:
                                e2 = (min(g, mapping[nb]), max(g, mapping[nb]))
                                if e2 not in avail:
                                    consistent = False
                                    break
                        if consistent:
                            cands.append(g)
                
                if not cands:
                    success = False
                    break
                
                # Pick best candidate
                def cand_score(g):
                    s = 0
                    for nb in adj[t_node]:
                        if nb != p_node and mapping[nb] != -1:
                            e2 = (min(g, mapping[nb]), max(g, mapping[nb]))
                            if e2 in avail:
                                s += 10
                    for nb2 in range(n):
                        if nb2 != g and nb2 not in used_g:
                            e3 = (min(g, nb2), max(g, nb2))
                            if e3 in avail:
                                s += 1
                    return s
                
                cands.sort(key=cand_score, reverse=True)
                mapping[t_node] = cands[0]
                used_g.add(cands[0])
            
            if success:
                # Verify all edges are in avail
                valid = True
                for u, v in edges:
                    e = (min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
                    if e not in avail:
                        valid = False
                        break
                if valid:
                    return mapping[:]
        
        return None
    
    # Try both methods with randomization
    rng = Random(42)
    
    for attempt in range(50):
        if not check_time():
            break
        
        available = set(all_edges)
        solution = [None] * n
        success = True
        
        for t_idx in order:
            if not check_time():
                success = False
                break
            
            k = t_idx + 1
            if k == 1:
                solution[t_idx] = [0]
                continue
            
            # Alternate between methods
            if attempt % 2 == 0:
                mapping = embed_tree_matching(t_idx, available)
            else:
                mapping = embed_tree_backtrack(t_idx, available)
            
            if mapping is None:
                # Try the other method
                if attempt % 2 == 0:
                    mapping = embed_tree_backtrack(t_idx, available)
                else:
                    mapping = embed_tree_matching(t_idx, available)
            
            if mapping is None:
                success = False
                break
            
            # Remove used edges
            for u, v in trees[t_idx]:
                e = (min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
                available.discard(e)
            
            solution[t_idx] = mapping
        
        if success:
            break
    
    # Calculate value
    value = 0
    for t_idx in range(n):
        if solution[t_idx] is not None:
            value += len(trees[t_idx])
    
    if value == 0:
        return [None] * 19 + [list(range(20))]
    
    return solution
