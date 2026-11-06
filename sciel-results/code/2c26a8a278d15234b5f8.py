import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (150s kill, leave buffer)
    
    n = 40
    n_groups = 8
    group_size = 5
    
    # Partition vertices into 8 groups of 5
    # Group g contains vertices [g*5, g*5+1, ..., g*5+4]
    
    best_edges = None
    best_count = 0
    
    # Pre-compute all inter-group edges (no intra-group edges)
    # For groups i < j, all 5*5 = 25 edges between them
    # Total possible inter-group edges: C(8,2) * 25 = 28 * 25 = 700
    
    # For C6 detection in a bipartite block structure:
    # A C6 must visit 6 distinct groups in a cycle
    # The cycle alternates between "even" and "odd" positions
    # We can model this at the group level with a bipartite graph
    
    # Strategy: 
    # 1. Build a group-level adjacency matrix (8x8) where group[i,j] = set of edges between group i and group j
    # 2. For C6 checking, we need to verify that no 6-cycle exists at the group level
    #    where each "edge" in the group cycle has at least one actual edge
    
    # Better approach: Use a randomized greedy with local search
    # 1. Start with random subset of inter-group edges
    # 2. Greedily add edges that don't create C6
    # 3. Local search: try to add more edges, remove edges to enable more additions
    
    def create_adj_from_edges(edges):
        """Create adjacency sets from edge list"""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def has_c6(adj):
        """Check if graph contains a C6 using BFS from each edge"""
        # For each edge (u,v), check if there's a path of length 5 from u to v
        # that doesn't use the edge (u,v) and is simple
        for u in range(n):
            for v in list(adj[u]):
                if u >= v:
                    continue
                # Check if removing (u,v) leaves a path of length 5 from u to v
                # This is equivalent to checking if there's a C6 containing edge (u,v)
                # BFS from u, avoiding v until depth 5
                visited = {u}
                queue = [(u, 0, frozenset())]  # (node, depth, path_mask)
                # For speed, we'll do a limited BFS
                # Actually, let's use a different approach: 
                # Count the number of length-5 paths between u and v
                # If > 0, there's a C6
                
                # More efficient: check if there exists a path u-a-b-c-d-v
                # where a,b,c,d are distinct and not u or v
                # and all edges exist
                
                # We can do this with a 4-level expansion
                # Level 0: {u}
                # Level 1: neighbors of u (excluding v)
                # Level 2: neighbors of level 1 (excluding u, v, and level 1)
                # Level 3: neighbors of level 2 (excluding u, v, level 1, level 2)
                # Level 4: neighbors of level 3 (excluding u, v, level 1, level 2, level 3)
                # Check if v is in level 4
                
                # For performance, limit the search
                level0 = {u}
                level1 = set(adj[u] - {v})
                if not level1:
                    continue
                level2 = set()
                for node in level1:
                    level2 |= (adj[node] - level0 - level1 - {v})
                if not level2:
                    continue
                level3 = set()
                for node in level2:
                    level3 |= (adj[node] - level0 - level1 - level2 - {v})
                if not level3:
                    continue
                level4 = set()
                for node in level3:
                    level4 |= (adj[node] - level0 - level1 - level2 - level3 - {v})
                if v in level4:
                    return True
        return False
    
    def would_create_c6(u, v, adj):
        """Check if adding edge (u,v) creates a C6"""
        # After adding (u,v), check if there's a C6 containing this edge
        # This means: is there a path of length 5 from u to v in the current graph?
        # (not using the edge (u,v) since it's not there yet)
        
        # Path: u - a - b - c - d - v
        # a in adj[u], b in adj[a], c in adj[b], d in adj[c], v in adj[d]
        # All vertices distinct
        
        if u not in adj or v not in adj:
            return False
        
        # Quick check: if deg(u) < 2 or deg(v) < 2, can't form C6
        if len(adj[u]) < 2 or len(adj[v]) < 2:
            return False
        
        # Do a 4-step expansion from u, checking if v is reachable at step 4
        # with a simple path
        # For efficiency, we'll track visited nodes in the path
        
        # Level 1: neighbors of u
        level1 = set(adj[u])
        if not level1:
            return False
        
        # Level 2: for each a in level1, neighbors of a excluding u
        # We need to track which a led to which b to ensure simplicity
        # For speed, we'll use a simplified check that might have false positives
        # but we'll verify later
        
        # Actually, let's do a proper DFS with depth limit
        # Since we're only checking one edge addition, this should be fast enough
        
        def dfs(current, depth, visited):
            if depth == 4:
                return v in adj[current]
            
            for nxt in adj[current]:
                if nxt in visited:
                    continue
                if nxt == v and depth < 4:
                    continue  # Can't reach v before depth 4
                if nxt == u:
                    continue  # Can't go back to u
                visited.add(nxt)
                if dfs(nxt, depth + 1, visited):
                    visited.remove(nxt)
                    return True
                visited.remove(nxt)
            return False
        
        visited = {u}
        return dfs(u, 0, visited)
    
    def greedy_construct(seed):
        """Greedy construction with random edge ordering"""
        rng = random.Random(seed)
        
        # Generate all inter-group edges
        all_edges = []
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                for a in range(i * group_size, (i + 1) * group_size):
                    for b in range(j * group_size, (j + 1) * group_size):
                        all_edges.append((a, b))
        
        rng.shuffle(all_edges)
        
        adj = [set() for _ in range(n)]
        current_edges = []
        
        for u, v in all_edges:
            if would_create_c6(u, v, adj):
                continue
            adj[u].add(v)
            adj[v].add(u)
            current_edges.append((u, v))
        
        return current_edges, adj
    
    def local_search(edges, adj, seed, time_limit):
        """Local search to improve the graph"""
        rng = random.Random(seed)
        start_ls = time.time()
        
        current_edges = list(edges)
        current_adj = [set(s) for s in adj]
        current_count = len(current_edges)
        
        # Try to add more edges
        all_inter_group_edges = set()
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                for a in range(i * group_size, (i + 1) * group_size):
                    for b in range(j * group_size, (j + 1) * group_size):
                        all_inter_group_edges.add((a, b))
        
        missing_edges = [e for e in all_inter_group_edges if e not in set(current_edges)]
        rng.shuffle(missing_edges)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for u, v in missing_edges:
                if time.time() >= time_limit:
                    break
                if (u, v) in set(current_edges) or (v, u) in set(current_edges):
                    continue
                if would_create_c6(u, v, current_adj):
                    continue
                # Add the edge
                current_adj[u].add(v)
                current_adj[v].add(u)
                current_edges.append((u, v))
                missing_edges.remove((u, v))
                improved = True
                current_count += 1
        
        return current_edges, current_adj, current_count
    
    # Main search loop
    num_restarts = 0
    best_edges = []
    best_count = 0
    
    while time.time() < deadline:
        num_restarts += 1
        seed = num_restarts * 12345 + 67890
        
        # Greedy construction
        edges, adj = greedy_construct(seed)
        count = len(edges)
        
        if count > best_count:
            best_count = count
            best_edges = edges[:]
        
        # Local search if we have time
        remaining_time = deadline - time.time()
        if remaining_time > 5.0 and count >= best_count - 5:
            ls_time_limit = min(remaining_time - 2.0, 10.0)
            edges_ls, adj_ls, count_ls = local_search(edges, adj, seed + 1, ls_time_limit)
            
            if count_ls > best_count:
                best_count = count_ls
                best_edges = edges_ls[:]
        
        # Early stop if we reach a good threshold
        if best_count >= 105:
            break
    
    # If we didn't find anything, return empty
    if not best_edges:
        return []
    
    # Convert to list of lists [u, v]
    return [[u, v] for u, v in best_edges]
