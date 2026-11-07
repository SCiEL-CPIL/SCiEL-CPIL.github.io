import time
import random
from itertools import combinations

def instance_trees(n=20, seed=611):
    rng = random.Random(seed)
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
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    trees = instance_trees(20, 611)
    
    # used_edges stores edges of K20 that are already occupied
    used_edges = set()
    result = [None] * 20
    
    # Process trees from largest to smallest to maximize edge count
    for k in range(20, 0, -1):
        if time.time() > deadline:
            break
            
        if k == 1:
            # T1 is a single vertex. Any host works.
            # We pick an arbitrary host, say 0.
            # But wait, if 0 is used by another tree's vertex, is that a problem?
            # "Embedded trees must be edge-disjoint". Vertices can overlap.
            # So T1 can map to any vertex. Let's pick 0.
            result[k-1] = [0]
            continue
            
        edges = trees[k-1]
        
        # Build tree structure for T_k
        # We need to root it and get BFS order
        # Since it's a tree, we can build adjacency list
        adj = {i: [] for i in range(k)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Root at 0
        parent = [-1] * k
        order = [0]
        visited = [False] * k
        visited[0] = True
        queue = [0]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
                    order.append(v)
                    
        # order is now BFS order
        # We will use a CSP-based backtracking with MRV and LCV heuristics
        
        def solve_tree_csp(order, parent, k, used_edges, rng):
            mapping = {}
            assigned_hosts = set()
            
            # Pre-compute neighbors for each vertex in the tree for faster access
            tree_neighbors = {i: adj[i] for i in range(k)}
            
            # MRV: Pick variable with fewest candidates
            # LCV: Order candidates by least constraining (e.g., higher degree in remaining graph)
            
            def get_candidates(v):
                candidates = []
                if v == 0:
                    # Root: any host
                    for u in range(20):
                        if u not in assigned_hosts:
                            candidates.append(u)
                else:
                    p = parent[v]
                    if p not in mapping:
                        return []
                    h_p = mapping[p]
                    for u in range(20):
                        if u in assigned_hosts:
                            continue
                        if u == h_p:
                            continue
                        e = (min(h_p, u), max(h_p, u))
                        if e in used_edges:
                            continue
                        candidates.append(u)
                return candidates

            def dfs(idx, remaining_vars):
                if idx == k:
                    return True
                    
                # MRV: Select the unassigned variable with the fewest candidates
                # We maintain a list of unassigned variables in 'remaining_vars'
                # But since we are following a specific order, we can just check the current one
                # OR we can dynamically pick the best variable from all unassigned.
                # Dynamic MRV is stronger.
                
                # Let's implement dynamic MRV.
                # We need to know which variables are unassigned.
                # Instead of passing 'idx', we pass a set of unassigned vars.
                # This is more expensive to manage but better for pruning.
                
                # For simplicity and speed in Python, let's stick to the fixed order but
                # ensure the order is good. Or implement simple MRV on the remaining set.
                
                # Let's try a simpler approach: Fixed order (BFS) but with strong LCV and
                # backtracking. The previous code did this but without restarts.
                # The key improvement suggested is MRV/LCV + Restarts.
                
                # Let's implement a recursive function that picks the best variable.
                # To avoid recomputing candidates for all vars at each step (expensive),
                # we can approximate.
                
                # Actually, for k<=20, computing candidates for all unassigned vars is O(k * 20).
                # Total steps k. Total work O(k^2 * 20) = 20*400*20 = 160,000 ops per node?
                # Number of nodes can be large.
                
                # Let's use the fixed order but improve the candidate selection with LCV.
                # And add randomization.
                
                v = order[idx]
                candidates = get_candidates(v)
                
                if not candidates:
                    return False
                
                # LCV: Sort candidates. 
                # A good proxy for LCV is the degree of the candidate in the remaining available graph.
                # Hosts with higher remaining degree are "less constraining" for future vertices?
                # Actually, we want to leave open options.
                # If we pick a host with many available edges, it might block fewer other hosts?
                # No, if we pick a host, we block its incident edges.
                # So we should pick a host that blocks the "least valuable" edges?
                # Or simply: pick the host that leaves the most options for its neighbors in the tree.
                
                # Simple heuristic: Sort by degree in K20 \ used_edges.
                # Higher degree means more options for connecting to other vertices in THIS tree?
                # No, the other vertices in THIS tree must connect to THIS vertex's neighbors in the tree.
                
                # Let's use a simple randomization + degree heuristic.
                
                # Calculate a score for each candidate
                # Score = number of available edges from this host to other UNASSIGNED tree vertices' potential hosts?
                # Too complex.
                
                # Let's just sort by degree in remaining graph (descending).
                # This tends to place high-degree nodes in high-degree host regions.
                
                def heuristic_score(u):
                    deg = 0
                    for w in range(20):
                        if w == u:
                            continue
                        e = (min(u, w), max(u, w))
                        if e not in used_edges:
                            deg += 1
                    return deg
                
                # To allow randomization for restarts, we can slightly perturb the sort or pick randomly from top few.
                # Or just shuffle if we want pure random restarts.
                # Let's sort by score, but with a small random jitter to break ties and allow variation.
                
                scored_candidates = [(heuristic_score(u), rng.random(), u) for u in candidates]
                scored_candidates.sort(key=lambda x: (-x[0], -x[1]))
                
                for _, _, u in scored_candidates:
                    mapping[v] = u
                    assigned_hosts.add(u)
                    
                    if dfs(idx + 1):
                        return True
                        
                    del mapping[v]
                    assigned_hosts.remove(u)
                    
                return False
                
            if dfs(0):
                return mapping
            return None

        # Try to solve with multiple restarts if needed
        # Since the graph is small, a single run with good heuristics might be enough,
        # but restarts help if we get stuck.
        
        max_restarts = 5
        found = None
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
            rng = random.Random(1000 + attempt)
            mapping = solve_tree_csp(order, parent, k, used_edges, rng)
            if mapping is not None:
                found = mapping
                break
        
        if found is not None:
            # Update used_edges
            for u, v in edges:
                h_u = found[u]
                h_v = found[v]
                e = (min(h_u, h_v), max(h_u, h_v))
                used_edges.add(e)
            
            # Create the result list
            # Position i is the image of vertex i
            res_list = [found[i] for i in range(k)]
            result[k-1] = res_list
            
    return result
