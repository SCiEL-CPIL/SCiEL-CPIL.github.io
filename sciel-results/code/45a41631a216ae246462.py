"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def instance_trees(n=20, seed=611):
    rng = random.Random(seed); trees = []
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
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    trees = instance_trees(20, 611)
    
    # Process trees from largest to smallest (most constrained first)
    order = list(range(19, -1, -1))  # indices 19 down to 0 (T20 first)
    
    # Track used edges in K20 (as frozensets of vertex pairs)
    used_edges = set()
    result = [None] * 20
    
    def edge_key(u, v):
        return (min(u, v), max(u, v))
    
    for idx in order:
        k = idx + 1  # tree size
        tree_edges = trees[idx]
        
        if k == 1:
            # T1: single vertex, no edges. Pick any available vertex.
            for v in range(20):
                result[idx] = [v]
                break
            continue
        
        if k == 2:
            # T2: single edge (0,1). Find any unused edge.
            found = False
            for u in range(20):
                for w in range(u + 1, 20):
                    ek = edge_key(u, w)
                    if ek not in used_edges:
                        result[idx] = [u, w]
                        used_edges.add(ek)
                        found = True
                        break
                if found:
                    break
            if not found:
                result[idx] = None
            continue
        
        # For k >= 3, use backtracking to embed the tree
        tree_n = k
        tree_edge_list = tree_edges
        
        # Build adjacency for the tree
        tree_adj = [[] for _ in range(tree_n)]
        for u, v in tree_edge_list:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        
        # Get BFS order of tree vertices starting from 0
        visited = [False] * tree_n
        order_list = [0]
        visited[0] = True
        q = deque([0])
        while q:
            u = q.popleft()
            for v in tree_adj[u]:
                if not visited[v]:
                    visited[v] = True
                    order_list.append(v)
                    q.append(v)
        
        # Precompute for each tree vertex in BFS order, which neighbors come before it
        pos_in_order = [0] * tree_n
        for i, v in enumerate(order_list):
            pos_in_order[v] = i
        
        # Precompute parent and children in BFS tree
        parent = [-1] * tree_n
        children = [[] for _ in range(tree_n)]
        for i in range(1, tree_n):
            v = order_list[i]
            for nb in tree_adj[v]:
                if pos_in_order[nb] < pos_in_order[v]:
                    parent[v] = nb
                    break
            if parent[v] >= 0:
                children[parent[v]].append(v)
        
        # Count subtree sizes for MRV heuristic
        subtree_size = [1] * tree_n
        for i in range(tree_n - 1, -1, -1):
            v = order_list[i]
            for c in children[v]:
                subtree_size[v] += subtree_size[c]
        
        # Backtracking: assign host vertices to tree vertices in BFS order
        # Use MRV: at each step, pick the unassigned vertex with fewest available host vertices
        assignment = [None] * tree_n
        assigned_count = 0
        
        def get_available_hosts(tree_v):
            """Get list of valid host vertices for tree_v given current assignment"""
            avail = []
            for host_v in range(20):
                if any(a == host_v for a in assignment if a is not None):
                    continue
                valid = True
                for nb in tree_adj[tree_v]:
                    if assignment[nb] is not None:
                        ek = edge_key(assignment[nb], host_v)
                        if ek in used_edges:
                            valid = False
                            break
                if valid:
                    avail.append(host_v)
            return avail
        
        def backtrack():
            nonlocal assigned_count
            if time.time() > deadline:
                return False
            if assigned_count == tree_n:
                return True
            
            # MRV: pick unassigned vertex with fewest available hosts
            best_v = -1
            best_avail = None
            best_count = 21
            
            for v in range(tree_n):
                if assignment[v] is not None:
                    continue
                # Check if all parents are assigned (necessary condition)
                if parent[v] != -1 and assignment[parent[v]] is None:
                    continue
                avail = get_available_hosts(v)
                if len(avail) < best_count:
                    best_count = len(avail)
                    best_v = v
                    best_avail = avail
                    if best_count == 0:
                        return False
                    if best_count == 1:
                        break
            
            if best_v == -1:
                return False
            
            # Try each host vertex for best_v
            for host_v in best_avail:
                assignment[best_v] = host_v
                assigned_count += 1
                temp_edges = []
                for nb in tree_adj[best_v]:
                    if assignment[nb] is not None:
                        ek = edge_key(assignment[nb], host_v)
                        temp_edges.append(ek)
                        used_edges.add(ek)
                
                if backtrack():
                    return True
                
                # Undo
                for ek in temp_edges:
                    used_edges.discard(ek)
                assignment[best_v] = None
                assigned_count -= 1
        
        if backtrack():
            result[idx] = assignment[:]
        else:
            result[idx] = None
    
    return result
# EVOLVE-BLOCK-END
