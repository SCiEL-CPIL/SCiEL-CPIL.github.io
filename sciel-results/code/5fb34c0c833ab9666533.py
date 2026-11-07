from random import Random
import sys

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
    trees = instance_trees(20, 611)
    n = 20
    
    # Track which K20 edges are used
    # used_edges[u][v] = True if edge (u,v) is used
    used_edges = [[False] * n for _ in range(n)]
    
    # Result: list of 20 entries, entry k-1 is None or list of k host vertices
    result = [None] * 20
    
    # Process trees in order of decreasing size (harder ones first)
    tree_indices = list(range(20))
    tree_indices.sort(key=lambda i: len(trees[i]), reverse=True)
    
    # Helper to check if edge (u,v) is available
    def is_edge_available(u, v):
        if u == v:
            return False
        if u > v:
            u, v = v, u
        return not used_edges[u][v]

    # Helper to mark edge (u,v) as used
    def mark_edge_used(u, v):
        if u > v:
            u, v = v, u
        used_edges[u][v] = True
        used_edges[v][u] = True

    # Helper to unmark edge (u,v)
    def unmark_edge_used(u, v):
        if u > v:
            u, v = v, u
        used_edges[u][v] = False
        used_edges[v][u] = False

    for idx in tree_indices:
        k = idx + 1  # Tree T_k has k vertices
        if k == 1:
            result[idx] = None
            continue
        
        tree_edges = trees[idx]
        
        # Build adjacency list for the tree
        adj = [[] for _ in range(k)]
        degree = [0] * k
        for (a, b) in tree_edges:
            adj[a].append(b)
            adj[b].append(a)
            degree[a] += 1
            degree[b] += 1
            
        # Order vertices for backtracking: high degree first
        # We will assign vertices in this order
        order = sorted(range(k), key=lambda x: -degree[x])
        
        # Map from position in order to original vertex id
        # pos 0 is the highest degree vertex, etc.
        
        # Backtracking state:
        # assigned_host[v] = host vertex index or -1
        assigned_host = [-1] * k
        # We also need to know which tree vertices have been assigned
        # We iterate through 'order'
        
        # To speed up, we fix the first vertex (highest degree) to host 0.
        # This breaks symmetry.
        # However, we must ensure that the rest of the tree can be embedded.
        
        def solve(pos):
            if pos == k:
                return True
            
            v = order[pos]
            
            # Get available host vertices for v
            # If pos == 0, we can fix v to 0 (symmetry breaking)
            # But wait, if we fix v to 0, we might miss solutions where v is not at 0?
            # No, K20 is vertex transitive. Any embedding can be rotated to put any chosen vertex at 0.
            # So fixing the first vertex to 0 is safe and reduces search space by 20x.
            
            candidates = []
            if pos == 0:
                candidates = [0]
            else:
                # Candidate must be adjacent to already assigned parents
                # Find all already assigned neighbors of v in the tree
                # Since we are building a tree, v might have multiple assigned neighbors if we are not doing BFS/DFS strictly?
                # Actually, if we pick an order that is a valid traversal (e.g., BFS or DFS order), each vertex (except root) has exactly one parent assigned before it.
                # The sorted-by-degree order is NOT necessarily a traversal order.
                # We need a traversal order that respects the tree structure.
                
                # Let's generate a BFS/DFS order starting from the highest degree node.
                # This ensures each node (except root) has exactly one parent in the order.
                
                pass # We need to change the 'order' logic
            
            # Re-evaluating 'order':
            # 1. Pick root (highest degree).
            # 2. Generate BFS/DFS sequence from root.
            # 3. This ensures each vertex has exactly one parent assigned before it.
            
            return False

        # Let's restructure the backtracking to use a proper traversal order.
        
        # 1. Select root: vertex with max degree.
        root = max(range(k), key=lambda x: degree[x])
        
        # 2. Generate a traversal order (BFS) starting from root.
        #    This order ensures that when we place a vertex, its parent is already placed.
        #    We can further sort children at each level by degree (descending) to prune earlier.
        
        # Build parent-child structure from root
        parent = [-1] * k
        children = [[] for _ in range(k)]
        visited = [False] * k
        queue = [root]
        visited[root] = True
        traversal_order = []
        
        # We will perform a BFS but store the order.
        # To optimize, we can sort children by degree descending before adding to queue?
        # Actually, for backtracking, it's better to have a fixed order.
        
        # Let's do a simple BFS to get a valid order.
        bfs_queue = [root]
        traversal_order = []
        while bfs_queue:
            u = bfs_queue.pop(0)
            traversal_order.append(u)
            # Sort neighbors by degree descending to prioritize constrained nodes
            neighbors = sorted(adj[u], key=lambda x: -degree[x])
            for w in neighbors:
                if not visited[w]:
                    visited[w] = True
                    parent[w] = u
                    children[u].append(w)
                    bfs_queue.append(w)
        
        # Now, traversal_order[0] is root.
        # For any i > 0, parent[traversal_order[i]] is in traversal_order[0..i-1].
        
        def backtrack(pos):
            if pos == k:
                return True
            
            v = traversal_order[pos]
            
            # Determine candidate host vertices for v
            if pos == 0:
                # Fix root to 0 to break symmetry
                candidates = [0]
            else:
                p = parent[v]
                hp = assigned_host[p]
                # Candidates are all host vertices adjacent to hp
                # and not used for other tree vertices
                # and edge (hp, h) is available in K20
                candidates = []
                for h in range(n):
                    if h == hp:
                        continue
                    if not is_edge_available(hp, h):
                        continue
                    # Check if h is already used by another tree vertex
                    # assigned_host contains -1 for unassigned, and host indices for assigned.
                    # We need to ensure h is not in assigned_host[0..pos-1]
                    # Since assigned_host is a list, checking 'in' is O(k). k is small (<=20).
                    # But we can maintain a set of used host vertices for the current tree.
                    # Let's pass a set or use a boolean array.
                    # For simplicity and speed, let's use a boolean array 'used_host' of size n.
                    
                    if used_host[h]:
                        continue
                    candidates.append(h)
            
            # Sort candidates? Maybe by degree of h in K20 (all 19) or remaining capacity?
            # All host vertices are symmetric in K20 except for used edges.
            # We can try candidates in random order or fixed order.
            
            for h in candidates:
                # Try assigning v to h
                assigned_host[v] = h
                used_host[h] = True
                # Mark edge (hp, h) as used if pos > 0
                if pos > 0:
                    p = parent[v]
                    hp = assigned_host[p]
                    mark_edge_used(hp, h)
                
                if backtrack(pos + 1):
                    return True
                
                # Undo
                assigned_host[v] = -1
                used_host[h] = False
                if pos > 0:
                    p = parent[v]
                    hp = assigned_host[p]
                    unmark_edge_used(hp, h)
            
            return False

        # Initialize state for this tree
        assigned_host = [-1] * k
        used_host = [False] * n
        
        # Try to solve
        if backtrack(0):
            result[idx] = list(assigned_host)
        else:
            result[idx] = None
            
    return result
