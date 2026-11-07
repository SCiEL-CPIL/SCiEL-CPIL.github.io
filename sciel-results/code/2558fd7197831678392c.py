"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

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
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 115.0  # safety margin
    n = 20
    trees = instance_trees(n, 611)
    
    # Track used edges as a set of tuples (u, v) with u < v for efficiency
    used_edges = set()
    result = [None] * n
    
    # Embed trees from smallest to largest (k=1 to 20)
    # This is generally safer because small trees are easier to fit
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
            
        tree_edges = trees[k-1]
        
        if k == 1:
            # Find any vertex
            for v in range(n):
                # A single vertex doesn't use edges, so any vertex is fine
                # But we must ensure injectivity? No, trees are edge-disjoint.
                # Vertex reuse is allowed across different trees?
                # "Embedded trees must be edge-disjoint in K20."
                # "Entry k-1 is None or a list of k distinct int host vertices"
                # It does NOT say vertices must be distinct across different trees.
                # It says "k distinct int host vertices" for the *entry*.
                # So T1 can use vertex 0, and T2 can use vertex 0 and 1.
                # Edge disjointness is the only constraint.
                result[k-1] = [0]
                break
            continue
        
        if k == 2:
            # Find any unused edge
            found = False
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) not in used_edges:
                        result[k-1] = [u, v]
                        used_edges.add((u, v))
                        found = True
                        break
                if found:
                    break
            continue
        
        # For k >= 3, use randomized backtracking
        embedding = _embed_tree_backtrack(tree_edges, used_edges, n, deadline)
        if embedding:
            result[k-1] = embedding
            # Add edges to used set
            for (a, b) in tree_edges:
                u, v = embedding[a], embedding[b]
                if u > v:
                    u, v = v, u
                used_edges.add((u, v))
        else:
            # If backtracking fails, try a few simple random permutations as a fallback
            # (though backtracking should be robust)
            for _ in range(100):
                if time.time() > deadline:
                    break
                perm = list(range(n))
                random.shuffle(perm)
                valid = True
                for (a, b) in tree_edges:
                    u, v = perm[a], perm[b]
                    if u > v:
                        u, v = v, u
                    if (u, v) in used_edges:
                        valid = False
                        break
                if valid:
                    result[k-1] = perm[:k]
                    for (a, b) in tree_edges:
                        u, v = perm[a], perm[b]
                        if u > v:
                            u, v = v, u
                        used_edges.add((u, v))
                    break

    # Convert to proper format
    final_result = []
    for i in range(n):
        if result[i] is None:
            final_result.append(None)
        else:
            final_result.append(list(result[i]))
    
    return final_result

def _embed_tree_backtrack(tree_edges, used_edges, n, deadline):
    """
    Attempts to embed tree_edges into K_n avoiding used_edges.
    Returns a list of length k (host vertices for tree vertices 0..k-1) or None.
    """
    k = len(tree_edges) + 1
    if k == 1:
        return [0]
        
    # Build adjacency list for the tree
    tree_adj = [[] for _ in range(k)]
    for u, v in tree_edges:
        tree_adj[u].append(v)
        tree_adj[v].append(u)
        
    # Order tree vertices to improve backtracking efficiency
    # Start with a leaf, then grow. Or just use 0..k-1 with randomization.
    # Let's try to pick a starting vertex (leaf is good) and then do DFS.
    
    # We will map tree_vertex -> host_vertex
    mapping = [-1] * k
    used_host_verts = [False] * n
    
    # Pick a starting tree vertex. A leaf is usually good.
    start_tv = 0
    for v in range(k):
        if len(tree_adj[v]) == 1:
            start_tv = v
            break
            
    # To add randomness, we can shuffle the order of host vertices we try
    host_order = list(range(n))
    random.shuffle(host_order)
    
    def backtrack(tv):
        if time.time() > deadline:
            return False
            
        if tv == k:
            return True
            
        for hv in host_order:
            if used_host_verts[hv]:
                continue
                
            # Check if mapping tv to hv is valid with respect to already mapped neighbors
            valid = True
            for neighbor in tree_adj[tv]:
                if mapping[neighbor] != -1:
                    nhv = mapping[neighbor]
                    # Check edge (hv, nhv)
                    u, v = hv, nhv
                    if u > v:
                        u, v = v, u
                    if (u, v) in used_edges:
                        valid = False
                        break
            
            if valid:
                mapping[tv] = hv
                used_host_verts[hv] = True
                
                if backtrack(tv + 1):
                    return True
                    
                mapping[tv] = -1
                used_host_verts[hv] = False
                
    # The simple tv+1 order might be bad if the tree is not a path.
    # We should order the tree vertices such that we map connected components.
    # Let's create a list of tree vertices in an order that respects connectivity.
    # We can use a BFS/DFS order starting from start_tv.
    order = [start_tv]
    visited = {start_tv}
    queue = deque([start_tv])
    while queue:
        u = queue.popleft()
        for v in tree_adj[u]:
            if v not in visited:
                visited.add(v)
                order.append(v)
                queue.append(v)
                
    # Now mapping[i] corresponds to tree vertex order[i]
    # We need to adjust the backtracking to use this order.
    
    mapping_order = [-1] * k # mapping_order[i] is host vertex for order[i]
    
    def backtrack_idx(idx):
        if time.time() > deadline:
            return False
        if idx == k:
            return True
            
        tv = order[idx]
        
        for hv in host_order:
            if used_host_verts[hv]:
                continue
                
            valid = True
            for neighbor in tree_adj[tv]:
                # Find if neighbor is already mapped
                # Since we are using BFS order, neighbors might be mapped or not.
                # We need to check all mapped neighbors.
                # To do this efficiently, we can just check all tree vertices that are mapped.
                # Or, since k is small (<=20), we can just iterate through all mapped ones.
                pass # Placeholder, logic below is better
                
            # Better check:
            for j in range(idx):
                neighbor_tv = order[j]
                if neighbor_tv in tree_adj[tv]:
                    nhv = mapping_order[j]
                    u, v = hv, nhv
                    if u > v:
                        u, v = v, u
                    if (u, v) in used_edges:
                        valid = False
                        break
                        
            if valid:
                mapping_order[idx] = hv
                used_host_verts[hv] = True
                
                if backtrack_idx(idx + 1):
                    return True
                    
                mapping_order[idx] = -1
                used_host_verts[hv] = False
                
    if backtrack_idx(0):
        # Convert mapping_order back to original tree vertex indices
        final_mapping = [0] * k
        for i, tv in enumerate(order):
            final_mapping[tv] = mapping_order[i]
        return final_mapping
        
    return None
# EVOLVE-BLOCK-END
