import time
import random
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
    deadline = start_time + 110.0  # 10s margin
    
    n = 20
    trees = instance_trees(n, 611)
    
    # Precompute tree adjacency and degrees
    tree_adj = []
    tree_degrees = []
    for i in range(n):
        k = i + 1
        adj = [[] for _ in range(k)]
        for u, v in trees[i]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
        tree_degrees.append([len(adj[v]) for v in range(k)])
        
    # Order: Largest to Smallest
    order = list(range(n - 1, -1, -1))
    
    def try_embed_tree(t_idx, used_edges, host_degrees):
        """
        Attempts to embed tree t_idx into K20 avoiding used_edges.
        Returns list of host vertices for tree vertices 0..k-1 or None.
        """
        k = t_idx + 1
        if k == 1:
            # Pick any vertex with available degree (all have >= 1 available if < 19 used)
            # Actually T1 has 0 edges, so any vertex works.
            # We just need to return a valid vertex index.
            # To be consistent, let's pick one with min used degree.
            min_deg = min(host_degrees)
            candidates = [v for v in range(n) if host_degrees[v] == min_deg]
            return [random.choice(candidates)]
        
        adj = tree_adj[t_idx]
        
        # Try multiple random roots and orders
        for attempt in range(20):
            if time.time() > deadline:
                return None
            
            # Random root
            root = random.randint(0, k - 1)
            # Random host for root
            root_host = random.randint(0, n - 1)
            
            # Check if root_host has enough remaining degree
            if host_degrees[root_host] + tree_degrees[t_idx][root] > n - 1:
                continue
            
            # Mapping: tree_vertex -> host_vertex
            mapping = {root: root_host}
            # Queue for BFS
            queue = [root]
            success = True
            # We need to track which edges are used in this embedding to check against global used_edges
            # But we also need to ensure we don't use the same host edge twice within this tree (which is impossible for a tree)
            # The main constraint is global used_edges.
            
            # To speed up, we can just do a greedy DFS
            # Since k is small (<=20), we can do a simple recursive DFS with backtracking
            # But to keep it fast, we'll try a few random DFS orders.
            
            # Let's use a simple randomized DFS
            def dfs(v, parent):
                if time.time() > deadline:
                    return False
                for u in adj[v]:
                    if u == parent:
                        continue
                    # Find a host for u
                    # u must be connected to mapping[v] via an available edge
                    hv = mapping[v]
                    candidates = []
                    for nh in range(n):
                        if nh == hv:
                            continue
                        e = (min(hv, nh), max(hv, nh))
                        if e not in used_edges:
                            # Check degree constraint for nh
                            # The tree vertex u has degree tree_degrees[t_idx][u]
                            # The host nh will take 1 edge for (v,u) and potentially more for its children.
                            # Simple check: nh must have remaining degree >= 1
                            if host_degrees[nh] + 1 <= n - 1:
                                # Also, if u has children, nh needs more degree.
                                # For a simple greedy, we just check if it's available.
                                # We'll rely on backtracking if it fails later.
                                candidates.append(nh)
                    
                    if not candidates:
                        return False
                    
                    # Randomize candidates
                    random.shuffle(candidates)
                    
                    for nh in candidates:
                        mapping[u] = nh
                        if dfs(u, v):
                            return True
                        del mapping[u]
                    
                    return False
                return True
            
            if dfs(root, -1):
                # Check if all vertices are mapped
                if len(mapping) == k:
                    # Construct result list
                    result = [0] * k
                    for tv, hv in mapping.items():
                        result[tv] = hv
                    return result
        return None

    best_solution = None
    
    while time.time() < deadline:
        used_edges = set()
        host_degrees = [0] * n
        solution = [None] * n
        success = True
        
        for t_idx in order:
            k = t_idx + 1
            emb = try_embed_tree(t_idx, used_edges, host_degrees)
            if emb is None:
                success = False
                break
            
            solution[t_idx] = emb
            # Update used_edges and host_degrees
            for u, v in trees[t_idx]:
                hu, hv = emb[u], emb[v]
                e = (min(hu, hv), max(hu, hv))
                used_edges.add(e)
                host_degrees[hu] += 1
                host_degrees[hv] += 1
        
        if success:
            # Verify
            if len(used_edges) == 190:
                return solution
    
    # If we run out of time, return best effort or None
    # The problem states "Entry k-1 is None or a list...".
    # If we can't find a full packing, we should return what we can?
    # "The raw value is the total number of embedded tree edges".
    # So we should return the best partial packing found.
    
    # Re-run one last time to get a partial solution if the loop above failed to complete
    # Actually, the loop above breaks on failure. We should keep track of the best partial.
    
    # Let's modify the loop to track best partial.
    
    best_val = 0
    best_sol = [None] * n
    
    while time.time() < deadline:
        used_edges = set()
        host_degrees = [0] * n
        solution = [None] * n
        current_val = 0
        
        for t_idx in order:
            k = t_idx + 1
            emb = try_embed_tree(t_idx, used_edges, host_degrees)
            if emb is None:
                break
            
            solution[t_idx] = emb
            current_val += k - 1
            for u, v in trees[t_idx]:
                hu, hv = emb[u], emb[v]
                e = (min(hu, hv), max(hu, hv))
                used_edges.add(e)
                host_degrees[hu] += 1
                host_degrees[hv] += 1
        
        if current_val > best_val:
            best_val = current_val
            best_sol = solution[:]
            
        if best_val == 190:
            return best_sol

    return best_sol

if __name__ == "__main__":
    pass
