"""Packing trees T1..T20 into K20 via backtracking with forward checking."""
import time
from random import Random

# Generate the fixed trees exactly as specified
def _instance_trees(n=20, seed=611):
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

def _build_adjacency(edges):
    """Build adjacency list from edge list."""
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    for v in adj:
        adj[v] = sorted(adj[v])
    return adj

def _embed_tree(tree_edges, n, used_edges_set, deadline, rng):
    """
    Find an embedding of the tree into K_n such that no edge conflicts with used_edges_set.
    Returns a list mapping tree-vertex -> host-vertex, or None if not found.
    Uses randomized backtracking.
    """
    if not tree_edges:
        # Single vertex tree (k=1): any free vertex works
        return [0]

    k = len(tree_edges) + 1  # number of vertices in tree
    adj = _build_adjacency(tree_edges)
    
    # Root the tree at the highest degree vertex
    root = max(range(k), key=lambda v: len(adj.get(v, [])))
    
    # Build parent/children structure
    parent = [-1] * k
    children = [[] for _ in range(k)]
    order = []  # BFS order
    visited = {root}
    queue = [root]
    while queue:
        v = queue.pop(0)
        order.append(v)
        for nb in adj.get(v, []):
            if nb not in visited:
                visited.add(nb)
                parent[nb] = v
                children[v].append(nb)
                queue.append(nb)
    
    # Backtracking state
    map_tree = [-1] * k
    used_hosts = set() # Track hosts used in THIS tree to ensure injectivity
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == len(order):
            return True
        v = order[idx]
        
        # Generate list of candidate hosts
        candidates = []
        for h in range(n):
            if h in used_hosts:
                continue
            p = parent[v]
            if p != -1:
                a = map_tree[p]
                if (a, h) in used_edges_set or (h, a) in used_edges_set:
                    continue
            candidates.append(h)
            
        if not candidates:
            return False
            
        # Shuffle candidates to randomize search path
        rng.shuffle(candidates)
        
        for h in candidates:
            # Forward checking: ensure children can be placed
            feasible = True
            for ch in children[v]:
                # Check if there is at least one free host for ch
                # that is not h, not in used_hosts, and not conflicting with h
                found = False
                for h2 in range(n):
                    if h2 == h or h2 in used_hosts:
                        continue
                    if (h, h2) in used_edges_set or (h2, h) in used_edges_set:
                        continue
                    found = True
                    break
                if not found:
                    feasible = False
                    break
            if not feasible:
                continue
                
            map_tree[v] = h
            used_hosts.add(h)
            if backtrack(idx + 1):
                return True
            used_hosts.remove(h)
            map_tree[v] = -1
            
        return False

    if backtrack(0):
        return map_tree[:]
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    n = 20
    trees = _instance_trees(n, 611)
    
    # Tree indices: 0 is T1 (0 edges), 1 is T2 (1 edge), ..., 19 is T20 (19 edges)
    # We want to maximize total edges.
    # Process trees in order of decreasing size (most constrained first)
    
    # We will use multiple restarts.
    # In each restart, we try to pack all trees.
    # We keep the best result (max total edges).
    
    best_result = [None] * n
    best_total_edges = 0
    restart_count = 0
    
    while time.time() < deadline:
        restart_count += 1
        restart_rng = Random(restart_count)
        
        # Order of trees: Largest first
        tree_indices = list(range(n))
        tree_indices.sort(key=lambda i: -i)
        
        used_edges_set = set()
        current_result = [None] * n
        total_edges = 0
        success = True
        
        for idx in tree_indices:
            if time.time() > deadline:
                success = False
                break
                
            tree_edges = trees[idx]
            if not tree_edges:
                # T1: single vertex
                current_result[idx] = [0]
                continue
                
            embedding = _embed_tree(tree_edges, n, used_edges_set, deadline, restart_rng)
            if embedding is None:
                success = False
                break
                
            # Add edges to used set
            for a, b in tree_edges:
                ha, hb = embedding[a], embedding[b]
                used_edges_set.add((ha, hb))
                used_edges_set.add((hb, ha))
            current_result[idx] = embedding
            total_edges += len(tree_edges)
            
        if success and total_edges > best_total_edges:
            best_total_edges = total_edges
            best_result = current_result[:]
            
        # If we achieved max edges, we can stop
        if best_total_edges == 190:
            break
            
    # Verify and convert to proper int lists
    final_result = []
    for i in range(n):
        if best_result[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_result[i]])
            
    return final_result
