import time
import random
from collections import defaultdict

def instance_graph(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def decompose_once(edges, rng):
    remaining = set(edges)
    pieces = []
    
    while remaining:
        # Build adjacency list
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
            
        # Find a cycle
        cycle = None
        # Try to find a cycle starting from a random node
        # We can try multiple starts if the first one fails or yields a short cycle
        # But for now, just find one.
        
        # Heuristic: Pick a start node with high degree
        start_nodes = list(adj.keys())
        if not start_nodes:
            break
            
        # Shuffle start nodes to randomize
        rng.shuffle(start_nodes)
        
        for start in start_nodes:
            cycle = find_cycle_heuristic(adj, remaining, start, rng)
            if cycle:
                break
                
        if not cycle:
            # No cycle found, so all remaining edges are isolated or form a forest?
            # If no cycle, we just add all remaining edges as single pieces.
            for e in remaining:
                pieces.append(list(e))
            break
            
        # Remove cycle edges
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edge = tuple(sorted((u, v)))
            if edge in remaining:
                remaining.remove(edge)
                
        pieces.append(cycle)
        
    return pieces

def find_cycle_heuristic(adj, remaining, start, rng):
    # DFS to find a cycle
    # We want to find a cycle that is reasonably long.
    # We use a stack for DFS.
    # path: list of nodes
    # parent: dict node -> parent
    
    path = [start]
    parent = {start: None}
    visited_in_path = {start}
    
    # We will use a recursive or iterative DFS.
    # Iterative is safer for deep recursion.
    
    # To encourage long cycles, we prefer neighbors with higher degree.
    # But we also need to ensure we can close the cycle.
    
    # Simple DFS:
    stack = [(start, iter(sorted(adj[start], key=lambda x: -len(adj[x]))))]
    # Wait, sorting every time is expensive. Let's just shuffle and sort once per node visit?
    # Or just shuffle.
    
    # Let's try a simple approach:
    # 1. Pick start.
    # 2. DFS.
    # 3. If we hit a node in `visited_in_path`, we have a cycle.
    # 4. Extract cycle.
    
    # To make it better, we can try to find the *longest* cycle by backtracking.
    # But that's complex. Let's stick to "find a cycle" with randomization.
    
    # Actually, a very effective trick is to find a cycle, then check if we can extend it?
    # No, the edges are removed.
    
    # Let's just do a standard DFS that finds a cycle.
    # The randomness in `decompose_once` (via rng) will handle the "restart" aspect.
    
    # We'll use a recursive DFS with a limit on depth to avoid infinite loops or excessive time.
    # Max depth is N.
    
    def dfs(u, path, parent):
        for v in adj[u]:
            if v == parent[u]:
                continue
            if v in path:
                # Found a cycle
                # Cycle is path from v to u, plus edge u-v
                idx = path.index(v)
                return path[idx:]
            if v not in parent:
                parent[v] = u
                path.append(v)
                cycle = dfs(v, path, parent)
                if cycle:
                    return cycle
                path.pop()
                del parent[v]
        return None
        
    # We need to be careful with the `path` and `parent` state in recursion.
    # The above implementation modifies `path` and `parent` in place.
    # This is correct for DFS backtracking.
    
    # However, the order of neighbors matters.
    # We can shuffle the neighbors of each node to randomize the path taken.
    
    # Let's create a shuffled adjacency list.
    adj_shuffled = {u: list(v) for u, v in adj.items()}
    for u in adj_shuffled:
        rng.shuffle(adj_shuffled[u])
        
    # Use the shuffled adj in DFS
    # We need to pass it to dfs or use a closure.
    
    # Let's rewrite dfs to use adj_shuffled.
    
    def dfs2(u, path, parent):
        for v in adj_shuffled[u]:
            if v == parent[u]:
                continue
            if v in path:
                idx = path.index(v)
                return path[idx:]
            if v not in parent:
                parent[v] = u
                path.append(v)
                cycle = dfs2(v, path, parent)
                if cycle:
                    return cycle
                path.pop()
                del parent[v]
        return None

    cycle = dfs2(start, [start], {start: None})
    return cycle

def construct():
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []
    
    # Total time budget 150s.
    # We have 3 graphs. Let's allocate ~45s each.
    time_per_graph = 45.0
    
    for n, p, seed in specs:
        edges = instance_graph(n, p, seed)
        best_pieces = None
        best_count = float('inf')
        
        deadline = start_time + time_per_graph
        
        # Run multiple restarts
        while time.time() < deadline:
            rng = random.Random()
            pieces = decompose_once(edges, rng)
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = pieces
                
            # Early stopping if we reach a very good score
            # Lower bound: E / 3 (if all cycles are length 3) or E/avg.
            # For E=100, count=30 is good.
            if best_count < 25:
                break
                
        results.append(best_pieces)
        
    return results
