import time
from random import Random
from collections import deque

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def decompose_graph(edges, n, deadline):
    if not edges:
        return []

    remaining = set(tuple(sorted(e)) for e in edges)
    adj = {u: set() for u in range(n)}
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    rng = Random(99)

    # Helper to find a cycle in the current graph via DFS
    def find_cycle(start):
        visited = {start}
        path = [start]
        # Use a randomized DFS to find a cycle
        # We want to find a cycle that returns to start
        # This is a standard DFS for cycle detection
        stack = [(start, iter(list(adj[start])))]
        # Shuffle neighbors for randomness
        # Actually, just iterate, but we can shuffle the list of neighbors for each node on first visit
        # To keep it simple and fast, we'll just do a standard DFS
        # If we find a neighbor in visited, and it's not the parent, we have a cycle.
        # But we want a simple cycle.
        # A standard DFS that tracks parent can find a back edge.
        
        parent = {start: -1}
        
        def dfs(u):
            for v in adj[u]:
                if v == start and len(path) >= 2:
                    # Found a cycle: path + [start]
                    # But path is global state, this is tricky.
                    # Let's use a different approach.
                    pass
            
        # Let's use a simpler iterative DFS that finds a cycle
        # We store the path in the stack
        stack = [(start, iter(list(adj[start])))]
        visited = {start}
        path = [start]
        
        # To ensure we find a cycle, we need to detect a back edge to start.
        # Since the graph is undirected, if we see a neighbor in visited that is not the parent, it's a cycle.
        # But we specifically want a cycle containing 'start' to extract it easily?
        # No, any cycle is fine.
        
        # Let's try to find the longest cycle? No, just any cycle is okay to start.
        # Actually, finding a long cycle is better.
        # Let's try to find a Hamiltonian cycle in the remaining graph? Too hard.
        # Let's find any cycle.
        
        # Simple DFS to find a cycle
        # Return the cycle as a list of vertices
        stack = [(start, iter(list(adj[start])))]
        visited = {start}
        path = [start]
        
        while stack:
            node, neighbors = stack[-1]
            advanced = False
            for nb in neighbors:
                e = (node, nb) if node < nb else (nb, node)
                if e not in remaining:
                    continue
                if nb in visited:
                    if nb == start and len(path) >= 3:
                        # Found a cycle
                        return path[:]
                    # If nb is in visited and not start, it's a cycle in the subgraph,
                    # but extracting it is harder. Let's just look for cycles containing start
                    # or use a general cycle finder.
                    # For now, let's just return if we find a cycle containing start.
                    # If we don't, we might miss cycles.
                    # A better approach: find ANY cycle.
                    # Let's modify the DFS to track parents.
                    pass
                else:
                    visited.add(nb)
                    path.append(nb)
                    stack.append((nb, iter(list(adj[nb]))))
                    advanced = True
                    break
            if not advanced:
                stack.pop()
                if path:
                    visited.discard(path.pop())
        
        return None

    # The above find_cycle is a bit weak because it only finds cycles containing 'start'
    # and might miss others.
    # Let's use a more robust cycle finder.
    # Find any cycle in the graph.
    def find_any_cycle():
        visited = set()
        parent = {}
        
        def dfs(u, p):
            visited.add(u)
            parent[u] = p
            for v in adj[u]:
                if v not in visited:
                    cycle = dfs(v, u)
                    if cycle:
                        return cycle
                elif v != p:
                    # Found a back edge u-v
                    # Reconstruct cycle
                    cycle = [v, u]
                    cur = u
                    while cur != v:
                        cur = parent[cur]
                        cycle.append(cur)
                    cycle.reverse() # v -> ... -> u -> v (but reversed)
                    # Actually, the cycle is v -> ... -> u -> v
                    # The path is v, ..., u. So cycle is [v, ..., u]
                    # The list built is [v, u, ..., v]
                    # Let's trace carefully.
                    # u is connected to v (visited).
                    # parent chain: u -> ... -> v
                    # So cycle is v, ..., u, v
                    # We have list: [v, u]
                    # cur = u
                    # loop: cur = parent[u] ... until cur == v
                    # append cur
                    # So list becomes [v, u, parent[u], ... , v]
                    # This is the cycle.
                    return cycle
            return None
        
        for i in range(n):
            if i not in visited:
                cycle = dfs(i, -1)
                if cycle:
                    # Ensure unique vertices in cycle (except start/end)
                    # The DFS cycle detection guarantees simple cycle
                    return cycle
        return None

    while remaining:
        if time.time() > deadline:
            break
            
        # Check if any cycle exists
        # If all degrees are < 2, no cycle exists
        has_cycle = False
        for u in range(n):
            if len(adj[u]) >= 2:
                has_cycle = True
                break
        if not has_cycle:
            break
            
        cycle = find_any_cycle()
        
        if not cycle:
            # Should not happen if has_cycle is True, but safeguard
            break
            
        # Extract the cycle
        # cycle is a list of vertices [v0, v1, ..., vk] where (vk, v0) is an edge
        # But wait, the DFS returns [v, u, ..., v]
        # So the edges are (v, u), (u, next), ..., (last, v)
        # The list has the first vertex repeated at the end.
        # Example: [0, 1, 2, 0]
        # Edges: (0,1), (1,2), (2,0)
        
        # Remove edges from remaining and adj
        for i in range(len(cycle) - 1):
            u = cycle[i]
            v = cycle[i+1]
            e = (u, v) if u < v else (v, u)
            remaining.discard(e)
            adj[u].discard(v)
            adj[v].discard(u)
            
        # Add cycle to pieces
        # The piece is the list of vertices without the repeated start
        # e.g. [0, 1, 2]
        pieces.append(cycle[:-1])
        
        if time.time() > deadline:
            break

    # Add remaining edges as singletons
    for e in remaining:
        if time.time() > deadline:
            break
        pieces.append([e[0], e[1]])
        
    return pieces

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for n, p, seed in specs:
        if time.time() > deadline:
            # Fallback: just return all edges as singletons
            edges = instance(n, p, seed)
            results.append([list(ed) for ed in edges])
            continue
            
        edges = instance(n, p, seed)
        pieces = decompose_graph(edges, n, deadline)
        results.append(pieces)
        
    while len(results) < 3:
        results.append([])
        
    return results[:3]

if __name__ == "__main__":
    construct()
