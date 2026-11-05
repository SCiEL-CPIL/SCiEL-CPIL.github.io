"""Decompositions of three fixed graphs into few cycles and edges (Erdős-Gallai)."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict, deque
import heapq
from random import Random

def construct():
    # Generate the three instance graphs
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    results = []
    for spec in specs:
        edges = instance(*spec)
        n = spec[0]
        result = decompose_graph(n, edges)
        results.append(result)
    
    return results

def decompose_graph(n, edges):
    """
    Decompose the undirected graph into minimum number of cycles and edges.
    
    Strategy:
    1. Try to find cycles greedily using DFS, prioritizing long cycles.
    2. Use edge-disjoint cycle finding.
    3. Remaining edges become individual pieces.
    
    A more sophisticated approach:
    - Repeatedly find a cycle in the remaining graph using DFS.
    - To minimize pieces, we want to maximize the number of edges covered by cycles.
    - Each cycle of length k uses k edges and 1 piece, vs k pieces if all edges separate.
    - So we want to pack as many edges into cycles as possible.
    
    Greedy approach:
    - While there's a cycle in the remaining graph, find one and remove its edges.
    - Use DFS to find cycles; prefer longer cycles to cover more edges per piece.
    """
    if not edges:
        return []
    
    # Build adjacency structure
    edge_set = set()
    adj = defaultdict(set)
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    remaining_edges = set(edge_set)
    remaining_adj = defaultdict(set)
    for u, v in edges:
        remaining_adj[u].add(v)
        remaining_adj[v].add(u)
    
    def find_cycle(adj, n):
        """Find a cycle in the undirected graph using DFS. Returns list of vertices in cycle or None."""
        visited = [False] * n
        parent = [-1] * n
        parent_edge = [None] * n
        
        def dfs(start):
            stack = [(start, iter(adj[start]))]
            visited[start] = True
            while stack:
                node, it = stack[-1]
                advanced = False
                for neighbor in it:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        parent[neighbor] = node
                        parent_edge[neighbor] = (node, neighbor)
                        stack.append((neighbor, iter(adj[neighbor])))
                        advanced = True
                        break
                    elif neighbor != parent[node]:
                        # Found a back edge, reconstruct cycle
                        # Cycle: node -> ... -> neighbor -> node
                        cycle = [node]
                        curr = node
                        while curr != neighbor:
                            curr = parent[curr]
                            cycle.append(curr)
                        # The cycle is: node, ..., neighbor, node (closing edge)
                        # But we need to return vertices in order
                        # Let's reconstruct properly
                        # Path from node up to neighbor via parents
                        path = [node]
                        curr = node
                        while curr != neighbor:
                            curr = parent[curr]
                            path.append(curr)
                        # path is [node, ..., neighbor]
                        # The cycle is path + [node] but we just need the vertices
                        # Actually the cycle vertices are: neighbor, ..., node
                        # Let's return the cycle as a list of vertices
                        return path
                    # else: visited and is parent, skip
                if not advanced:
                    stack.pop()
            return None
        
        for start in range(n):
            if not visited[start] and adj[start]:
                cycle = dfs(start)
                if cycle and len(cycle) >= 3:
                    return cycle
        
        return None
    
    # Greedy: repeatedly find and remove cycles
    # To prefer longer cycles, we can try to find the longest cycle, but that's expensive.
    # Instead, let's just find any cycle and remove it. The order might affect the result.
    
    # Better approach: try to find cycles, preferring to use edges that are in high-degree vertices
    # or just do a simple greedy
    
    max_iterations = 1000
    iteration = 0
    while remaining_edges and iteration < max_iterations:
        iteration += 1
        # Build adjacency from remaining edges
        rem_adj = defaultdict(set)
        for u, v in remaining_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        # Find a cycle
        cycle = find_cycle(rem_adj, n)
        if cycle is None:
            break
        
        # Verify the cycle is valid: all consecutive pairs (and last-first) are in remaining_edges
        cycle_edges = []
        valid = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            e = (min(u,v), max(u,v))
            if e not in remaining_edges:
                valid = False
                break
            cycle_edges.append(e)
        
        if not valid or len(cycle) < 3:
            # Try a different approach: use a simpler cycle finder
            # Find any cycle using a different method
            cycle = find_cycle_simple(rem_adj, n, remaining_edges)
            if cycle is None:
                break
            cycle_edges = []
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i+1) % len(cycle)]
                e = (min(u,v), max(u,v))
                if e not in remaining_edges:
                    valid = False
                    break
                cycle_edges.append(e)
            if not valid:
                break
        
        # Remove cycle edges
        for e in cycle_edges:
            remaining_edges.discard(e)
        
        # Add cycle as a piece
        pieces.append(cycle)
    
    # Remaining edges become individual pieces
    for u, v in remaining_edges:
        pieces.append([u, v])
    
    return pieces

def find_cycle_simple(adj, n, remaining_edges):
    """Find a simple cycle using BFS/DFS with explicit edge tracking."""
    for start in range(n):
        if not adj[start]:
            continue
        # BFS to find a cycle
        visited = {start: None}  # node -> parent
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if neighbor == visited.get(node):
                    continue
                if neighbor in visited:
                    # Found a cycle
                    # Reconstruct cycle from node back to neighbor
                    cycle = [node]
                    curr = node
                    while curr != neighbor:
                        curr = visited[curr]
                        cycle.append(curr)
                    if len(cycle) >= 3:
                        # Verify all edges exist
                        valid = True
                        for i in range(len(cycle)):
                            u = cycle[i]
                            v = cycle[(i+1) % len(cycle)]
                            e = (min(u,v), max(u,v))
                            if e not in remaining_edges:
                                valid = False
                                break
                        if valid:
                            return cycle
                else:
                    visited[neighbor] = node
                    queue.append(neighbor)
    return None
# EVOLVE-BLOCK-END
