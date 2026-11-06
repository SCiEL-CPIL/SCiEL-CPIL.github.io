import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = 120.0  # Keep a safety margin from 150s
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(n, edges):
        # Build adjacency list
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        remaining_edges = set(edges)
        
        # Helper to find a cycle in the current graph
        def find_cycle(adj, remaining_edges):
            # We want to find a cycle. To make it long, we use a heuristic DFS.
            # Start from a vertex with max degree.
            if not remaining_edges:
                return None
            
            # Find vertex with max degree in remaining graph
            degs = {v: len([e for e in remaining_edges if v in e]) for v in adj}
            # Actually, simpler: just count neighbors in adj that are still connected?
            # No, adj is updated. Let's just pick a vertex from remaining edges.
            # To be efficient, let's just pick the first vertex we see in remaining_edges.
            # But to get long cycles, starting at high degree is better.
            
            # Let's build a temporary degree map for the remaining graph
            temp_deg = defaultdict(int)
            for u, v in remaining_edges:
                temp_deg[u] += 1
                temp_deg[v] += 1
            
            if not temp_deg:
                return None
                
            start_vertex = max(temp_deg, key=lambda x: temp_deg[x])
            
            # DFS with backtracking to find a cycle
            # We use a stack to keep track of the path.
            # To favor long cycles, we try to extend the path as much as possible
            # before closing it, but we must ensure we can close it.
            # A simpler heuristic: Just find ANY cycle. The greedy removal will
            # naturally tend to use up high-degree areas first if we start there.
            
            visited = {start_vertex}
            stack = [(start_vertex, [start_vertex])]
            parent = {start_vertex: None}
            
            # Standard DFS to find any cycle
            while stack:
                node, path = stack[-1]
                # Get neighbors in current adj
                neighbors = list(adj[node])
                # Sort by degree descending to encourage long paths?
                neighbors.sort(key=lambda x: temp_deg.get(x, 0), reverse=True)
                
                advanced = False
                for neighbor in neighbors:
                    if (min(node, neighbor), max(node, neighbor)) not in remaining_edges:
                        continue
                    
                    if neighbor not in visited:
                        visited.add(neighbor)
                        parent[neighbor] = node
                        stack.append((neighbor, path + [neighbor]))
                        advanced = True
                        break
                    elif neighbor != parent.get(node, None) and neighbor in [p for p in path]:
                        # Found a cycle
                        # Extract cycle from path
                        idx = path.index(neighbor)
                        cycle = path[idx:]
                        return cycle
                
                if not advanced:
                    stack.pop()
                    # Note: We don't remove from visited here in standard DFS for cycle detection
                    # because we are looking for *any* cycle in the connected component.
                    # However, since we are modifying the graph, we are effectively
                    # doing a DFS on the current subgraph.
                    # If we pop, we backtrack.
            
            return None

        while remaining_edges:
            if time.time() > deadline - 5:
                # Fallback: return all remaining edges as single pieces
                for u, v in remaining_edges:
                    pieces.append([u, v])
                break
            
            cycle = find_cycle(adj, remaining_edges)
            
            if cycle:
                # Remove edges of the cycle
                for i in range(len(cycle) - 1):
                    u, v = cycle[i], cycle[i+1]
                    edge = tuple(sorted((u, v)))
                    remaining_edges.discard(edge)
                    adj[u].discard(v)
                    adj[v].discard(u)
                pieces.append(cycle)
            else:
                # No cycles left, add all remaining edges as single pieces
                for u, v in remaining_edges:
                    pieces.append([u, v])
                break
        
        return pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for n, p, seed in specs:
        if time.time() > deadline - 10:
            # Not enough time, return trivial solution for remaining
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(n, p, seed)
        pieces = decompose(n, edges)
        results.append(pieces)

    return results
