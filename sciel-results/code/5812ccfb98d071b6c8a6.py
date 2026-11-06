"""Decompose fixed random graphs into few cycles and edges (Erdős-Gallai style)."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict
from itertools import combinations

def construct():
    # Deadline guard: kill before 150s wall-clock.
    deadline = time.time() + 130  # safety margin

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, n, time_budget):
        """
        Find a decomposition of the edge set into cycles (length >= 3) and single edges.
        Minimize total number of pieces.
        
        Strategy: iteratively find edge-disjoint cycles. Use a combination of:
        1. Find cycles via DFS/backtracking with pruning.
        2. Greedy: repeatedly find the shortest available cycle (or any cycle) to maximize count.
        3. Use networkx-like BFS/DFS to find cycles in the remaining graph.
        
        Since we want to minimize pieces, we want to maximize the number of cycles.
        Each cycle of length k covers k edges in 1 piece, saving k-1 pieces vs individual edges.
        So we want as many cycles as possible, and cycles that are edge-disjoint.
        
        Approach: 
        - Maintain adjacency structure for remaining edges.
        - Repeatedly find a cycle in the remaining graph.
        - Remove its edges.
        - Continue until no more cycles can be found.
        - Remaining edges are single-edge pieces.
        
        To find cycles efficiently, we can use a modified DFS that looks for back edges.
        """
        # Build adjacency list
        adj = defaultdict(set)
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
        
        pieces = []
        remaining_edges = list(edge_set)
        
        # We'll iteratively find cycles
        max_iterations = len(remaining_edges) + 10  # safety
        iteration = 0
        
        while iteration < max_iterations:
            if time.time() > time_budget:
                break
            iteration += 1
            
            if not remaining_edges:
                break
            
            # Try to find a cycle in the current remaining edge set
            cycle = find_cycle(remaining_edges, adj, n, time_budget)
            
            if cycle is None:
                # No more cycles found; break
                break
            
            # Remove cycle edges from remaining
            cycle_edges = set()
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                cycle_edges.add(e)
            
            # Verify all cycle edges are in remaining
            for e in cycle_edges:
                if e not in remaining_edges:
                    # This shouldn't happen if we're careful, but break
                    break
            else:
                for e in cycle_edges:
                    remaining_edges.remove(e)
                pieces.append(cycle)
        
        # Add remaining edges as single-edge pieces
        for u, v in remaining_edges:
            pieces.append([u, v])
        
        return pieces
    
    def find_cycle(remaining_edges, adj, n, time_budget):
        """
        Find a cycle in the graph defined by remaining_edges.
        Returns a list of vertices forming the cycle, or None.
        """
        if len(remaining_edges) < 3:
            return None
        
        # Build local adjacency for remaining edges
        local_adj = defaultdict(list)
        for u, v in remaining_edges:
            local_adj[u].append(v)
            local_adj[v].append(u)
        
        # Try to find a cycle using DFS from each vertex
        # We look for a back edge in DFS
        visited = {}  # vertex -> (parent, state) where state: 1=visiting, 2=done
        parent = {}
        
        for start in range(n):
            if start not in local_adj:
                continue
            
            # Reset for this DFS attempt
            visited = {start: 1}
            parent = {start: -1}
            
            stack = [start]
            found_cycle = None
            
            while stack:
                if time.time() > time_budget:
                    return None
                node = stack[-1]
                neighbors = local_adj.get(node, [])
                found_next = False
                for nb in neighbors:
                    if nb not in visited:
                        visited[nb] = 1
                        parent[nb] = node
                        stack.append(nb)
                        found_next = True
                        break
                    elif visited[nb] == 1 and nb != parent.get(node, -1):
                        # Found a back edge: node -> nb where nb is still being visited
                        # Extract cycle
                        cycle = [node]
                        cur = node
                        while cur != nb:
                            cur = parent[cur]
                            cycle.append(cur)
                        cycle.append(nb)
                        # Reverse to get proper order
                        cycle.reverse()
                        # Remove duplicate if any
                        if cycle[0] == cycle[-1]:
                            cycle = cycle[:-1]
                        found_cycle = cycle
                        break
                if found_cycle:
                    return found_cycle
                if found_next:
                    continue
                # Backtrack
                visited[node] = 2
                stack.pop()
        
        # If DFS didn't find a cycle, try a different approach:
        # Look for any triangle or 4-cycle
        # For small graphs, we can try all triples
        if n <= 35:
            # Try to find a triangle
            edge_set = set(remaining_edges)
            for u in range(n):
                if u not in local_adj:
                    continue
                for v in local_adj[u]:
                    if v <= u:
                        continue
                    for w in local_adj[v]:
                        if w <= v:
                            continue
                        if (u, w) in edge_set or (w, u) in edge_set:
                            # Triangle u-v-w
                            return [u, v, w]
        
        return None
    
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        if time.time() > deadline - 5:
            # Not enough time left; return empty pieces (invalid but better than timeout)
            # Actually, let's just return all edges as single pieces for remaining
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        # Time budget per graph: leave some for others
        remaining_time = deadline - time.time() - 5
        per_graph_time = remaining_time / max(1, len(specs) - len(results))
        per_graph_time = max(1.0, min(per_graph_time, 40.0))
        
        pieces = decompose(edges, n, time.time() + per_graph_time)
        results.append(pieces)
    
    return results

# EVOLVE-BLOCK-END
