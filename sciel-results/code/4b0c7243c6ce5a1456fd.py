import sys
sys.setrecursionlimit(10000)
import random

def construct():
    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycle_fast(adj, n, start):
        """Find a cycle starting from 'start' using iterative DFS to avoid recursion limits."""
        if not adj[start]:
            return None
        
        stack = [(start, iter(adj[start]))]
        path = [start]
        visited = {start: True}
        parent = {start: None}
        
        while stack:
            node, neighbors = stack[-1]
            found_next = False
            for neighbor in neighbors:
                if neighbor == start and len(path) >= 3:
                    return path[:]
                if not visited.get(neighbor, False):
                    visited[neighbor] = True
                    parent[neighbor] = node
                    stack.append((neighbor, iter(adj[neighbor])))
                    path.append(neighbor)
                    found_next = True
                    break
            if not found_next:
                stack.pop()
                path.pop()
        
        return None

    def decompose_greedy(edges, rng):
        if not edges:
            return []
        
        n = max(max(u, v) for u, v in edges) + 1
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))
        
        pieces = []
        
        while edge_set:
            # Randomize the order of start vertices to explore different decompositions
            starts = [v for v in range(n) if adj[v]]
            rng.shuffle(starts)
            
            best_cycle = None
            best_len = 0
            
            for start in starts:
                if not adj[start]:
                    continue
                cycle = find_cycle_fast(adj, n, start)
                if cycle:
                    if len(cycle) > best_len:
                        best_len = len(cycle)
                        best_cycle = cycle
                    if best_len >= n:
                        break
            
            if best_cycle is None:
                for u, v in sorted(edge_set):
                    pieces.append([u, v])
                break
            
            # Remove cycle edges
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                edge = (min(u, v), max(u, v))
                if edge in edge_set:
                    edge_set.remove(edge)
                    adj[u].discard(v)
                    adj[v].discard(u)
            
            pieces.append(best_cycle[:])
        
        return pieces

    def polish_pieces(pieces, edges):
        """Try to merge edge pieces into cycles if possible."""
        # Identify which pieces are single edges
        edge_pieces = [p for p in pieces if len(p) == 2]
        cycle_pieces = [p for p in pieces if len(p) > 2]
        
        if not edge_pieces:
            return pieces
        
        # Build a graph of the edge pieces
        adj = {i: set() for i in range(max(max(max(u, v) for u, v in edges), 1))}
        ep_set = set()
        for p in edge_pieces:
            u, v = p
            adj[u].add(v)
            adj[v].add(u)
            ep_set.add((min(u, v), max(u, v)))
        
        new_pieces = list(cycle_pieces)
        remaining_edges = list(ep_set)
        
        # Greedily find cycles in the subgraph of edge pieces
        while remaining_edges:
            start = remaining_edges[0][0]
            # Find a cycle in the edge-piece subgraph
            cycle = find_cycle_fast(adj, max(adj) + 1, start)
            if cycle:
                # Remove these edges from remaining
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                        adj[u].discard(v)
                        adj[v].discard(u)
                new_pieces.append(cycle[:])
            else:
                # No cycle found, take the first edge as a single piece
                e = remaining_edges.pop(0)
                new_pieces.append([e[0], e[1]])
                adj[e[0]].discard(e[1])
                adj[e[1]].discard(e[0])
                
        return new_pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        
        best_pieces = None
        best_count = float('inf')
        
        # Run multiple randomized greedy attempts
        for i in range(10):
            rng = Random(1000 + i)
            pieces = decompose_greedy(edges, rng)
            # Polish
            polished = polish_pieces(pieces, edges)
            count = len(polished)
            if count < best_count:
                best_count = count
                best_pieces = polished
        
        # Ensure we have a valid decomposition if best is None (shouldn't happen)
        if best_pieces is None:
            best_pieces = [[u, v] for u, v in edges]
            
        results.append(best_pieces)
    
    return results
