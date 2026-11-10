"""Decomposition of fixed graphs into cycles and edges."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
import random
import numpy as np

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_2_core(edge_set):
        """Extract the 2-core of the graph."""
        if not edge_set:
            return set(), {}
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        changed = True
        while changed:
            changed = False
            to_remove = set()
            for u in list(adj.keys()):
                if len(adj[u]) < 2:
                    to_remove.add(u)
            if to_remove:
                changed = True
                for u in to_remove:
                    for v in adj[u]:
                        adj[v].discard(u)
                    del adj[u]
        
        core_edges = set()
        for u in adj:
            for v in adj[u]:
                if u < v:
                    core_edges.add((u, v))
        return core_edges, dict(adj)

    def find_cycles_greedy(remaining_edges, n, max_iterations=5000):
        """
        Greedily extract cycles from the remaining edge set.
        Returns list of cycles (each as list of vertices).
        """
        cycles = []
        remaining = set(remaining_edges)
        
        if not remaining:
            return cycles
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        # Iteratively find and remove cycles
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
            if not remaining:
                break
            
            # Find a cycle using DFS
            cycle = _find_cycle_dfs(adj, remaining)
            if cycle is None:
                break
            
            # Remove cycle edges from remaining
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                remaining.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)
            
            cycles.append(cycle)
        
        return cycles, remaining

    def _find_cycle_dfs(adj, remaining):
        """Find a cycle in the graph using DFS."""
        visited = set()
        parent = {}
        
        for start in list(adj.keys()):
            if start in visited:
                continue
            
            # DFS from start
            stack = [(start, iter(list(adj[start])))]
            visited.add(start)
            parent[start] = None
            
            while stack:
                node, neighbors = stack[-1]
                advanced = False
                for neighbor in neighbors:
                    e = tuple(sorted((node, neighbor)))
                    if e not in remaining:
                        continue
                    if neighbor == start and len(stack) >= 2:
                        # Found a cycle
                        cycle = [start]
                        cur = node
                        while cur != start:
                            cycle.append(cur)
                            cur = parent[cur]
                        cycle.reverse()
                        return cycle
                    elif neighbor not in visited:
                        visited.add(neighbor)
                        parent[neighbor] = node
                        stack.append((neighbor, iter(list(adj[neighbor]))))
                        advanced = True
                        break
                if not advanced:
                    stack.pop()
        
        return None

    def find_best_decomposition(edges, n):
        """Find decomposition with minimum pieces."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        if not edge_set:
            return []
        
        best_pieces = None
        best_count = len(edge_set) + 1
        
        # Strategy: Use 2-core to identify cycle regions, then extract cycles
        core_edges, core_adj = get_2_core(edge_set)
        
        # Phase 1: Extract cycles from the 2-core
        cycles, remaining = find_cycles_greedy(core_edges, n)
        
        # Phase 2: Try to extract more cycles from remaining edges
        # (including non-core edges that might form cycles)
        extra_cycles, remaining = find_cycles_greedy(remaining, n)
        cycles.extend(extra_cycles)
        
        # Phase 3: Remaining edges become single-edge pieces
        pieces = [list(c) for c in cycles]
        for e in remaining:
            pieces.append(list(e))
        
        best_count = len(pieces)
        best_pieces = pieces[:]
        
        # Try randomized restarts to potentially find better decompositions
        for restart in range(200):
            if time.time() > deadline - 10:
                break
            
            # Randomize the order of cycle extraction
            random_edges = list(edge_set)
            random.shuffle(random_edges)
            
            # Use a different strategy: prioritize longer cycles
            cycles2, remaining2 = _find_cycles_randomized(random_edges, n)
            pieces2 = [list(c) for c in cycles2]
            for e in remaining2:
                pieces2.append(list(e))
            
            if len(pieces2) < best_count:
                best_count = len(pieces2)
                best_pieces = pieces2[:]
        
        return best_pieces if best_pieces else [list(e) for e in edge_set]

    def _find_cycles_randomized(edge_list, n):
        """Find cycles using randomized greedy approach."""
        remaining = set(edge_list)
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        cycles = []
        max_iter = 2000
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            if not remaining:
                break
            
            # Pick a random edge and try to find a cycle through it
            sample = random.sample(list(remaining), min(10, len(remaining)))
            found_cycle = False
            
            for e in sample:
                u, v = e
                # Try to find a path from v to u avoiding edge (u,v)
                path = _find_path_avoiding_edge(u, v, adj, remaining, e)
                if path and len(path) >= 2:
                    cycle = [u] + path
                    if len(cycle) >= 3:
                        # Remove cycle edges
                        for i in range(len(cycle)):
                            cu, cv = cycle[i], cycle[(i + 1) % len(cycle)]
                            ce = tuple(sorted((cu, cv)))
                            remaining.discard(ce)
                            adj[cu].discard(cv)
                            adj[cv].discard(cu)
                        cycles.append(cycle)
                        found_cycle = True
                        break
            
            if not found_cycle:
                break
        
        return cycles, remaining

    def _find_path_avoiding_edge(u, v, adj, remaining_edges, exclude_edge):
        """Find a path from v to u avoiding the specific edge (u,v)."""
        parent = {v: None}
        queue = deque([v])
        
        while queue:
            current = queue.popleft()
            if current == u:
                path = []
                node = u
                while node != v:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            
            for nb in adj[current]:
                if nb not in parent:
                    e = tuple(sorted((current, nb)))
                    if e == exclude_edge:
                        continue
                    if e in remaining_edges:
                        parent[nb] = current
                        queue.append(nb)
        
        return None

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = find_best_decomposition(edges, n)
        results.append(pieces)
        
        if time.time() > deadline - 5:
            # Fallback for remaining graphs
            for spec2 in specs[len(results):]:
                n2, p2, seed2 = spec2
                edges2 = instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break
    
    return results
# EVOLVE-BLOCK-END
