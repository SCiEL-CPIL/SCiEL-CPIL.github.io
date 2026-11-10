"""Decomposition of three fixed graphs into few cycles and edges (Erdős-Gallai).

Approach:
- Reproduce the verifier's instance generator exactly.
- Decompose each graph into cycles (>=3 vertices) and single edges, minimizing total pieces.
- Strategy:
  1. Use a 2-matching / flow-based approach to find a large set of edge-disjoint cycles.
     We model this as finding a 2-factor (or near 2-factor) in the graph.
  2. Extract cycles from the 2-factor.
  3. For remaining edges, use a greedy long-cycle extraction heuristic.
  4. Leftover edges become single-edge pieces.
"""

import random
from collections import defaultdict, deque
from scipy.optimize import linprog
import numpy as np

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def extract_cycles_from_subgraph(edges_subset):
        """Given a set of edges forming a subgraph where each vertex has degree 0, 1, or 2,
        extract all cycles (each cycle >= 3 vertices). Return list of cycles and leftover edges."""
        if not edges_subset:
            return [], []
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges_subset:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find connected components
        visited = set()
        cycles = []
        leftover_edges = set()
        
        for node in list(adj.keys()):
            if node in visited:
                continue
            # BFS/DFS to find component
            comp_nodes = set()
            queue = deque([node])
            visited.add(node)
            while queue:
                cur = queue.popleft()
                comp_nodes.add(cur)
                for nb in adj[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            
            # Collect edges in this component
            comp_edges = set()
            for u in comp_nodes:
                for v in adj[u]:
                    if u < v:
                        comp_edges.add((u, v))
            
            # Check if it's a cycle (all vertices degree 2) or has paths
            all_degree_2 = all(len(adj[u]) == 2 for u in comp_nodes)
            
            if all_degree_2 and len(comp_nodes) >= 3:
                # It's a cycle - trace it
                start = sorted(comp_nodes)[0]
                cycle = [start]
                prev = None
                cur = start
                while True:
                    nxt = None
                    for nb in adj[cur]:
                        if nb != prev:
                            nxt = nb
                            break
                    if nxt is None or nxt == start:
                        break
                    cycle.append(nxt)
                    prev, cur = cur, nxt
                if len(cycle) >= 3:
                    cycles.append(cycle)
                else:
                    for e in comp_edges:
                        leftover_edges.add(e)
            else:
                # Not a pure cycle; edges here are leftover
                for e in comp_edges:
                    leftover_edges.add(e)
        
        return cycles, leftover_edges

    def decompose(edges):
        if not edges:
            return []
        
        n = max(max(u, v) for u, v in edges) + 1
        edge_list = list(edges)
        num_edges = len(edge_list)
        
        # Map edge to index
        edge_to_idx = {e: i for i, e in enumerate(edge_list)}
        
        # Step 1: Find a maximum 2-matching (each vertex degree <= 2) using ILP-like approach
        # with scipy linprog. Maximize number of edges in the 2-matching.
        # This gives us a subgraph where each vertex has degree 0, 1, or 2.
        # Cycles in this subgraph are edge-disjoint.
        
        # Decision variables: x_e in [0, 1] for each edge
        # Objective: maximize sum x_e
        # Constraints: for each vertex v, sum_{e incident to v} x_e <= 2
        
        c = np.ones(num_edges)  # maximize -> minimize -c
        A_ub = []
        b_ub = []
        
        # Vertex constraints
        for v in range(n):
            row = np.zeros(num_edges)
            for e, (u, w) in enumerate(edge_list):
                if u == v or w == v:
                    row[e] = 1
            A_ub.append(row)
            b_ub.append(2)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        bounds = [(0, 1)] * num_edges
        
        try:
            res = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                # Get the 2-matching
                selected_edges = set()
                for i, x in enumerate(res.x):
                    if x > 0.5:
                        selected_edges.add(edge_list[i])
            else:
                selected_edges = set()
        except Exception:
            selected_edges = set()
        
        # Extract cycles from the 2-matching
        cycles, leftover_from_2match = extract_cycles_from_subgraph(selected_edges)
        
        # Remaining edges after removing the 2-matching edges
        remaining = set(edge_list) - selected_edges
        
        # Step 2: Greedy long-cycle extraction on remaining edges
        rng = random.Random(12345)
        
        def find_long_cycle(rem_edges, adj_map):
            """Try to find a long cycle in the remaining graph using DFS with heuristics."""
            if not rem_edges:
                return None
            
            best_cycle = None
            
            # Try multiple starting points
            vertices = sorted(adj_map.keys())
            # Prioritize high-degree vertices
            vertices.sort(key=lambda v: -len(adj_map[v]))
            
            for start in vertices[:min(10, len(vertices))]:
                # DFS to find a cycle back to start
                visited = set()
                parent = {}
                
                def dfs(node, depth):
                    nonlocal best_cycle
                    visited.add(node)
                    for nb in sorted(adj_map[node], key=lambda x: -len(adj_map[x])):
                        if nb == start and depth >= 2:
                            # Found a cycle
                            path = [node]
                            cur = node
                            while cur != start:
                                cur = parent[cur]
                                path.append(cur)
                            path.reverse()
                            if best_cycle is None or len(path) > len(best_cycle):
                                best_cycle = path
                            return True
                        if nb not in visited:
                            parent[nb] = node
                            if dfs(nb, depth + 1):
                                return True
                    visited.discard(node)
                    return False
                
                visited = set()
                parent = {}
                dfs(start, 0)
                
                if best_cycle and len(best_cycle) >= 3:
                    return best_cycle
            
            return None
        
        while remaining:
            adj = defaultdict(set)
            for u, v in remaining:
                adj[u].add(v)
                adj[v].add(u)
            
            # Remove isolated vertices
            adj = {v: nbs for v, nbs in adj.items() if nbs}
            
            if not adj:
                break
            
            # Try to find a cycle
            cycle = find_long_cycle(remaining, adj)
            
            if cycle and len(cycle) >= 3:
                # Verify all edges exist
                valid = True
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    e = tuple(sorted((u, v)))
                    if e not in remaining:
                        valid = False
                        break
                if valid:
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        remaining.discard(tuple(sorted((u, v))))
                    cycles.append(cycle)
                else:
                    break
            else:
                break
        
        # Step 3: Remaining edges become single-edge pieces
        for e in remaining:
            cycles.append(list(e))
        
        return cycles

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
    
    return results
