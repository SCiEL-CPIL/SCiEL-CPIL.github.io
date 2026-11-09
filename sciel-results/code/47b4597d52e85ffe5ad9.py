import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_graph_lp(edges, n):
        """
        Solve the edge-disjoint path cover problem using min-cost flow.
        Each undirected edge is split into two directed arcs with capacity 1.
        Node balances are set based on degree parity to minimize path count.
        """
        from scipy.optimize import linprog
        import numpy as np

        if not edges:
            return []

        edge_set = set(tuple(sorted(e)) for e in edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Build directed arcs
        arcs = []
        arc_index = {}
        for u, v in edge_set:
            arcs.append((u, v))
            arc_index[(u, v)] = len(arcs) - 1
            arcs.append((v, u))
            arc_index[(v, u)] = len(arcs) - 1

        m = len(arcs)  # number of arcs
        # Variables: x_a for each arc (0 or 1), y_v for each node (0 or 1, indicating if node is an endpoint)
        # Actually, let's use a different formulation.
        
        # Standard min-cost flow formulation for path cover:
        # We want to find a set of edge-disjoint paths covering all edges.
        # This is equivalent to finding a circulation with demands.
        # Let x_{uv} be 1 if arc (u,v) is used.
        # Flow conservation: sum_{v} x_{uv} - sum_{v} x_{vu} = b_u
        # where b_u = 1 if u is the start of a path, -1 if u is the end, 0 otherwise.
        # The number of paths is (1/2) * sum_u |b_u| = (1/2) * sum_u b_u^2 (since b_u in {-1,0,1}).
        # Actually, sum of b_u must be 0.
        # We want to minimize the number of paths, which is (1/2) * sum_u |b_u|.
        # Since b_u is integer, this is equivalent to minimizing sum_u |b_u|.
        # We can introduce variables s_u, t_u >= 0 such that b_u = s_u - t_u, and minimize sum_u (s_u + t_u).
        # But b_u is determined by the flow.
        
        # Alternative:
        # Minimize sum_u (s_u + t_u)
        # Subject to:
        #   sum_{v: (u,v) in arcs} x_{uv} - sum_{v: (v,u) in arcs} x_{vu} = s_u - t_u
        #   0 <= x_{uv} <= 1
        #   s_u, t_u >= 0
        #   s_u * t_u = 0 (but this is non-linear, however in LP relaxation with total unimodularity it might work, or we can just let the solver handle it if we keep x binary? No, LP needs linear constraints.)
        # Actually, for min-cost flow with convex costs, we can use piecewise linear.
        # But simpler: the number of paths is exactly (1/2) * sum_u |b_u|.
        # And sum_u |b_u| = sum_u (s_u + t_u) where b_u = s_u - t_u and s_u, t_u >= 0.
        # Since we are minimizing, the solver will naturally set either s_u or t_u to 0.
        
        # Variables:
        # x_0 ... x_{m-1} for arcs
        # s_0 ... s_{n-1} for start surplus
        # t_0 ... t_{n-1} for end surplus
        # Total variables: m + 2n
        
        # Objective: minimize sum(s_u) + sum(t_u)
        c = [0.0] * m + [1.0] * n + [1.0] * n
        
        # Constraints:
        # For each node u:
        #   sum_{v: (u,v) in arcs} x_{uv} - sum_{v: (v,u) in arcs} x_{vu} - s_u + t_u = 0
        #   => sum_{v: (u,v) in arcs} x_{uv} - sum_{v: (v,u) in arcs} x_{vu} - s_u + t_u = 0
        
        A_eq = []
        b_eq = []
        for u in range(n):
            row = [0.0] * m + [0.0] * n + [0.0] * n
            for (a, b) in arcs:
                if a == u:
                    row[arc_index[(a, b)]] = 1.0
                if b == u:
                    row[arc_index[(a, b)]] = -1.0
            row[u] = -1.0  # -s_u
            row[n + u] = 1.0  # +t_u
            A_eq.append(row)
            b_eq.append(0.0)
        
        # Bounds:
        # x_{uv} in [0, 1]
        # s_u >= 0, t_u >= 0
        bounds = [(0, 1)] * m + [(0, None)] * n + [(0, None)] * n
        
        # Solve LP
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not res.success:
            return None
        
        # Extract arcs used
        used_arcs = set()
        for i in range(m):
            if res.x[i] > 0.5:  # Should be close to 0 or 1
                used_arcs.add(arcs[i])
        
        # Reconstruct paths from used arcs
        # Build adjacency from used arcs
        adj_used = defaultdict(list)
        for u, v in used_arcs:
            adj_used[u].append(v)
        
        # Find paths
        paths = []
        # Find nodes with out-degree > in-degree (start of path)
        # Or simply find nodes that are starts: out_degree > in_degree in the directed subgraph
        in_deg = defaultdict(int)
        out_deg = defaultdict(int)
        for u, v in used_arcs:
            out_deg[u] += 1
            in_deg[v] += 1
        
        starts = [u for u in range(n) if out_deg[u] > in_deg[u]]
        if not starts:
            # Could be cycles, but we need to cover edges, so there should be starts if there are odd degree nodes
            # If all degrees are even, we have cycles. But the problem asks for paths.
            # Wait, the problem says "paths must be simple... cover all edges".
            # A cycle is not a path. But if the graph has an Eulerian circuit, we can break it into paths?
            # No, the problem likely implies open paths.
            # If all degrees are even, we can still have a path cover?
            # Actually, if all degrees are even, we can have 0 paths? No, we must cover edges.
            # We can take one edge as a path of length 1? No, that's not a path in the usual sense (needs at least 2 vertices).
            # But a single edge is a path of length 1.
            # However, if we have a cycle, we can break it at any vertex to form a path.
            # But then we use all edges in one path? No, a path cannot repeat vertices.
            # So a cycle cannot be a single path.
            # We must break cycles into paths.
            # The min number of paths to cover edges in an Eulerian graph is 0? No.
            # It is at least 1 if there are edges.
            # Actually, the LP formulation above minimizes the number of path endpoints.
            # If the graph is Eulerian, we can have a circulation, which means b_u = 0 for all u.
            # Then the number of paths is 0? But we need to cover edges.
            # A circulation is a set of cycles. But the problem asks for paths.
            # So we need to break cycles into paths.
            # The LP solution gives a circulation. We need to convert it to paths.
            # This is tricky.
            
            # For now, let's assume there are starts.
            # If there are no starts, we have cycles. We can pick any edge and make it a path?
            # No, we need to cover all edges.
            # Let's just return None if no starts, and use a fallback.
            return None
        
        # Reconstruct paths
        for start in starts:
            if not adj_used[start]:
                continue
            path = [start]
            cur = start
            while adj_used[cur]:
                next_node = adj_used[cur].pop(0)
                path.append(next_node)
                cur = next_node
            paths.append(path)
        
        # Verify coverage
        # If some edges are not covered, or paths are invalid, return None
        # For simplicity, assume LP solution is valid.
        
        return paths

    def solve_graph_greedy(edges, n):
        """Fallback greedy solver."""
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        used_edges = set()
        paths = []
        all_edges = list(edge_set)
        
        while len(used_edges) < len(all_edges):
            start_edge = None
            for e in all_edges:
                if e not in used_edges:
                    start_edge = e
                    break
            
            if start_edge is None:
                break
            
            u, v = start_edge
            current_path = [u, v]
            used_edges.add(start_edge)
            
            last = v
            while True:
                next_w = None
                for w in adj[last]:
                    e = tuple(sorted((last, w)))
                    if e in used_edges:
                        continue
                    if w in current_path:
                        continue
                    next_w = w
                    break
                
                if next_w is None:
                    break
                
                current_path.append(next_w)
                used_edges.add(tuple(sorted((last, next_w))))
                last = next_w
            
            paths.append(current_path)
        
        return paths

    results = []
    for i, spec in enumerate([(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]):
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        edges = instance(*spec)
        n = spec[0]
        
        # Try LP solver
        lp_paths = solve_graph_lp(edges, n)
        if lp_paths is not None:
            # Verify LP solution
            valid = True
            used_edges_check = set()
            for path in lp_paths:
                if len(path) < 2:
                    valid = False
                    break
                # Check simple
                if len(set(path)) != len(path):
                    valid = False
                    break
                # Check edges
                for j in range(len(path) - 1):
                    e = tuple(sorted((path[j], path[j+1])))
                    if e not in set(tuple(sorted(e)) for e in edges):
                        valid = False
                        break
                    if e in used_edges_check:
                        valid = False
                        break
                    used_edges_check.add(e)
                if not valid:
                    break
            
            if valid and used_edges_check == set(tuple(sorted(e)) for e in edges):
                results.append(lp_paths)
            else:
                # Fallback to greedy
                results.append(solve_graph_greedy(edges, n))
        else:
            # Fallback to greedy
            results.append(solve_graph_greedy(edges, n))
        
    return results
