import random
import sys
import numpy as np
from scipy.optimize import linprog

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def solve_graph(n, edges):
        # Build adjacency list and degree count
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        if not edges:
            return []

        # 1. Determine the minimum number of paths based on odd degree vertices
        odd_vertices = [v for v in range(n) if len(adj[v]) % 2 == 1]
        num_odd = len(odd_vertices)
        
        # The minimum number of paths is max(1, num_odd // 2)
        # But we need to construct them.
        
        # We will use a flow formulation to find an orientation that minimizes the number of paths.
        # Variables: x_{uv} for each directed arc (u,v).
        # Constraints:
        # 1. For each undirected edge {u,v}, x_{uv} + x_{vu} = 1
        # 2. Flow conservation: sum_in(v) - sum_out(v) = b_v
        #    where b_v is 0 for even degree, and +/-1 for odd degree.
        #    We want to choose signs for b_v to minimize the number of paths.
        #    Number of paths = sum(b_v for b_v > 0).
        #    Since sum(b_v) = 0, this is 0.5 * sum(|b_v|).
        #    To minimize this, we set b_v = 0 for even, and for odd, we must have |b_v| >= 1.
        #    So min paths = num_odd // 2.
        
        # To find the specific orientation, we can fix b_v for odd vertices?
        # No, the choice of which odd vertices are +1 and which are -1 affects feasibility and cycles.
        # However, for random graphs, any valid assignment of +/-1 to odd vertices that sums to 0 is likely feasible.
        # Let's just pick the first half as +1 and second half as -1.
        
        b = [0] * n
        if num_odd > 0:
            # Assign +1 to first half, -1 to second half
            for i in range(num_odd // 2):
                b[odd_vertices[i]] = 1
            for i in range(num_odd // 2, num_odd):
                b[odd_vertices[i]] = -1
        
        # Now we need to find x_{uv} such that:
        # x_{uv} + x_{vu} = 1
        # sum_{u} x_{uv} - sum_{u} x_{vu} = b_v
        
        # This is a system of linear equations. We can solve it using linprog or just standard linear algebra?
        # Since it's a bipartite-like structure (edge variables vs node balances), we can use linprog.
        # But we just need *a* solution, not necessarily minimizing cost (since cost is fixed by b).
        # Actually, we want to minimize cycles? No, just find *a* valid orientation.
        
        # Let's use linprog to find a feasible solution.
        # Minimize 0
        # Subject to:
        # For each edge {u,v}: x_{uv} + x_{vu} = 1
        # For each node v: sum_{u \in N(v)} x_{uv} - sum_{u \in N(v)} x_{vu} = b_v
        
        # Variables: x_{uv} for all directed arcs.
        # Let index (u,v) be the variable.
        
        arc_indices = {}
        var_count = 0
        for u, v in edges:
            arc_indices[(u, v)] = var_count
            var_count += 1
            arc_indices[(v, u)] = var_count
            var_count += 1
            
        c = np.zeros(var_count)
        
        # Equality constraints
        A_eq = []
        b_eq = []
        
        # Edge constraints: x_{uv} + x_{vu} = 1
        for u, v in edges:
            row = np.zeros(var_count)
            row[arc_indices[(u, v)]] = 1
            row[arc_indices[(v, u)]] = 1
            A_eq.append(row)
            b_eq.append(1)
            
        # Node balance constraints
        for v in range(n):
            row = np.zeros(var_count)
            for u in adj[v]:
                # Arc (u, v) is IN to v
                row[arc_indices[(u, v)]] = 1
                # Arc (v, u) is OUT from v
                row[arc_indices[(v, u)]] = -1
            A_eq.append(row)
            b_eq.append(b[v])
            
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        
        # Bounds: 0 <= x <= 1
        bounds = [(0, 1)] * var_count
        
        # Solve
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not res.success:
            # Fallback to greedy if LP fails
            return solve_graph_greedy(n, edges)
            
        # Extract orientation
        oriented_edges = []
        for u, v in edges:
            if res.x[arc_indices[(u, v)]] > 0.5:
                oriented_edges.append((u, v))
            else:
                oriented_edges.append((v, u))
                
        # Now we have a directed graph.
        # Decompose into paths and cycles.
        # In-degree and out-degree in the oriented graph:
        in_deg = {v: 0 for v in range(n)}
        out_deg = {v: 0 for v in range(n)}
        adj_dir = {v: [] for v in range(n)}
        for u, v in oriented_edges:
            in_deg[v] += 1
            out_deg[u] += 1
            adj_dir[u].append(v)
            
        paths = []
        visited_edges = set()
        
        # Find paths starting from nodes with in_deg < out_deg (sources)
        # And cycles from remaining nodes.
        
        # We'll traverse all edges.
        # Since each node has out_deg = (deg + b_v)/2 and in_deg = (deg - b_v)/2.
        # Sources have out > in. Sinks have in > out.
        
        # We can just do a DFS on the directed graph.
        # But we need to ensure edge-disjointness.
        # Since it's a functional graph? No, out_deg can be > 1.
        
        # We can decompose the directed graph into paths and cycles by following edges.
        # 1. Find all nodes with out_deg > 0.
        # 2. For each, start a path.
        
        # To avoid duplicates, we mark edges as visited.
        
        for v in range(n):
            while out_deg[v] > 0:
                # Start a path from v
                path = [v]
                curr = v
                while True:
                    # Pick an unvisited outgoing edge from curr
                    next_node = None
                    for u in adj_dir[curr]:
                        if (curr, u) not in visited_edges:
                            next_node = u
                            break
                            
                    if next_node is None:
                        break
                    
                    visited_edges.add((curr, next_node))
                    path.append(next_node)
                    curr = next_node
                    # Decrement out_deg of curr (previous) ? No, we just mark edge.
                    # We need to track remaining out-degree to know when to stop.
                    # Actually, if we follow a path, we consume one out-edge at each step.
                    # We stop when we reach a node with no unvisited out-edges.
                    
                paths.append(path)
                
        return paths

    def solve_graph_greedy(n, edges):
        # Fallback to the original greedy approach if LP fails
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining_edges = set(edges)
        paths = []
        
        while remaining_edges:
            min_deg = float('inf')
            start_node = None
            for u in range(n):
                deg = len(adj[u])
                if deg > 0 and deg < min_deg:
                    min_deg = deg
                    start_node = u
            
            if start_node is None:
                break
                
            path = [start_node]
            current = start_node
            visited_in_path = {start_node}
            
            while True:
                best_next = None
                max_deg = -1
                for v in adj[current]:
                    if v not in visited_in_path:
                        if len(adj[v]) > max_deg:
                            max_deg = len(adj[v])
                            best_next = v
                
                if best_next is None:
                    break
                
                path.append(best_next)
                visited_in_path.add(best_next)
                
                u = current
                v = best_next
                edge_key = tuple(sorted((u, v)))
                remaining_edges.remove(edge_key)
                
                adj[u].remove(v)
                adj[v].remove(u)
                
                current = best_next
                
            paths.append(path)
            
        return paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = solve_graph(n, edges)
        results.append(paths)
        
    return results
