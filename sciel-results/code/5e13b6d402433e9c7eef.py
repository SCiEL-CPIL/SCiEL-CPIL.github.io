import random
import time
from collections import defaultdict
import numpy as np
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_mcf(edges, n):
        """
        Decompose edges into minimum number of edge-disjoint paths using min-cost flow.
        
        Formulation:
        - Each undirected edge (u,v) becomes two directed arcs: u->v and v->u, each with capacity 1.
        - We want to select a subset of arcs such that:
          * Each undirected edge is used at most once (in either direction)
          * The selected arcs form a collection of vertex-disjoint paths and cycles
          * Minimize the number of paths (equivalently, maximize the number of "connected" arc pairs)
        
        Actually, let's think differently. The minimum path decomposition equals:
        max(0, (number of odd-degree vertices) / 2) for connected graphs (by Gallai's theorem context)
        But we need to actually find the paths.
        
        Better approach: Use flow to find the minimum number of paths.
        
        Key insight: A path decomposition with k paths means we can orient edges such that:
        - Each vertex has out-degree - in-degree in {-1, 0, 1}
        - Vertices with balance +1 are path starts, -1 are path ends, 0 are internal
        - Number of paths = (number of +1 vertices) = (number of -1 vertices)
        
        To minimize paths, we want to minimize the number of vertices with nonzero balance.
        
        This is a min-cost flow problem:
        - Nodes: 0..n-1
        - Arcs: for each undirected edge (u,v), add arcs u->v and v->u with capacity 1, cost 0
        - We need to assign flows such that flow conservation holds except at source/sink nodes
        - The number of paths = sum of positive imbalances
        
        Actually, let's use a different formulation. We can model this as:
        - Create a flow network where we send flow along edges
        - Each edge can carry at most 1 unit of flow in each direction
        - But we want to use each undirected edge at most once total
        
        Let me use a standard approach:
        - For each undirected edge (u,v), create two arcs: u->v (index 2*i) and v->u (index 2*i+1)
        - Add a constraint: x_{u->v} + x_{v->u} <= 1 (edge used at most once)
        - Flow conservation: for each node, sum(out) - sum(in) = b_v where b_v in {-1, 0, 1}
        - Minimize sum of b_v for b_v > 0 (which equals number of paths)
        
        This is an integer linear program. Let's solve it with linprog.
        """
        if not edges:
            return []
        
        m = len(edges)
        
        # Variables:
        # x_{2i} = flow on arc u_i -> v_i (0 or 1)
        # x_{2i+1} = flow on arc v_i -> u_i (0 or 1)
        # y_v = 1 if vertex v is a path start (balance +1), 0 otherwise
        # z_v = 1 if vertex v is a path end (balance -1), 0 otherwise
        
        # Total variables: 2*m + 2*n
        # x: 0..2m-1
        # y: 2m..2m+n-1
        # z: 2m+n..2m+2n-1
        
        # Objective: minimize sum(y) = number of path starts
        c = np.zeros(2*m + 2*n)
        for v in range(n):
            c[2*m + v] = 1.0  # y_v
        
        # Constraints:
        # 1. For each edge i=(u,v): x_{2i} + x_{2i+1} <= 1
        # 2. For each vertex v: sum_{u: (u,v) in E} x_{u->v} - sum_{w: (v,w) in E} x_{v->w} = y_v - z_v
        #    i.e., sum_in - sum_out = z_v - y_v
        # 3. y_v + z_v <= 1 (vertex can't be both start and end)
        # 4. y_v, z_v >= 0, x >= 0, x <= 1
        
        # Let's build the constraint matrix A_ub for:
        # - Edge constraints: x_{2i} + x_{2i+1} <= 1
        # - Vertex constraints as equalities (handled separately)
        # - y_v + z_v <= 1
        
        # Actually, let's use A_eq for flow conservation and A_ub for the rest.
        
        # Flow conservation: for each vertex v:
        # sum_{arcs into v} x_arc - sum_{arcs out of v} x_arc + y_v - z_v = 0
        # i.e., sum_in - sum_out + y_v - z_v = 0
        
        A_eq = np.zeros((n, 2*m + 2*n))
        b_eq = np.zeros(n)
        
        for i, (u, v) in enumerate(edges):
            # Arc u->v is variable 2*i
            # Arc v->u is variable 2*i+1
            
            # For vertex u: arc u->v is outgoing, arc v->u is incoming
            A_eq[u, 2*i] = -1   # u->v leaves u
            A_eq[u, 2*i+1] = 1   # v->u enters u
            A_eq[u, 2*m + u] = 1   # y_u
            A_eq[u, 2*m + n + u] = -1  # -z_u
            
            # For vertex v: arc u->v is incoming, arc v->u is outgoing
            A_eq[v, 2*i] = 1   # u->v enters v
            A_eq[v, 2*i+1] = -1  # v->u leaves v
            A_eq[v, 2*m + v] = 1   # y_v
            A_eq[v, 2*m + n + v] = -1  # -z_v
        
        # Upper bound constraints:
        # 1. For each edge: x_{2i} + x_{2i+1} <= 1
        # 2. For each vertex: y_v + z_v <= 1
        
        A_ub = np.zeros((m + n, 2*m + 2*n))
        b_ub = np.zeros(m + n)
        
        for i in range(m):
            A_ub[i, 2*i] = 1
            A_ub[i, 2*i+1] = 1
            b_ub[i] = 1
        
        for v in range(n):
            A_ub[m + v, 2*m + v] = 1
            A_ub[m + v, 2*m + n + v] = 1
            b_ub[m + v] = 1
        
        # Bounds: 0 <= x <= 1, 0 <= y, z <= 1
        bounds = [(0, 1)] * (2*m) + [(0, 1)] * (2*n)
        
        # Solve
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                           bounds=bounds, method='highs')
        except Exception:
            # Fallback to greedy
            return decompose_greedy(edges, n)
        
        if not result.success:
            return decompose_greedy(edges, n)
        
        # Extract solution
        x = result.x
        
        # Check integrality (should be close to integer due to total unimodularity)
        # Round to nearest integer
        x_rounded = np.round(x)
        
        # Extract paths from the flow solution
        # Build directed graph from selected arcs
        selected_arcs = []
        for i, (u, v) in enumerate(edges):
            if x_rounded[2*i] > 0.5:
                selected_arcs.append((u, v))
            elif x_rounded[2*i+1] > 0.5:
                selected_arcs.append((v, u))
        
        # Build adjacency for selected arcs
        out_adj = defaultdict(list)
        in_adj = defaultdict(list)
        for u, v in selected_arcs:
            out_adj[u].append(v)
            in_adj[v].append(u)
        
        # Find path starts (vertices with out-degree > in-degree, i.e., y_v = 1)
        path_starts = []
        path_ends = []
        for v in range(n):
            if x_rounded[2*m + v] > 0.5:
                path_starts.append(v)
            elif x_rounded[2*m + n + v] > 0.5:
                path_ends.append(v)
        
        # Build paths by following arcs
        paths = []
        
        # For each path start, follow the path
        visited_arcs = set()
        for start in path_starts:
            if not out_adj[start]:
                continue
            path = [start]
            current = start
            while out_adj[current]:
                # Pick an unvisited outgoing arc
                next_node = None
                for nb in out_adj[current]:
                    arc = (current, nb)
                    if arc not in visited_arcs:
                        next_node = nb
                        visited_arcs.add(arc)
                        break
                if next_node is None:
                    break
                path.append(next_node)
                current = next_node
            if len(path) >= 2:
                paths.append(path)
        
        # Handle cycles (if any remain)
        remaining_arcs = set(selected_arcs) - visited_arcs
        while remaining_arcs:
            if time.time() - start_time > time_limit - 5:
                break
            # Pick an arbitrary remaining arc
            u, v = next(iter(remaining_arcs))
            remaining_arcs.discard((u, v))
            
            # Build a cycle
            cycle = [u, v]
            current = v
            while current != u and remaining_arcs:
                found = False
                for nb in out_adj.get(current, []):
                    arc = (current, nb)
                    if arc in remaining_arcs:
                        remaining_arcs.discard(arc)
                        cycle.append(nb)
                        current = nb
                        found = True
                        break
                if not found:
                    break
            
            if len(cycle) >= 2 and cycle[0] == cycle[-1]:
                # It's a cycle, convert to path by removing one edge
                # Actually, a cycle is a valid path decomposition element
                # But the problem asks for paths, not cycles. Let's keep it as a path.
                paths.append(cycle[:-1])  # Remove the last vertex to make it a path
            elif len(cycle) >= 2:
                paths.append(cycle)
        
        # Verify all edges are covered
        covered_edges = set()
        for path in paths:
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                covered_edges.add(e)
        
        if covered_edges != set(edges):
            # Fallback to greedy
            return decompose_greedy(edges, n)
        
        return paths

    def decompose_greedy(edges, n):
        """Fallback greedy decomposition."""
        if not edges:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edges)
        paths = []
        
        # Compute degrees
        degree = defaultdict(int)
        for u, v in remaining_edges:
            degree[u] += 1
            degree[v] += 1
        
        # Greedy: cover remaining edges
        while remaining_edges:
            if time.time() - start_time > time_limit - 2:
                for e in remaining_edges:
                    paths.append(list(e))
                break
            
            # Pick an edge with highest endpoint degrees
            best_edge = None
            best_key = float('-inf')
            for e in remaining_edges:
                u, v = e
                du = sum(1 for nb in adj[u] if tuple(sorted((u, nb))) in remaining_edges)
                dv = sum(1 for nb in adj[v] if tuple(sorted((v, nb))) in remaining_edges)
                k = du + dv
                if k > best_key:
                    best_key = k
                    best_edge = e
            
            if best_edge is None:
                break
            
            u, v = best_edge
            path = [u, v]
            remaining_edges.discard(best_edge)
            
            # Extend from u end
            while True:
                current = path[0]
                candidates = [nb for nb in adj[current] 
                             if tuple(sorted((current, nb))) in remaining_edges 
                             and nb not in set(path)]
                if not candidates:
                    break
                candidates.sort(key=lambda nb: -sum(1 for nb2 in adj[nb] 
                                                    if tuple(sorted((nb, nb2))) in remaining_edges))
                next_node = candidates[0]
                path.insert(0, next_node)
                remaining_edges.discard(tuple(sorted((current, next_node))))
            
            # Extend from v end
            while True:
                current = path[-1]
                candidates = [nb for nb in adj[current] 
                             if tuple(sorted((current, nb))) in remaining_edges 
                             and nb not in set(path)]
                if not candidates:
                    break
                candidates.sort(key=lambda nb: -sum(1 for nb2 in adj[nb] 
                                                    if tuple(sorted((nb, nb2))) in remaining_edges))
                next_node = candidates[0]
                path.append(next_node)
                remaining_edges.discard(tuple(sorted((current, next_node))))
            
            paths.append(path)
        
        return paths

    def decompose(edges):
        if not edges:
            return []
        
        n = max(max(u, v) for u, v in edges) + 1
        
        # Try MCF approach first
        paths = decompose_mcf(edges, n)
        
        # Verify paths are valid
        valid = True
        used_edges = set()
        for path in paths:
            if len(path) < 2:
                valid = False
                break
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
                if e not in set(edges):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid or used_edges != set(edges):
            # Fallback to greedy
            paths = decompose_greedy(edges, n)
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
    return results
