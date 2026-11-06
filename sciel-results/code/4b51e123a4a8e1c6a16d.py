import time
import random
from collections import defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_via_flow(edges, n_vertices):
        """Decompose edges into minimum number of paths using min-cost flow / MILP."""
        if not edges:
            return []
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Degrees
        deg = defaultdict(int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        
        # For each undirected edge, we have two directed arcs
        # x[e][d] = 1 if edge e is traversed in direction d
        # We'll use a single binary variable per directed arc
        # Arcs: for edge (u,v) with u<v, arcs are (u,v) and (v,u)
        
        # Number of arcs = 2 * len(edges)
        m = len(edges)
        
        # Map from (u,v) directed arc to index
        arc_to_idx = {}
        idx_to_arc = []
        for i, (u, v) in enumerate(edges):
            arc_to_idx[(u, v)] = 2 * i
            arc_to_idx[(v, u)] = 2 * i + 1
            idx_to_arc.append((u, v))
            idx_to_arc.append((v, u))
        
        n_arcs = 2 * m
        
        # Variables:
        # x[j] for j in 0..n_arcs-1: binary, 1 if directed arc j is used
        # y[v] for v in 0..n_vertices-1: integer >= 0, number of path starts at v
        # z[v] for v in 0..n_vertices-1: integer >= 0, number of path ends at v
        
        # Total variables: n_arcs + 2*n_vertices
        n_vars = n_arcs + 2 * n_vertices
        
        # Objective: minimize sum of y[v] + z[v] for all v
        # Since each path has exactly one start and one end, total paths = sum(y) = sum(z)
        # So we minimize sum(y[v]) for all v (or equivalently sum(z[v]))
        c = np.zeros(n_vars)
        for v in range(n_vertices):
            c[n_arcs + v] = 1.0  # y[v]
        
        # Constraints:
        # 1. For each undirected edge (u,v): x[(u,v)] + x[(v,u)] = 1
        # 2. Flow conservation at each vertex v:
        #    sum of x[(v,u)] for all neighbors u (outgoing) + y[v] = 
        #    sum of x[(u,v)] for all neighbors u (incoming) + z[v]
        #    Rearranged: sum_out x - sum_in x + y[v] - z[v] = 0
        
        # Build constraint matrix
        # Constraint 1: 2m constraints (one per edge)
        # Constraint 2: n_vertices constraints (one per vertex)
        n_constraints = m + n_vertices
        
        A = np.zeros((n_constraints, n_vars))
        lb = np.zeros(n_constraints)
        ub = np.zeros(n_constraints)
        
        # Constraint 1: for each edge i with (u,v), x[2i] + x[2i+1] = 1
        for i, (u, v) in enumerate(edges):
            A[i, 2 * i] = 1.0
            A[i, 2 * i + 1] = 1.0
            lb[i] = 1.0
            ub[i] = 1.0
        
        # Constraint 2: for each vertex v
        # sum of outgoing arcs from v - sum of incoming arcs to v + y[v] - z[v] = 0
        for v in range(n_vertices):
            row = m + v
            # Outgoing arcs from v: arcs (v, u) for each neighbor u
            for u in adj[v]:
                arc = (v, u)
                if arc in arc_to_idx:
                    A[row, arc_to_idx[arc]] = 1.0
                # Incoming arcs to v: arcs (u, v) for each neighbor u
                arc_in = (u, v)
                if arc_in in arc_to_idx:
                    A[row, arc_to_idx[arc_in]] = -1.0
            # y[v]
            A[row, n_arcs + v] = 1.0
            # z[v]
            A[row, n_arcs + n_vertices + v] = -1.0
            lb[row] = 0.0
            ub[row] = 0.0
        
        # Variable bounds
        lower = np.zeros(n_vars)
        upper = np.ones(n_vars)
        
        # x variables are binary
        integrality = np.zeros(n_vars)
        for j in range(n_arcs):
            integrality[j] = 1.0  # binary
        
        # y and z are non-negative integers
        for v in range(n_vertices):
            integrality[n_arcs + v] = 1.0
            upper[n_arcs + v] = deg[v]  # at most deg[v] paths can start at v
            integrality[n_arcs + n_vertices + v] = 1.0
            upper[n_arcs + n_vertices + v] = deg[v]
        
        constraints = LinearConstraint(A, lb, ub)
        bounds = Bounds(lower, upper)
        
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        except Exception:
            return None
        
        if not res.success:
            return None
        
        # Extract solution
        x = res.x
        used_arcs = []
        for j in range(n_arcs):
            if x[j] > 0.5:
                used_arcs.append(idx_to_arc[j])
        
        # Reconstruct paths from used arcs
        # Build directed adjacency from used arcs
        dir_adj = defaultdict(list)
        for u, v in used_arcs:
            dir_adj[u].append(v)
        
        # Find path starts: vertices where y[v] > 0
        # Actually, let's find them by looking for vertices with outdegree > indegree in the selected arcs
        outdeg = defaultdict(int)
        indeg = defaultdict(int)
        for u, v in used_arcs:
            outdeg[u] += 1
            indeg[v] += 1
        
        # Path starts are vertices where outdeg > indeg (specifically outdeg = indeg + 1 for paths)
        # More precisely, starts are vertices with outdeg - indeg = 1 (for non-cyclic paths)
        # But we might have cycles too. Let's handle it generally.
        
        # Find all paths and cycles
        visited_arcs = set()
        paths = []
        
        # Get all vertices that are in the used arcs
        all_verts = set()
        for u, v in used_arcs:
            all_verts.add(u)
            all_verts.add(v)
        
        # For each vertex, compute net outflow
        # Start vertices: outdeg - indeg = 1
        # End vertices: indeg - outdeg = 1
        # Middle vertices: outdeg = indeg
        # For cycles: all vertices have outdeg = indeg
        
        starts = []
        for v in all_verts:
            if outdeg[v] - indeg[v] == 1:
                starts.append(v)
        
        # Trace paths from each start
        for start in starts:
            path = [start]
            current = start
            while True:
                # Find next vertex
                next_v = None
                for v in dir_adj[current]:
                    if (current, v) not in visited_arcs:
                        next_v = v
                        break
                if next_v is None:
                    break
                visited_arcs.add((current, next_v))
                path.append(next_v)
                current = next_v
            paths.append(path)
        
        # Check if there are any cycles (arcs not in any path)
        remaining_arcs = set(used_arcs) - visited_arcs
        while remaining_arcs:
            # Find a cycle
            arc = next(iter(remaining_arcs))
            cycle = [arc[0]]
            current = arc[0]
            while True:
                remaining_arcs.discard((current, arc[1]))
                cycle.append(arc[1])
                current = arc[1]
                # Find next arc from current
                next_arc = None
                for a in remaining_arcs:
                    if a[0] == current:
                        next_arc = a
                        break
                if next_arc is None:
                    break
                arc = next_arc
                if current == cycle[0]:
                    break
            # Remove duplicate endpoint
            if len(cycle) > 1 and cycle[0] == cycle[-1]:
                cycle = cycle[:-1]
            paths.append(cycle)
        
        return paths

    def greedy_decompose(edges):
        """Fallback greedy decomposition."""
        if not edges:
            return []
        
        degs = defaultdict(int)
        for u, v in edges:
            degs[u] += 1
            degs[v] += 1
        
        odd_vertices = [v for v, d in degs.items() if d % 2 != 0]
        K = max(1, len(odd_vertices) // 2)
        
        if K == 1:
            # Eulerian path
            adj = defaultdict(list)
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            start = odd_vertices[0] if odd_vertices else next(iter(adj))
            path = [start]
            stack = [start]
            remaining = set(edges)
            
            while stack:
                v = stack[-1]
                next_v = None
                while adj[v]:
                    u = adj[v].pop()
                    e = tuple(sorted((v, u)))
                    if e in remaining:
                        remaining.remove(e)
                        adj[u].remove(v)
                        next_v = u
                        break
                if next_v is not None:
                    stack.append(next_v)
                    path.append(next_v)
                else:
                    stack.pop()
            return [path]
        
        # Greedy: repeatedly find longest path
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        paths = []
        rng = random.Random(42)
        
        while remaining:
            # Find a good start vertex (high degree)
            active_verts = set()
            for u, v in remaining:
                active_verts.add(u)
                active_verts.add(v)
            
            # Try multiple starts, pick best path
            best_path = None
            candidates = list(active_verts)
            rng.shuffle(candidates)
            for start in candidates[:min(10, len(candidates))]:
                path = [start]
                current = start
                visited = {start}
                while True:
                    avail = [nb for nb in adj[current] 
                             if tuple(sorted((current, nb))) in remaining 
                             and nb not in visited]
                    if not avail:
                        break
                    # Pick neighbor with highest remaining degree
                    avail.sort(key=lambda x: sum(1 for e in remaining if x in e), reverse=True)
                    next_v = avail[0]
                    edge = tuple(sorted((current, next_v)))
                    remaining.discard(edge)
                    path.append(next_v)
                    visited.add(next_v)
                    current = next_v
                
                if best_path is None or len(path) > len(best_path):
                    best_path = path
            
            if best_path:
                paths.append(best_path)
                # Remove edges from adj
                for i in range(len(best_path) - 1):
                    u, v = best_path[i], best_path[i+1]
                    adj[u].discard(v)
                    adj[v].discard(u)
        
        return paths

    def try_merge(paths, edges):
        """Try to merge paths that share endpoints."""
        if len(paths) <= 1:
            return paths
        
        edge_set = set(tuple(sorted(p)) for p in edges)
        
        path_info = []
        for p in paths:
            if len(p) < 2:
                continue
            p_edges = set()
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i+1])))
                p_edges.add(e)
            path_info.append({
                'path': p,
                'edges': p_edges,
                'start': p[0],
                'end': p[-1]
            })
        
        improved = True
        while improved:
            improved = False
            for i in range(len(path_info)):
                for j in range(i + 1, len(path_info)):
                    if improved:
                        break
                    pi, pj = path_info[i], path_info[j]
                    if set(pi['path']) & set(pj['path']):
                        continue
                    
                    new_path = None
                    if pi['end'] == pj['start']:
                        new_path = pi['path'] + pj['path'][1:]
                    elif pi['start'] == pj['end']:
                        new_path = list(reversed(pj['path'])) + pi['path'][1:]
                    elif pi['end'] == pj['end']:
                        new_path = pi['path'] + list(reversed(pj['path']))[1:]
                    elif pi['start'] == pj['start']:
                        new_path = list(reversed(pj['path'])) + pi['path'][1:]
                    
                    if new_path is None:
                        continue
                    if len(set(new_path)) != len(new_path):
                        continue
                    
                    valid = True
                    for k in range(len(new_path) - 1):
                        e = tuple(sorted((new_path[k], new_path[k+1])))
                        if e not in edge_set:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    new_edges = set()
                    for k in range(len(new_path) - 1):
                        new_edges.add(tuple(sorted((new_path[k], new_path[k+1]))))
                    
                    path_info[i]['path'] = new_path
                    path_info[i]['edges'] = new_edges
                    path_info[i]['start'] = new_path[0]
                    path_info[i]['end'] = new_path[-1]
                    path_info.pop(j)
                    improved = True
                    break
                if improved:
                    break
        
        return [p['path'] for p in path_info]

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for spec in specs:
        if time.time() > deadline - 10:
            # Not enough time, use greedy for remaining
            edges = instance(*spec)
            paths = greedy_decompose(edges)
            results.append(paths)
            continue
        
        edges = instance(*spec)
        n = spec[0]
        
        # Try MILP-based approach
        paths = decompose_via_flow(edges, n)
        
        if paths is None or len(paths) == 0:
            paths = greedy_decompose(edges)
        else:
            paths = try_merge(paths, edges)
        
        results.append(paths)
    
    return results
