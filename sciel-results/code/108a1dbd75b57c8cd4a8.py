"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def path_cover_milp(edges, time_limit):
        """
        Model edge-disjoint path cover as a Mixed Integer Linear Program (MILP).
        Variables:
          x_{uv} in {0, 1} for each directed arc (u, v) corresponding to undirected edge {u, v}.
          t_v in R+ for each vertex v.
        Objective: Minimize sum(t_v)
        Constraints:
          1. x_{uv} + x_{vu} <= 1 for each undirected edge {u, v}.
          2. t_v >= sum_w x_{vw} - sum_w x_{wv}
          3. t_v >= -(sum_w x_{vw} - sum_w x_{wv})
        
        This minimizes the sum of absolute imbalances, which is twice the number of paths.
        """
        import numpy as np
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
        except ImportError:
            return [[u, v] for u, v in edges]
        
        if not edges:
            return []
        
        n = max(max(u, v) for u, v in edges) + 1
        m = len(edges)
        
        # Variable indices:
        # 0 to 2*m - 1: x_{uv} for each arc
        # 2*m to 2*m + n - 1: t_v
        num_vars = 2 * m + n
        
        c = np.zeros(num_vars)
        for v in range(n):
            c[2 * m + v] = 1.0
        
        # Constraints
        # 1. x_{uv} + x_{vu} <= 1
        A1 = np.zeros((m, num_vars))
        b1 = np.ones(m)
        for i in range(m):
            A1[i, 2 * i] = 1.0
            A1[i, 2 * i + 1] = 1.0
        
        # 2. t_v - (outdeg(v) - indeg(v)) >= 0  =>  (outdeg(v) - indeg(v)) - t_v <= 0
        # 3. t_v + (outdeg(v) - indeg(v)) >= 0  =>  -(outdeg(v) - indeg(v)) - t_v <= 0
        
        A2 = np.zeros((2 * n, num_vars))
        b2 = np.zeros(2 * n)
        
        for v in range(n):
            # Row 2*v: outdeg(v) - indeg(v) - t_v <= 0
            # Coeff for t_v is -1
            A2[2 * v, 2 * m + v] = -1.0
            for i in range(m):
                u, w = edges[i]
                # Arc 2*i: u->w
                if u == v:
                    A2[2 * v, 2 * i] += 1.0
                if w == v:
                    A2[2 * v, 2 * i] -= 1.0
                # Arc 2*i+1: w->u
                if w == v:
                    A2[2 * v, 2 * i + 1] += 1.0
                if u == v:
                    A2[2 * v, 2 * i + 1] -= 1.0
            
            # Row 2*v+1: -(outdeg(v) - indeg(v)) - t_v <= 0
            A2[2 * v + 1, 2 * m + v] = -1.0
            for i in range(m):
                u, w = edges[i]
                # Arc 2*i: u->w
                if u == v:
                    A2[2 * v + 1, 2 * i] -= 1.0
                if w == v:
                    A2[2 * v + 1, 2 * i] += 1.0
                # Arc 2*i+1: w->u
                if w == v:
                    A2[2 * v + 1, 2 * i + 1] -= 1.0
                if u == v:
                    A2[2 * v + 1, 2 * i + 1] += 1.0
        
        A = np.vstack([A1, A2])
        b = np.concatenate([b1, b2])
        
        constraints = LinearConstraint(A, -np.inf, b)
        
        integrality = np.zeros(num_vars)
        for i in range(2 * m):
            integrality[i] = 1  # x variables are binary (0 or 1)
        
        bounds = Bounds(lb=np.zeros(num_vars), ub=np.full(num_vars, np.inf))
        for i in range(2 * m):
            bounds.ub[i] = 1.0
        
        try:
            if time.time() > time_limit:
                return [[u, v] for u, v in edges]
            
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if not res.success:
                return [[u, v] for u, v in edges]
            
            x = res.x
            directed = []
            for i in range(m):
                u, v = edges[i]
                xv = x[2 * i]
                xu = x[2 * i + 1]
                if xv > 0.5:
                    directed.append((u, v))
                elif xu > 0.5:
                    directed.append((v, u))
                else:
                    # Should not happen if integer, but fallback
                    directed.append((u, v))
            
            # Decompose into paths
            adj = defaultdict(list)
            indeg = defaultdict(int)
            outdeg = defaultdict(int)
            for u, v in directed:
                adj[u].append(v)
                outdeg[u] += 1
                indeg[v] += 1
            
            paths = []
            visited_edges = set()
            
            starts = []
            for v in range(n):
                if outdeg[v] > indeg[v]:
                    starts.append(v)
            
            for start in starts:
                path = [start]
                cur = start
                while True:
                    found = False
                    for nbr in adj[cur]:
                        e = (cur, nbr)
                        if e not in visited_edges:
                            visited_edges.add(e)
                            path.append(nbr)
                            cur = nbr
                            found = True
                            break
                    if not found:
                        break
                paths.append(path)
            
            remaining_edges = set(directed) - visited_edges
            while remaining_edges:
                start_edge = next(iter(remaining_edges))
                path = [start_edge[0], start_edge[1]]
                visited_edges.add(start_edge)
                remaining_edges.discard(start_edge)
                cur = start_edge[1]
                while cur != path[0]:
                    found = False
                    for nbr in adj[cur]:
                        e = (cur, nbr)
                        if e in remaining_edges:
                            remaining_edges.discard(e)
                            visited_edges.add(e)
                            path.append(nbr)
                            cur = nbr
                            found = True
                            break
                    if not found:
                        break
                paths.append(path)
            
            return paths
        
        except Exception:
            return [[u, v] for u, v in edges]

    def greedy_path_cover(edges, time_limit):
        """Fallback: repeatedly extract longest simple path from remaining edges."""
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        remaining = set(edges)
        paths = []
        
        while remaining:
            if time.time() > time_limit:
                for u, v in remaining:
                    paths.append([u, v])
                break
            
            best_path = None
            verts = list(adj.keys())
            verts.sort(key=lambda v: len(adj[v]), reverse=True)
            
            for start in verts:
                if not any((start, v) in remaining or (v, start) in remaining for v in adj[start]):
                    continue
                best = _dfs_longest(start, remaining, adj, time_limit)
                if best and (best_path is None or len(best) > len(best_path)):
                    best_path = best
                if best_path and len(best_path) > 2 * len(remaining) / max(1, len(verts)):
                    break
            
            if best_path is None or len(best_path) < 2:
                for u, v in remaining:
                    paths.append([u, v])
                break
            
            paths.append(best_path)
            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i + 1]
                e = tuple(sorted((u, v)))
                remaining.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)
        
        return paths

    def _dfs_longest(start, remaining, adj, time_limit):
        best = [start]
        
        def dfs(node, path):
            nonlocal best
            if time.time() > time_limit:
                return
            if len(path) > len(best):
                best = path[:]
            for nbr in list(adj[node]):
                e = tuple(sorted((node, nbr)))
                if e in remaining and nbr not in path:
                    path.append(nbr)
                    dfs(nbr, path)
                    path.pop()
                    if time.time() > time_limit:
                        return
        
        dfs(start, [start])
        return best

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        elapsed = time.time() - t0
        remaining_time = DEADLINE - elapsed
        if remaining_time < 5:
            # Not enough time; use trivial
            results.append([[u, v] for u, v in edges])
            continue
        
        # Try MILP approach first
        per_graph_limit = min(remaining_time - 2, 45.0)
        paths = path_cover_milp(edges, time.time() + per_graph_limit)
        
        # Validate: check that paths are valid (simple, use edges, edge-disjoint, cover all)
        if not _validate_paths(paths, edges):
            # Fallback to greedy
            paths = greedy_path_cover(edges, time.time() + min(remaining_time - 2, 45.0))
        
        results.append(paths)
    
    # Fill in any missing graphs
    while len(results) < 3:
        spec = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])
    
    return results[:3]

def _validate_paths(paths, edges):
    """Validate that paths form a valid edge-disjoint path cover."""
    if not paths:
        return len(edges) == 0
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    used_edges = set()
    
    for path in paths:
        if len(path) < 2:
            return False
        # Check simple (no repeated vertices)
        if len(set(path)) != len(path):
            return False
        # Check edges exist
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            e = tuple(sorted((u, v)))
            if e not in edge_set:
                return False
            if e in used_edges:
                return False
            used_edges.add(e)
    
    # Check all edges covered
    if used_edges != edge_set:
        return False
    
    return True
# EVOLVE-BLOCK-END
