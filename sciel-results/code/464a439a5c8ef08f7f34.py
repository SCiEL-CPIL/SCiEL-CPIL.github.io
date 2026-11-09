"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    t0 = time.time()
    DEADLINE = 145.0

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_orientation_milp(edges, time_limit):
        if not edges:
            return {}
        
        m = len(edges)
        n = max(max(u, v) for u, v in edges) + 1
        
        # Variables: y_i for i in range(m), y_i in {0, 1}
        # y_i = 1 means edge i is oriented u_i -> v_i
        # y_i = 0 means edge i is oriented v_i -> u_i
        
        # Objective: minimize sum(t_v)
        # t_v >= b_v, t_v >= -b_v
        # b_v = out(v) - in(v)
        
        # Coefficients for b_v in terms of y_i
        # For edge i = (u, v):
        # if y_i = 1: u->v. b_u += 1, b_v -= 1.
        # if y_i = 0: v->u. b_u -= 1, b_v += 1.
        # b_u = 2*y_i - 1
        # b_v = -2*y_i + 1
        
        # So b_v = sum_{i in delta(v)} c_{v,i} * y_i + d_v
        # where c_{v,i} = 2 if v is u_i, -2 if v is v_i.
        # and d_v = -1 if v is u_i, 1 if v is v_i.
        
        c_mat = np.zeros((n, m))
        d_vec = np.zeros(n)
        
        for i, (u, v) in enumerate(edges):
            c_mat[u, i] = 2.0
            d_vec[u] -= 1.0
            c_mat[v, i] = -2.0
            d_vec[v] += 1.0
            
        # Variables: y (m vars) + t (n vars) = m + n vars
        # Order: y_0...y_{m-1}, t_0...t_{n-1}
        
        c_obj = np.zeros(m + n)
        c_obj[m:] = 1.0 # Minimize sum t
        
        # Constraints:
        # 1) t_v - (c_mat[v] . y + d_vec[v]) >= 0  =>  -t_v + c_mat[v] . y <= -d_vec[v]
        #    Wait, t_v >= b_v => b_v - t_v <= 0
        #    (c_mat[v] . y + d_vec[v]) - t_v <= 0
        # 2) t_v >= -b_v => -b_v - t_v <= 0
        #    -(c_mat[v] . y + d_vec[v]) - t_v <= 0
        
        A_ub = np.zeros((2 * n, m + n))
        b_ub = np.zeros(2 * n)
        
        for v in range(n):
            # Constraint 1: c_mat[v] . y - t_v <= -d_vec[v]
            A_ub[v, :m] = c_mat[v]
            A_ub[v, m + v] = -1.0
            b_ub[v] = -d_vec[v]
            
            # Constraint 2: -c_mat[v] . y - t_v <= d_vec[v]
            A_ub[n + v, :m] = -c_mat[v]
            A_ub[n + v, m + v] = -1.0
            b_ub[n + v] = d_vec[v]
            
        # Bounds
        bounds = Bounds(lb=np.zeros(m + n), ub=np.ones(m + n))
        # t_v can be larger than 1, so ub for t is inf
        bounds.ub[m:] = np.inf
        
        # Integrality
        integrality = np.zeros(m + n)
        integrality[:m] = 1.0 # y must be integer
        
        try:
            res = milp(
                c=c_obj,
                constraints=LinearConstraint(A_ub, ub=b_ub),
                integrality=integrality,
                bounds=bounds
            )
            
            if not res.success:
                return None
                
            y = res.x[:m]
            orientation = {}
            for i, (u, v) in enumerate(edges):
                if y[i] > 0.5:
                    orientation[(u, v)] = (u, v)
                else:
                    orientation[(u, v)] = (v, u)
            return orientation
            
        except Exception:
            return None

    def decompose_to_trails(orientation):
        adj = defaultdict(list)
        indeg = defaultdict(int)
        outdeg = defaultdict(int)
        
        for (u, v), (su, sv) in orientation.items():
            adj[su].append(sv)
            outdeg[su] += 1
            indeg[sv] += 1
            
        trails = []
        visited_edges = set()
        
        # Start from nodes with outdeg > indeg
        starts = [v for v in adj if outdeg[v] > indeg[v]]
        
        # If no starts, we have cycles. Pick any node with outdeg > 0.
        if not starts:
            starts = [v for v in adj if outdeg[v] > 0]
            
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
            if len(path) > 1:
                trails.append(path)
                
        # Handle remaining cycles (if any edges not visited)
        remaining_edges = set(orientation.keys()) - set((u, v) for u, v in visited_edges)
        # Note: visited_edges stores directed edges (su, sv)
        # orientation keys are undirected (u,v)
        # We need to map back.
        
        # Reconstruct remaining undirected edges
        rem_undirected = set()
        for (u, v), (su, sv) in orientation.items():
            if (su, sv) not in visited_edges:
                rem_undirected.add((u, v))
                
        while rem_undirected:
            start_edge = next(iter(rem_undirected))
            u, v = start_edge
            su, sv = orientation[(u, v)]
            path = [su, sv]
            visited_edges.add((su, sv))
            rem_undirected.discard((u, v))
            cur = sv
            while cur != su:
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
            if len(path) > 1:
                trails.append(path)
                
        return trails

    def split_trails_to_simple_paths(trails):
        simple_paths = []
        for trail in trails:
            if len(trail) <= 2:
                simple_paths.append(trail)
                continue
            
            # Check for repeated vertices
            # Simple greedy split: whenever we see a repeated vertex, split.
            current_path = [trail[0]]
            seen = {trail[0]}
            
            for i in range(1, len(trail)):
                v = trail[i]
                if v in seen:
                    # Split before v
                    if len(current_path) > 1:
                        simple_paths.append(current_path)
                    current_path = [v]
                    seen = {v}
                else:
                    current_path.append(v)
                    seen.add(v)
            
            if len(current_path) > 1:
                simple_paths.append(current_path)
                
        return simple_paths

    def greedy_path_cover(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        remaining = set(tuple(sorted(e)) for e in edges)
        paths = []
        
        while remaining:
            best_path = None
            verts = list(adj.keys())
            for start in verts:
                if not any(tuple(sorted((start, v))) in remaining for v in adj[start]):
                    continue
                # DFS for longest path
                def dfs(node, path, visited):
                    nonlocal best_path
                    if best_path is None or len(path) > len(best_path):
                        best_path = path[:]
                    for nbr in adj[node]:
                        e = tuple(sorted((node, nbr)))
                        if e in remaining and nbr not in visited:
                            visited.add(nbr)
                            path.append(nbr)
                            dfs(nbr, path, visited)
                            path.pop()
                            visited.remove(nbr)
                            
                dfs(start, [start], {start})
                
            if best_path is None:
                for e in remaining:
                    paths.append(list(e))
                break
                
            paths.append(best_path)
            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i+1]
                e = tuple(sorted((u, v)))
                remaining.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)
                
        return paths

    def validate(paths, edges):
        edge_set = set(tuple(sorted(e)) for e in edges)
        used = set()
        for p in paths:
            if len(p) < 2: return False
            if len(set(p)) != len(p): return False
            for i in range(len(p)-1):
                e = tuple(sorted((p[i], p[i+1])))
                if e not in edge_set: return False
                if e in used: return False
                used.add(e)
        return used == edge_set

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        elapsed = time.time() - t0
        if time.time() > DEADLINE - 5:
            results.append(greedy_path_cover(edges))
            continue
            
        orientation = solve_orientation_milp(edges, DEADLINE - 5)
        if orientation:
            trails = decompose_to_trails(orientation)
            paths = split_trails_to_simple_paths(trails)
            if validate(paths, edges):
                results.append(paths)
            else:
                results.append(greedy_path_cover(edges))
        else:
            results.append(greedy_path_cover(edges))
            
    return results

# EVOLVE-BLOCK-END
