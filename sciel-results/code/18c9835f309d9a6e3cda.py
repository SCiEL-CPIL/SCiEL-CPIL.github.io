import time
import random
import sys
from collections import defaultdict
from scipy.optimize import linprog
import numpy as np

def _instance(n, p, seed):
    """Exact copy of the verifier's instance generator."""
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _greedy_walk(adj, remaining, deg, start_v, rng=None, randomize=False, mode="high_deg"):
    """
    Perform a single greedy walk from start_v, returning the path and updating
    remaining and deg in place.
    """
    path = [start_v]
    visited = {start_v}
    
    while True:
        v = path[-1]
        valid_neighbors = []
        for u in adj[v]:
            if u not in visited:
                e = (min(v, u), max(v, u))
                if e in remaining:
                    valid_neighbors.append(u)
        
        if not valid_neighbors:
            break
        
        if randomize and rng is not None:
            if mode == "high_deg":
                weights = [float(deg.get(u, 0) + 1) for u in valid_neighbors]
            elif mode == "low_deg":
                weights = [1.0 / (deg.get(u, 0) + 1) for u in valid_neighbors]
            else:
                weights = [1.0 + deg.get(u, 0) * 0.5 for u in valid_neighbors]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            next_v = valid_neighbors[-1]
            for i, u in enumerate(valid_neighbors):
                cum += weights[i]
                if r <= cum:
                    next_v = u
                    break
        else:
            if mode == "high_deg":
                next_v = max(valid_neighbors, key=lambda u: deg.get(u, 0))
            elif mode == "low_deg":
                next_v = min(valid_neighbors, key=lambda u: deg.get(u, 0))
            else:
                next_v = max(valid_neighbors, key=lambda u: (deg.get(u, 0), -u))
        
        e = (min(v, next_v), max(v, next_v))
        remaining.discard(e)
        deg[v] -= 1
        deg[next_v] -= 1
        path.append(next_v)
        visited.add(next_v)
        
    return path

def _greedy_cover(edges, deadline):
    """Fallback greedy cover."""
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    remaining = set(map(tuple, edges))
    paths = []
    deg = defaultdict(int)
    for e in remaining:
        deg[e[0]] += 1
        deg[e[1]] += 1
        
    while remaining and time.time() < deadline:
        start_candidates = set()
        for e in remaining:
            start_candidates.add(e[0])
            start_candidates.add(e[1])
        
        leaves = [v for v in start_candidates if deg[v] == 1]
        if leaves:
            start_v = leaves[0]
        else:
            min_deg = min(deg[v] for v in start_candidates)
            candidates = [v for v in start_candidates if deg[v] == min_deg]
            start_v = min(candidates)
            
        path = _greedy_walk(adj, remaining, deg, start_v, None, False, "high_deg")
        paths.append(path)
        
    for e in remaining:
        paths.append(list(e))
        
    return paths

def _solve_lp_cover(edges, n):
    """
    Solve the Minimum Edge Path Cover using Linear Programming.
    """
    if not edges:
        return []
        
    # Variable indices
    # x_uv for directed edges
    # s_v for start counts
    # t_v for end counts
    
    # Map vertices to 0..n-1
    # We assume vertices are 0..n-1
    
    # List of directed edges
    dir_edges = []
    for u, v in edges:
        dir_edges.append((u, v))
        dir_edges.append((v, u))
        
    num_dir_edges = len(dir_edges)
    num_vertices = n
    
    # Total variables: num_dir_edges + 2 * num_vertices
    num_vars = num_dir_edges + 2 * num_vertices
    
    # Objective: minimize sum(s_v)
    c = [0.0] * num_vars
    for v in range(num_vertices):
        c[num_dir_edges + v] = 1.0 # s_v
        
    # Constraints:
    # 1. x_uv + x_vu = 1 for each undirected edge {u,v}
    # 2. Flow balance: sum_in(v) - sum_out(v) + s_v - t_v = 0  => sum_in - sum_out + s_v - t_v = 0
    #    Wait, standard flow conservation:
    #    In(v) + s_v = Out(v) + t_v
    #    Out(v) - In(v) + t_v - s_v = 0
    #    Let's use: sum_{u!=v} x_uv - sum_{u!=v} x_vu + t_v - s_v = 0
    
    A_eq = []
    b_eq = []
    
    # Constraint 1: Edge coverage
    for i, (u, v) in enumerate(edges):
        row = [0.0] * num_vars
        # Find index of (u,v) and (v,u) in dir_edges
        idx_uv = dir_edges.index((u, v))
        idx_vu = dir_edges.index((v, u))
        row[idx_uv] = 1.0
        row[idx_vu] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)
        
    # Constraint 2: Flow balance
    # Precompute indices for faster access
    # out_idx[v] = list of indices in dir_edges where u == v
    # in_idx[v] = list of indices in dir_edges where v == v (target)
    out_idx = defaultdict(list)
    in_idx = defaultdict(list)
    for i, (u, v) in enumerate(dir_edges):
        out_idx[u].append(i)
        in_idx[v].append(i)
        
    for v in range(num_vertices):
        row = [0.0] * num_vars
        # Outgoing edges: x_uv
        for idx in out_idx[v]:
            row[idx] = 1.0
        # Incoming edges: x_vu (where v is target)
        for idx in in_idx[v]:
            row[idx] = -1.0
        # s_v and t_v
        # Equation: Out - In + t_v - s_v = 0
        # So coeff of t_v is 1, coeff of s_v is -1
        row[num_dir_edges + v] = -1.0       # s_v
        row[num_dir_edges + num_vertices + v] = 1.0 # t_v
        A_eq.append(row)
        b_eq.append(0.0)
        
    # Bounds
    # x_uv in [0, 1]
    # s_v, t_v in [0, inf]
    bounds = []
    for i in range(num_dir_edges):
        bounds.append((0, 1))
    for i in range(2 * num_vertices):
        bounds.append((0, None))
        
    # Solve
    try:
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.status == 0:
            # Extract x
            x = res.x[:num_dir_edges]
            # Reconstruct paths
            # Directed graph
            adj_dir = defaultdict(list)
            for i, (u, v) in enumerate(dir_edges):
                if x[i] > 0.5:
                    adj_dir[u].append(v)
                    
            # Find paths
            # Starts are vertices with s_v > 0.5
            starts = []
            for v in range(num_vertices):
                if res.x[num_dir_edges + v] > 0.5:
                    starts.append(v)
                    
            paths = []
            visited_edges = set()
            
            for start in starts:
                path = [start]
                cur = start
                while True:
                    # Find next
                    next_v = None
                    for u in adj_dir[cur]:
                        e = (min(cur, u), max(cur, u))
                        if e not in visited_edges:
                            next_v = u
                            break
                    if next_v is None:
                        break
                    visited_edges.add((min(cur, next_v), max(cur, next_v)))
                    path.append(next_v)
                    cur = next_v
                paths.append(path)
                
            # Check if all edges covered
            if len(visited_edges) == len(edges):
                return paths
            else:
                # Fallback
                return _greedy_cover(edges, time.time() + 5)
        else:
            return _greedy_cover(edges, time.time() + 5)
    except Exception:
        return _greedy_cover(edges, time.time() + 5)

def construct():
    """
    Returns three lists of paths (vertex lists), one per fixed instance graph.
    """
    total_deadline = time.time() + 145
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        
        # Try LP first
        # LP is fast for small graphs
        try:
            paths = _solve_lp_cover(edges, n)
        except Exception:
            paths = _greedy_cover(edges, total_deadline - 5)
            
        results.append(paths)
        
        # If time is running out, use greedy for remaining
        if time.time() > total_deadline - 10:
            for j in range(i + 1, len(specs)):
                n_j, p_j, seed_j = specs[j]
                edges_j = _instance(n_j, p_j, seed_j)
                results.append(_greedy_cover(edges_j, total_deadline - 1))
            break
            
    return results

if __name__ == "__main__":
    result = construct()
    for i, paths in enumerate(result):
        print(f"Graph {i}: {len(paths)} paths")
    print(f"Total: {sum(len(p) for p in result)}")
