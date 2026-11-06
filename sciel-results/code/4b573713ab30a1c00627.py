import sys
import numpy as np
from itertools import combinations
import random
import scipy.sparse as sp

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_MILP = True
except ImportError:
    HAS_MILP = False

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def get_candidates(n, adj, edge_set, rng):
        candidates = set()
        
        # 1. Find all triangles
        for u, v, w in combinations(range(n), 3):
            if (u, v) in edge_set and (v, w) in edge_set and (w, u) in edge_set:
                cycle = [u, v, w]
                min_idx = cycle.index(min(cycle))
                cycle = cycle[min_idx:] + cycle[:min_idx]
                if cycle[1] > cycle[2]:
                    cycle = [cycle[0], cycle[2], cycle[1]]
                candidates.add(tuple(cycle))
                
        # 2. Find 4-cycles
        for u in range(n):
            for v in adj[u]:
                if v <= u: continue
                for w in adj[v]:
                    if w == u or w == v: continue
                    common = adj[w] & adj[u]
                    for x in common:
                        if x == u or x == v or x == w: continue
                        cycle = [u, v, w, x]
                        min_idx = cycle.index(min(cycle))
                        cycle = cycle[min_idx:] + cycle[:min_idx]
                        if cycle[1] > cycle[-1]:
                            cycle = [cycle[0]] + cycle[1:][::-1]
                        candidates.add(tuple(cycle))
                        
        # 3. Find longer cycles using randomized DFS
        max_depth = 12
        for start in range(n):
            for _ in range(50):
                path = [start]
                visited = {start}
                last = start
                while True:
                    neighbors = [v for v in adj[last] if v not in visited]
                    if not neighbors:
                        break
                    next_v = rng.choice(neighbors)
                    path.append(next_v)
                    visited.add(next_v)
                    last = next_v
                    
                    if len(path) >= 5 and start in adj[last]:
                        cycle = path[:]
                        min_idx = cycle.index(min(cycle))
                        cycle = cycle[min_idx:] + cycle[:min_idx]
                        if cycle[1] > cycle[-1]:
                            cycle = [cycle[0]] + cycle[1:][::-1]
                        candidates.add(tuple(cycle))
                        
                    if len(path) >= max_depth:
                        break
                    last = next_v
                    
        return list(candidates)

    def decompose_greedy_set_packing(candidates, edge_set):
        # Sort by length descending
        sorted_candidates = sorted(candidates, key=lambda x: len(x), reverse=True)
        used_edges = set()
        selected = []
        
        for cyc in sorted_candidates:
            valid = True
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i + 1) % len(cyc)]
                e = (min(u, v), max(u, v))
                if e in used_edges:
                    valid = False
                    break
            if valid:
                selected.append(cyc)
                for i in range(len(cyc)):
                    u = cyc[i]
                    v = cyc[(i + 1) % len(cyc)]
                    used_edges.add((min(u, v), max(u, v)))
                    
        return selected, used_edges

    def decompose_milp(edges):
        if not edges:
            return []
        
        n = max(max(u, v) for u, v in edges) + 1
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))
            
        rng = random.Random(42)
        candidates = get_candidates(n, adj, edge_set, rng)
        
        if not candidates:
            return [[u, v] for u, v in sorted(edge_set)]
            
        edge_list = sorted(edge_set)
        edge_index = {e: i for i, e in enumerate(edge_list)}
        num_edges = len(edge_list)
        num_cycles = len(candidates)
        
        selected_cycles = []
        
        if HAS_MILP:
            c = np.zeros(num_cycles)
            for i, cyc in enumerate(candidates):
                c[i] = -(len(cyc) - 1)
                
            rows = []
            cols = []
            vals = []
            
            for i, cyc in enumerate(candidates):
                for j in range(len(cyc)):
                    u = cyc[j]
                    v = cyc[(j + 1) % len(cyc)]
                    e = (min(u, v), max(u, v))
                    idx = edge_index[e]
                    rows.append(idx)
                    cols.append(i)
                    vals.append(1)
                    
            A = sp.csr_matrix((vals, (rows, cols)), shape=(num_edges, num_cycles))
            constraints = LinearConstraint(A, ub=1)
            integrality = np.ones(num_cycles)
            bounds = Bounds(lb=0, ub=1)
            
            try:
                res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
                if res.success:
                    for i in range(num_cycles):
                        if res.x[i] > 0.5:
                            selected_cycles.append(list(candidates[i]))
            except Exception:
                pass # Fall through to greedy if MILP fails
                
        if not selected_cycles:
            # Fallback to greedy set packing
            selected_cycles, _ = decompose_greedy_set_packing(candidates, edge_set)
            
        # Build final pieces
        pieces = []
        used_edges = set()
        for cyc in selected_cycles:
            # Re-verify to be safe against solver errors
            valid = True
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i + 1) % len(cyc)]
                e = (min(u, v), max(u, v))
                if e in used_edges or e not in edge_set:
                    valid = False
                    break
            if valid:
                pieces.append(cyc)
                for i in range(len(cyc)):
                    u = cyc[i]
                    v = cyc[(i + 1) % len(cyc)]
                    used_edges.add((min(u, v), max(u, v)))
                    
        for e in edge_list:
            if e not in used_edges:
                pieces.append(list(e))
                
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose_milp(edges)
        results.append(pieces)
    
    return results
