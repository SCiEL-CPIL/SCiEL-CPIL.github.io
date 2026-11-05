import time
import random
import sys
from collections import defaultdict
import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_MILP = True
except ImportError:
    HAS_MILP = False

sys.setrecursionlimit(100000)


def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _find_cycles_in_graph(adj, n, rng, num_cycles=200):
    """
    Find a diverse pool of cycles using randomized DFS.
    Returns a list of cycles, where each cycle is a list of vertices.
    """
    cycles = []
    # To avoid duplicates in the pool, we can use a set of frozensets of edges
    seen_edge_sets = set()
    
    for _ in range(num_cycles * 2): # Try more to get enough unique
        if len(cycles) >= num_cycles:
            break
        if time.time() > 0: # Placeholder, will pass deadline
            break
            
        start = rng.choice(range(n))
        if not adj[start]:
            continue
        
        path = [start]
        visited = {start}
        current = start
        
        # Randomized walk
        max_steps = n * 2
        for _ in range(max_steps):
            neighbors = list(adj[current])
            if not neighbors:
                break
            
            # Prefer unvisited, but sometimes allow visiting to find other cycles?
            # For simple cycle, we must not revisit any vertex except start at the end.
            unvisited = [v for v in neighbors if v not in visited]
            
            if unvisited:
                # Heuristic: pick neighbor with highest degree or random
                if rng.random() < 0.3:
                    next_v = max(unvisited, key=lambda x: len(adj[x]))
                else:
                    next_v = rng.choice(unvisited)
                path.append(next_v)
                visited.add(next_v)
                current = next_v
            else:
                # No unvisited neighbors. Check if we can close cycle
                if start in adj[current] and len(path) >= 3:
                    # Found a cycle
                    cycle_edges = set()
                    for i in range(len(path)):
                        u, v = path[i], path[(i + 1) % len(path)]
                        cycle_edges.add(tuple(sorted((u, v))))
                    
                    # Check if this cycle is new (edge set not seen)
                    edge_key = frozenset(cycle_edges)
                    if edge_key not in seen_edge_sets:
                        seen_edge_sets.add(edge_key)
                        cycles.append(path[:])
                    break
                else:
                    # Dead end, restart or backtrack? 
                    # For simplicity, restart this walk
                    break
    
    return cycles


def _decompose_greedy(edges, n, deadline, rng):
    """
    Greedy decomposition: repeatedly find a cycle and remove it.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    remaining = len(edges)

    while remaining > 0:
        if time.time() > deadline:
            break

        # Try to find a long cycle using randomized greedy approach
        best_cycle = None
        best_len = 0
        
        num_starts = min(n, 20)
        starts = rng.sample(range(n), num_starts) if n >= num_starts else list(range(n))
        
        for start in starts:
            if not adj[start]:
                continue
            for _ in range(5):
                path = [start]
                visited = {start}
                current = start
                
                while True:
                    neighbors = list(adj[current])
                    unvisited = [v for v in neighbors if v not in visited]
                    
                    if not unvisited:
                        if start in adj[current] and len(path) >= 3:
                            if len(path) > best_len:
                                best_len = len(path)
                                best_cycle = path[:]
                        break
                    else:
                        if rng.random() < 0.5:
                            best_next = max(unvisited, key=lambda x: len([y for y in adj[x] if y not in visited]))
                        else:
                            best_next = rng.choice(unvisited)
                        
                        path.append(best_next)
                        visited.add(best_next)
                        current = best_next

        if best_cycle:
            m = len(best_cycle)
            for i in range(m):
                a = best_cycle[i]
                b = best_cycle[(i + 1) % m]
                adj[a].discard(b)
                adj[b].discard(a)
            remaining -= m
            pieces.append(best_cycle)
        else:
            edge_found = False
            for u in range(n):
                for v in list(adj[u]):
                    if u < v:
                        adj[u].discard(v)
                        adj[v].discard(u)
                        remaining -= 1
                        pieces.append([u, v])
                        edge_found = True
                        break
                if edge_found:
                    break
            if not edge_found:
                break

    if remaining > 0:
        for u in range(n):
            for v in list(adj[u]):
                if u < v:
                    pieces.append([u, v])
    
    return pieces


def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')
    
    # 1. Get a baseline with greedy
    rng_base = random.Random(42)
    greedy_pieces = _decompose_greedy(edges, n, deadline - 10.0, rng_base)
    best_count = len(greedy_pieces)
    best = greedy_pieces
    
    # 2. Try ILP if available and time permits
    if HAS_MILP:
        # Generate pool
        adj_pool = [set() for _ in range(n)]
        for u, v in edges:
            adj_pool[u].add(v)
            adj_pool[v].add(u)
            
        pool_cycles = _find_cycles_in_graph(adj_pool, n, random.Random(123), num_cycles=300)
        
        if pool_cycles:
            # Map edges to indices
            edge_to_idx = {e: i for i, e in enumerate(edges)}
            m = len(edges)
            k = len(pool_cycles)
            
            # A: m x k matrix, A[i, j] = 1 if edge i is in cycle j
            A = np.zeros((m, k))
            for j, cycle in enumerate(pool_cycles):
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    e = tuple(sorted((u, v)))
                    if e in edge_to_idx:
                        A[edge_to_idx[e], j] = 1
                    else:
                        # Should not happen if pool is valid
                        pass
            
            # Variables: x (k cycles), y (m edges)
            # Objective: min sum(x) + sum(y)
            c = np.ones(k + m)
            
            # Constraints: A @ x + I @ y = 1
            # M = [A, I]
            M = np.hstack([A, np.eye(m)])
            
            constraints = LinearConstraint(M, 1, 1)
            integrality = np.array([1] * k + [0] * m) # x are integer (binary), y can be continuous? 
            # Actually y should be binary too, but if x covers it, y=0. 
            # Since A has 0/1 and we want sum=1, and x,y >= 0, 
            # if x is binary, y will naturally be 0 or 1 if we add integrality or if LP relaxation is tight.
            # Let's force y to be binary as well to be safe.
            integrality = np.array([1] * (k + m))
            
            bounds = Bounds(lb=np.zeros(k + m), ub=np.ones(k + m))
            
            try:
                res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
                if res.status == 0:
                    x_sol = res.x[:k]
                    y_sol = res.x[k:]
                    
                    new_pieces = []
                    # Add selected cycles
                    for j in range(k):
                        if x_sol[j] > 0.5:
                            new_pieces.append(pool_cycles[j])
                    
                    # Add selected single edges
                    for i in range(m):
                        if y_sol[i] > 0.5:
                            u, v = edges[i]
                            new_pieces.append([u, v])
                    
                    # Verify validity
                    if len(new_pieces) < best_count:
                        # Check edge disjointness
                        used_edges = set()
                        valid = True
                        for p in new_pieces:
                            if len(p) == 2:
                                e = tuple(sorted(p))
                                if e in used_edges:
                                    valid = False
                                    break
                                used_edges.add(e)
                            else:
                                for i in range(len(p)):
                                    u, v = p[i], p[(i + 1) % len(p)]
                                    e = tuple(sorted((u, v)))
                                    if e in used_edges:
                                        valid = False
                                        break
                                    used_edges.add(e)
                                if not valid:
                                    break
                        
                        if valid and len(used_edges) == m:
                            best_count = len(new_pieces)
                            best = new_pieces
            except Exception:
                pass
                
    return best if best else [[u, v] for u, v in edges]


def construct():
    start_time = time.time()
    total_budget = 145.0
    per_graph_budget = total_budget / 3.0
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        graph_deadline = start_time + per_graph_budget * (i + 1)
        
        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0
        
        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)
        
        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break
    
    return results
