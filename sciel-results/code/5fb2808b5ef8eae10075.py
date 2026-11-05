import time
from collections import defaultdict
import numpy as np

def instance(n, p, seed):
    import random
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def decompose_milp(n, edges):
    if not edges:
        return []
    
    # Build adjacency and degrees
    deg = [0] * n
    adj = defaultdict(list)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
        adj[u].append(v)
        adj[v].append(u)
        
    # Map edges to indices
    edge_list = list(edges)
    m = len(edge_list)
    
    # Variables:
    # x[i][0] for edge i direction 0 (u->v)
    # x[i][1] for edge i direction 1 (v->u)
    # s[v] for vertex v
    # t[v] for vertex v (positive part of s)
    
    num_x = m * 2
    num_s = n
    num_t = n
    total_vars = num_x + num_s + num_t
    
    # Objective: minimize sum(t)
    c = np.zeros(total_vars)
    for i in range(num_t):
        c[num_x + num_s + i] = 1.0
        
    # Constraints:
    # 1. x[i][0] + x[i][1] = 1
    # 2. out(v) - in(v) = s[v]
    # 3. t[v] - s[v] >= 0  => -t[v] + s[v] <= 0
    # 4. t[v] >= 0 (handled by bounds)
    # 5. s[v] bounds: -deg[v] <= s[v] <= deg[v]
    
    # We will use scipy.optimize.milp
    # A_ub * x <= b_ub
    # A_eq * x = b_eq
    
    # Equality constraints:
    # 1. Edge direction sum
    A_eq = np.zeros((m + n, total_vars))
    b_eq = np.zeros(m + n)
    
    for i in range(m):
        u, v = edge_list[i]
        A_eq[i, 2*i] = 1.0
        A_eq[i, 2*i+1] = 1.0
        b_eq[i] = 1.0
        
    # 2. Flow conservation: sum_out - sum_in = s[v]
    # sum_out(v) = sum_{(u,v) in E} x_{uv}
    # sum_in(v) = sum_{(v,w) in E} x_{vw}
    # Equation: sum_{(u,v)} x_{uv} - sum_{(v,w)} x_{vw} - s[v] = 0
    
    for v in range(n):
        row = m + v
        # out edges: edges where v is the second node?
        # We need to check orientation in edge_list
        # For each edge i (u, v):
        # if u == v: impossible (no self loops in simple graph generation usually, but check)
        # if edge is (a, b):
        #   x[2*i] is a->b, x[2*i+1] is b->a
        
        for i, (u, w) in enumerate(edge_list):
            if u == v:
                # a->v is x[2*i]
                A_eq[row, 2*i] += 1.0
            elif w == v:
                # v->a is x[2*i+1]
                A_eq[row, 2*i+1] += 1.0
                # a->v is x[2*i]
                A_eq[row, 2*i] -= 1.0
            # else: v not incident
        
        # subtract s[v]
        A_eq[row, num_x + v] = -1.0
        b_eq[row] = 0.0
        
    # Inequality constraints:
    # t[v] - s[v] >= 0  =>  s[v] - t[v] <= 0
    A_ub = np.zeros((n, total_vars))
    b_ub = np.zeros(n)
    
    for v in range(n):
        A_ub[v, num_x + v] = 1.0       # s[v]
        A_ub[v, num_x + num_s + v] = -1.0 # -t[v]
        b_ub[v] = 0.0
        
    # Bounds
    # x[i] in {0, 1}
    # s[v] in [-deg[v], deg[v]]
    # t[v] in [0, inf]
    
    bounds = []
    for i in range(num_x):
        bounds.append((0, 1))
    for v in range(n):
        bounds.append((-deg[v], deg[v]))
    for v in range(n):
        bounds.append((0, None))
        
    # Integrality
    integrality = np.zeros(total_vars)
    for i in range(num_x):
        integrality[i] = 1
    for v in range(n):
        integrality[num_x + v] = 1
    # t[v] can be continuous, but since s[v] is integer and t[v] >= s[v] and we minimize sum t, t[v] will be integer.
    # Actually, if s[v] = 1.5, t[v] >= 1.5. But s[v] is integer. So t[v] will be integer.
    # We can mark t[v] as integer too to be safe? Or leave continuous.
    # Leaving continuous is faster.
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        # Construct constraints in scipy format
        cons = [
            LinearConstraint(A_eq, lb=b_eq, ub=b_eq),
            LinearConstraint(A_ub, lb=-np.inf, ub=b_ub)
        ]
        bounds_scipy = Bounds(lb=[b[0] for b in bounds], ub=[b[1] if b[1] is not None else np.inf for b in bounds])
        
        res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds_scipy)
        
        if res.success:
            # Extract solution
            x_vals = res.x[:num_x]
            s_vals = res.x[num_x:num_x+num_s]
            
            # Reconstruct paths
            # We have a directed graph defined by x_vals.
            # x_vals[2*i] == 1 means edge i is oriented u->v.
            # x_vals[2*i+1] == 1 means edge i is oriented v->u.
            
            out_adj = defaultdict(list)
            in_adj = defaultdict(list)
            
            for i, (u, v) in enumerate(edge_list):
                if x_vals[2*i] > 0.5:
                    out_adj[u].append(v)
                    in_adj[v].append(u)
                else:
                    out_adj[v].append(u)
                    in_adj[u].append(v)
                    
            # Find paths
            # Vertices with out > in are starts (s[v] > 0)
            # Vertices with in > out are ends (s[v] < 0)
            # Vertices with out == in are internal or isolated (but all edges covered, so internal or cycle nodes)
            
            # We need to trace paths.
            # Start from vertices with s[v] > 0.
            # If there are cycles (s[v]=0 for all nodes in a component), we need to break them.
            # However, the objective minimized sum(t) which is sum of positive s.
            # If there is a cycle, s=0, so it doesn't add to cost.
            # But a cycle is not a path.
            # We must break cycles into paths.
            
            # Strategy:
            # 1. Identify all directed edges.
            # 2. Find all vertices with out_degree > in_degree (starts).
            # 3. Find all vertices with in_degree > out_degree (ends).
            # 4. Trace from starts to ends.
            # 5. If there are remaining edges (cycles), pick a vertex in a cycle, break it, and treat it as start/end.
            
            paths = []
            visited_edges = set()
            
            # Calculate current in/out degrees from the flow
            cur_out = defaultdict(int)
            cur_in = defaultdict(int)
            for u, v in out_adj.items():
                for w in v:
                    cur_out[u] += 1
                    cur_in[w] += 1
            
            # Starts: cur_out > cur_in
            # Ends: cur_in > cur_out
            
            # We will greedily trace paths.
            # To ensure we cover all edges, we must handle cycles.
            
            # Let's just find all paths in the directed graph.
            # A "path" in this context is a maximal directed trail?
            # No, the flow is a set of edge-disjoint paths and cycles.
            
            # Algorithm to decompose flow into paths:
            # 1. While there are edges:
            #    a. Find a vertex v with out_degree > 0.
            #    b. If in_degree(v) < out_degree(v), start at v.
            #    c. Else, we are in a cycle. Pick any edge (u,v) in the cycle, remove it, and start at v?
            #       Actually, if we are in a cycle, we can start anywhere.
            #       Let's pick a vertex v with out_degree > 0.
            #       If in_degree(v) == out_degree(v), it's a cycle node.
            #       We can start a path at v.
            
            remaining_edges = set()
            for i, (u, v) in enumerate(edge_list):
                if x_vals[2*i] > 0.5:
                    remaining_edges.add((u, v))
                else:
                    remaining_edges.add((v, u))
                    
            # Adjacency for remaining edges
            rem_adj = defaultdict(list)
            for u, v in remaining_edges:
                rem_adj[u].append(v)
                
            while remaining_edges:
                # Find a start vertex
                # A vertex is a start if it has outgoing edges and (in_degree < out_degree OR we just pick one)
                # To minimize path count, we should only start new paths when forced (in < out).
                # But if we are in a cycle, in == out. We must start somewhere to break the cycle.
                
                start = None
                # Prefer vertices with out > in
                for v in range(n):
                    if rem_adj[v]:
                        # Check in-degree in remaining graph
                        # This is expensive to calculate every time.
                        # Let's just pick any vertex with outgoing edges.
                        # To be smart, pick one with out > in.
                        pass
                
                # Simple approach:
                # 1. Find all vertices with out > in in the REMAINING graph.
                # 2. If any, start from one of them.
                # 3. If none, pick any vertex with out > 0 (cycle).
                
                # Calculate degrees in remaining graph
                r_out = defaultdict(int)
                r_in = defaultdict(int)
                for u, v in remaining_edges:
                    r_out[u] += 1
                    r_in[v] += 1
                    
                starts = [v for v in r_out if r_out[v] > r_in[v]]
                
                if starts:
                    start = starts[0]
                else:
                    # Pick any vertex with outgoing edge
                    for v in r_out:
                        if r_out[v] > 0:
                            start = v
                            break
                            
                if start is None:
                    break # Should not happen if remaining_edges not empty
                    
                path = [start]
                curr = start
                path_set = {start}
                
                while True:
                    # Find next edge
                    next_v = None
                    for w in rem_adj[curr]:
                        if (curr, w) in remaining_edges:
                            next_v = w
                            break
                    if next_v is None:
                        break
                    remaining_edges.remove((curr, next_v))
                    path.append(next_v)
                    path_set.add(next_v)
                    curr = next_v
                    
                paths.append(path)
                
            return paths
            
        else:
            # Fallback to greedy if MILP fails
            return decompose_greedy(n, edges)
            
    except Exception:
        return decompose_greedy(n, edges)

def decompose_greedy(n, edges):
    if not edges:
        return []
    
    remaining = set(edges)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    paths = []
    
    while remaining:
        best_edge = None
        best_degree_sum = -1
        for e in remaining:
            u, v = e
            deg_sum = len(adj[u]) + len(adj[v])
            if deg_sum > best_degree_sum:
                best_degree_sum = deg_sum
                best_edge = e
        
        u, v = best_edge
        remaining.remove((u, v))
        adj[u].discard(v)
        adj[v].discard(u)
        
        path = [u, v]
        path_set = {u, v}
        
        curr = v
        while True:
            next_v = None
            for w in adj[curr]:
                if w not in path_set:
                    next_v = w
                    break
            if next_v is None:
                break
            e2 = tuple(sorted((curr, next_v)))
            if e2 in remaining:
                remaining.remove(e2)
                adj[curr].discard(next_v)
                adj[next_v].discard(curr)
                path.append(next_v)
                path_set.add(next_v)
                curr = next_v
            else:
                break

        curr = u
        while True:
            next_v = None
            for w in adj[curr]:
                if w not in path_set:
                    next_v = w
                    break
            if next_v is None:
                break
            e2 = tuple(sorted((curr, next_v)))
            if e2 in remaining:
                remaining.remove(e2)
                adj[curr].discard(next_v)
                adj[next_v].discard(curr)
                path.insert(0, next_v)
                path_set.add(next_v)
                curr = next_v
            else:
                break

        paths.append(path)
        
    return paths

def construct():
    deadline = time.time() + 140
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        # Try MILP
        paths = decompose_milp(n, edges)
        
        # Verify validity
        # 1. Simple paths
        # 2. Use graph edges
        # 3. Edge-disjoint
        # 4. Cover all edges
        
        valid = True
        used_edges = set()
        for path in paths:
            if len(path) < 2:
                valid = False
                break
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                e = tuple(sorted((u, v)))
                if e not in set(edges):
                    valid = False
                    break
                if e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            # Check simple
            if len(set(path)) != len(path):
                valid = False
                break
            if valid:
                continue
            break
        
        if not valid or len(used_edges) != len(edges):
            # Fallback
            if time.time() < deadline:
                paths = decompose_greedy(n, edges)
            else:
                paths = decompose_greedy(n, edges)
                
        results.append(paths)
        
    return results
