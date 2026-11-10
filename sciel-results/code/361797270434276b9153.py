import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep safety margin
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_graph(n, edges):
        # Map edges to indices
        edge_list = list(edges)
        m = len(edge_list)
        if m == 0:
            return []
        
        # Build adjacency for quick lookup
        adj = {i: set() for i in range(n)}
        edge_index = {}
        for idx, (u, v) in enumerate(edge_list):
            edge_index[(u, v)] = idx
            edge_index[(v, u)] = idx
            adj[u].add(v)
            adj[v].add(u)
            
        # Variables:
        # x[e] for e in edges: 1 if edge is used in a path (all edges must be used, so x[e]=1)
        # Actually, we need to select a subset of directed edges that form paths.
        # Since graph is undirected, we can model it as a directed graph with both arcs for each undirected edge.
        # Let's create directed arcs.
        arcs = []
        arc_index = {}
        for u, v in edge_list:
            arcs.append((u, v))
            arcs.append((v, u))
            arc_index[(u, v)] = len(arcs) - 1
            arc_index[(v, u)] = len(arcs) - 1
        M = len(arcs)
        
        # Variables:
        # y[a] for each arc a: 1 if arc is used in the path cover.
        # z[v] for each vertex v: 1 if vertex is an endpoint of a path (start or end).
        # Number of paths = sum(z[v]) / 2.
        
        # Constraints:
        # 1. For each undirected edge (u,v), exactly one of the two arcs (u,v) or (v,u) is used.
        #    y[(u,v)] + y[(v,u)] = 1
        # 2. Flow conservation: For each vertex v,
        #    sum(y[in-arc]) - sum(y[out-arc]) = z_start[v] - z_end[v]
        #    Actually, simpler:
        #    Let d[v] = out_degree_used - in_degree_used.
        #    If d[v] = 1, v is a start.
        #    If d[v] = -1, v is an end.
        #    If d[v] = 0, v is internal.
        #    The number of paths is (number of starts + number of ends) / 2.
        #    Since paths are simple and cover edges, the graph formed by used arcs is a collection of paths.
        #    We can just minimize the number of paths.
        #    Let s[v] be 1 if v is a start, e[v] be 1 if v is an end.
        #    s[v] + e[v] <= 1 (a vertex cannot be both start and end in a non-trivial path, unless path length 0 which is not allowed here as we cover edges).
        #    Actually, a path of length 1 (single edge) has a start and an end.
        #    Constraint: out[v] - in[v] = s[v] - e[v].
        #    s[v], e[v] in {0,1}.
        #    Minimize sum(s[v]) (which equals sum(e[v])).
        
        # Variables:
        # y[a] for a in 0..M-1 (binary)
        # s[v] for v in 0..n-1 (binary)
        # e[v] for v in 0..n-1 (binary)
        
        # Total variables: M + 2*n
        # Objective: minimize sum(s[v])
        
        c = np.zeros(M + 2 * n)
        for v in range(n):
            c[M + v] = 1  # s[v]
            # e[v] coefficient is 0, but we know sum(s)=sum(e)
            
        # Bounds
        lb = np.zeros(M + 2 * n)
        ub = np.ones(M + 2 * n)
        integrality = np.ones(M + 2 * n)
        
        # Constraints
        # 1. Edge selection: for each undirected edge (u,v), y[u,v] + y[v,u] = 1
        A_rows = []
        lb_rows = []
        ub_rows = []
        
        for (u, v) in edge_list:
            row = np.zeros(M + 2 * n)
            row[arc_index[(u, v)]] = 1
            row[arc_index[(v, u)]] = 1
            A_rows.append(row)
            lb_rows.append(1)
            ub_rows.append(1)
            
        # 2. Flow conservation: out[v] - in[v] = s[v] - e[v]
        # => out[v] - in[v] - s[v] + e[v] = 0
        for v in range(n):
            row = np.zeros(M + 2 * n)
            # Outgoing arcs
            for u in adj[v]:
                if (v, u) in arc_index:
                    row[arc_index[(v, u)]] = 1
            # Incoming arcs
            for u in adj[v]:
                if (u, v) in arc_index:
                    row[arc_index[(u, v)]] = -1
            # s[v] and e[v]
            row[M + v] = -1
            row[M + n + v] = 1
            A_rows.append(row)
            lb_rows.append(0)
            ub_rows.append(0)
            
        # 3. s[v] + e[v] <= 1 (a vertex cannot be both start and end of the same path component in a way that creates a cycle? 
        # Actually, if s[v]=1 and e[v]=1, then out-in=0, which means it's an internal node or isolated. 
        # But if it's a start and end, it must be a cycle or a single vertex. 
        # Since we are covering edges, and paths are simple, a start and end at the same vertex would imply a cycle. 
        # Cycles are not allowed in a path cover of edges? 
        # The problem asks for "paths". A cycle is not a path. 
        # So s[v] and e[v] cannot both be 1.
        row = np.zeros(M + 2 * n)
        row[M + v] = 1
        row[M + n + v] = 1
        A_rows.append(row)
        lb_rows.append(0)
        ub_rows.append(1)
        
        # Wait, the constraint s[v]+e[v]<=1 should be for all v.
        # I only added it for the last v in the loop above? No, I need to add it for all v.
        
        # Let's rebuild constraints properly.
        A_rows = []
        lb_rows = []
        ub_rows = []
        
        # 1. Edge selection
        for (u, v) in edge_list:
            row = np.zeros(M + 2 * n)
            row[arc_index[(u, v)]] = 1
            row[arc_index[(v, u)]] = 1
            A_rows.append(row)
            lb_rows.append(1)
            ub_rows.append(1)
            
        # 2. Flow conservation for all v
        for v in range(n):
            row = np.zeros(M + 2 * n)
            for u in adj[v]:
                if (v, u) in arc_index:
                    row[arc_index[(v, u)]] = 1
                if (u, v) in arc_index:
                    row[arc_index[(u, v)]] = -1
            row[M + v] = -1
            row[M + n + v] = 1
            A_rows.append(row)
            lb_rows.append(0)
            ub_rows.append(0)
            
        # 3. s[v] + e[v] <= 1 for all v
        for v in range(n):
            row = np.zeros(M + 2 * n)
            row[M + v] = 1
            row[M + n + v] = 1
            A_rows.append(row)
            lb_rows.append(0)
            ub_rows.append(1)
            
        A = np.array(A_rows)
        lb_arr = np.array(lb_rows)
        ub_arr = np.array(ub_rows)
        
        constraints = LinearConstraint(A, lb_arr, ub_arr)
        bounds = Bounds(lb, ub)
        
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.success:
                y_vals = res.x[:M]
                # Reconstruct paths
                # y_vals[a] > 0.5 means arc a is used.
                used_arcs = [(arcs[i][0], arcs[i][1]) for i in range(M) if y_vals[i] > 0.5]
                
                # Build path graph
                path_adj = {v: [] for v in range(n)}
                for u, v in used_arcs:
                    path_adj[u].append(v)
                    
                # Find paths
                # Starts are vertices with out-degree 1 and in-degree 0 in the path graph?
                # Actually, s[v]=1 means start.
                s_vals = res.x[M:M+n]
                starts = [v for v in range(n) if s_vals[v] > 0.5]
                
                paths = []
                for start in starts:
                    path = [start]
                    curr = start
                    while True:
                        nexts = path_adj[curr]
                        if not nexts:
                            break
                        # Should be exactly one next
                        nxt = nexts[0]
                        path.append(nxt)
                        curr = nxt
                    paths.append(path)
                    
                # Verify coverage
                covered_edges = set()
                for p in paths:
                    for i in range(len(p) - 1):
                        u, v = p[i], p[i+1]
                        covered_edges.add(tuple(sorted((u, v))))
                
                if covered_edges == set(edge_list):
                    return paths
                else:
                    # Fallback to trivial solution
                    return [[u, v] for u, v in edge_list]
            else:
                return [[u, v] for u, v in edge_list]
        except Exception:
            return [[u, v] for u, v in edge_list]

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for spec in specs:
        if time.time() > deadline:
            # Return best effort (trivial for remaining)
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
            
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = solve_graph(n, edges)
        results.append(paths)
        
    return results
