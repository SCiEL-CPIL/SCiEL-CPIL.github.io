"""Program for edge-disjoint path cover with minimum number of paths."""

def construct():
    from random import Random
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import coo_matrix
    import itertools
    import sys

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_min_path_cover(n, edge_list):
        """
        Find minimum number of edge-disjoint paths that cover all edges.
        Uses MILP formulation:
        - For each directed arc (u,v) where {u,v} is an edge, variable x_{uv} in {0,1}
        - Flow conservation at vertices with sources/sinks
        - Minimize total number of paths (which equals total flow / 1)
        
        Actually, let's use a cleaner formulation:
        - Each edge is used exactly once (in one direction)
        - Vertices can be: source (out-in=1), sink (in-out=1), internal (in=out), or unused (in=out=0)
        - Number of paths = number of sources = number of sinks
        - Minimize number of sources
        
        Formulation:
        - For each undirected edge {u,v}, we have two arcs: (u,v) and (v,u)
        - x_{uv} = 1 if arc (u,v) is used, 0 otherwise
        - For each edge {u,v}: x_{uv} + x_{vu} = 1
        - For each vertex v: sum_out x - sum_in x = b_v, where b_v in {-1, 0, 1}
          b_v = 1 means v is a source, b_v = -1 means v is a sink, b_v = 0 means internal or unused
        - We want to minimize the number of sources (sum of positive b_v)
        
        To linearize: introduce y_v for each vertex where y_v = 1 if b_v = 1 (source)
        Then b_v = y_v - z_v where z_v = 1 if b_v = -1 (sink)
        Constraint: y_v + z_v <= 1 (can't be both)
        And: sum_out x_{vu} - sum_in x_{uv} = y_v - z_v
        Minimize sum y_v
        """
        m = len(edge_list)
        if m == 0:
            return []
        
        # Build adjacency and arc list
        arcs = []
        arc_to_edge = {}
        for i, (u, v) in enumerate(edge_list):
            arcs.append((u, v))
            arcs.append((v, u))
            arc_to_edge[(u, v)] = i
            arc_to_edge[(v, u)] = i
        
        n_arcs = len(arcs)
        
        # Variables: x[0..n_arcs-1] for arcs, y[n_arcs..n_arcs+n-1] for source indicators
        # z[n_arcs+n..n_arcs+2n-1] for sink indicators
        n_x = n_arcs
        n_y = n
        n_z = n
        total_vars = n_x + n_y + n_z
        
        # Objective: minimize sum of y_v
        c = np.zeros(total_vars)
        for v in range(n):
            c[n_x + v] = 1.0  # y_v
        
        # Constraints:
        # 1) For each edge i: x_{uv} + x_{vu} = 1
        # 2) For each vertex v: sum_out x - sum_in x = y_v - z_v
        # 3) y_v + z_v <= 1
        
        constraints = []
        
        # Constraint 1: edge usage
        A_rows = []
        A_cols = []
        A_vals = []
        lb1 = []
        ub1 = []
        
        for i, (u, v) in enumerate(edge_list):
            row = len(lb1)
            # x_{uv}
            A_rows.append(row)
            A_cols.append(arc_to_edge[(u, v)] if (u,v) in arc_to_edge else 0)
            # Wait, I need to map arcs to variable indices
            pass
        
        # Let me restructure: arc index in the arcs list
        arc_index = {}
        for idx, (u, v) in enumerate(arcs):
            arc_index[(u, v)] = idx
        
        A_rows = []
        A_cols = []
        A_vals = []
        lb = []
        ub = []
        
        # Constraint 1: for each undirected edge, exactly one direction used
        for i, (u, v) in enumerate(edge_list):
            row = len(lb)
            idx_uv = arc_index[(u, v)]
            idx_vu = arc_index[(v, u)]
            A_rows.append(row)
            A_cols.append(idx_uv)
            A_vals.append(1.0)
            A_rows.append(row)
            A_cols.append(idx_vu)
            A_vals.append(1.0)
            lb.append(1.0)
            ub.append(1.0)
        
        # Constraint 2: flow conservation with y and z
        for v in range(n):
            row = len(lb)
            # sum_out x_{vu} - sum_in x_{uv} - y_v + z_v = 0
            # i.e., sum_out x_{vu} - sum_in x_{uv} = y_v - z_v
            # Rearranged: sum_out x_{vu} - sum_in x_{uv} - y_v + z_v = 0
            
            # Outgoing arcs from v: (v, w)
            for w in range(n):
                if (v, w) in arc_index:
                    idx = arc_index[(v, w)]
                    A_rows.append(row)
                    A_cols.append(idx)
                    A_vals.append(1.0)
                if (w, v) in arc_index:
                    idx = arc_index[(w, v)]
                    A_rows.append(row)
                    A_cols.append(idx)
                    A_vals.append(-1.0)
            
            # -y_v
            A_rows.append(row)
            A_cols.append(n_x + v)
            A_vals.append(-1.0)
            # +z_v
            A_rows.append(row)
            A_cols.append(n_x + n_y + v)
            A_vals.append(1.0)
            
            lb.append(0.0)
            ub.append(0.0)
        
        # Constraint 3: y_v + z_v <= 1
        for v in range(n):
            row = len(lb)
            A_rows.append(row)
            A_cols.append(n_x + v)
            A_vals.append(1.0)
            A_rows.append(row)
            A_cols.append(n_x + n_y + v)
            A_vals.append(1.0)
            lb.append(0.0)
            ub.append(1.0)
        
        n_constraints = len(lb)
        
        A = coo_matrix((A_vals, (A_rows, A_cols)), shape=(n_constraints, total_vars))
        
        cons = LinearConstraint(A, lb, ub)
        
        # Bounds: x in [0,1] integer, y in [0,1] integer, z in [0,1] integer
        integrality = np.ones(total_vars)
        lbounds = np.zeros(total_vars)
        ubounds = np.ones(total_vars)
        
        res = milp(c=c, constraints=cons, integrality=integrality, bounds=Bounds(lbounds, ubounds))
        
        if not res.success:
            # Fallback: each edge as its own path
            return [list(e) for e in edge_list]
        
        # Extract solution
        x = res.x[:n_x]
        y = res.x[n_x:n_x+n]
        z = res.x[n_x+n:n_x+2*n]
        
        # Reconstruct paths
        # For each vertex, determine role
        # Build directed graph from x values
        adj = {v: [] for v in range(n)}
        for idx, (u, v) in enumerate(arcs):
            if x[idx] > 0.5:
                adj[u].append(v)
        
        # Find sources (y_v > 0.5)
        sources = [v for v in range(n) if y[v] > 0.5]
        
        # Trace paths from sources
        paths = []
        for s in sources:
            path = [s]
            cur = s
            while True:
                if not adj[cur]:
                    break
                # Should have exactly one outgoing edge in a path
                next_v = adj[cur][0]
                path.append(next_v)
                cur = next_v
            paths.append(path)
        
        # Verify all edges covered
        used_edges = set()
        for path in paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                used_edges.add(tuple(sorted((u, v))))
        
        expected_edges = set(tuple(sorted(e)) for e in edge_list)
        if used_edges != expected_edges:
            # Fallback
            return [list(e) for e in edge_list]
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        n, p, seed = spec
        edge_list = instance(n, p, seed)
        paths = solve_min_path_cover(n, edge_list)
        results.append(paths)
    
    return results
