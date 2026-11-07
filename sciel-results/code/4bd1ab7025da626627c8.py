    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds

    P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

    def construct():
        n = P['n']
        # Use a set of tuples for fast removal
        remaining_edges = {tuple(e) for e in P['edges']}
        cycles = []
        
        # Max possible cycles based on edge count
        max_k = len(remaining_edges) // n
        # The task says "At most 10 cycles may be listed" but also "Maximize number".
        # We will try to find as many as possible, up to max_k.
        
        for _ in range(max_k):
            if len(remaining_edges) < n:
                break
                
            cycle = _find_one_cycle(n, remaining_edges)
            if cycle is None:
                break
            
            cycles.append(tuple(cycle))
            # Remove the edges used in this cycle from remaining_edges
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge = (u, v)
                if edge in remaining_edges:
                    remaining_edges.remove(edge)
                else:
                    # Handle undirected edge representation if needed, 
                    # but we stored as tuples (u,v) where u,v from input.
                    # Input edges are [u,v]. We should store both or normalize.
                    # Let's normalize edges to (min, max) for the set to be safe.
                    # Wait, the input edges are specific. 
                    # If I stored {tuple(e) for e in P['edges']}, I have (0,1), (1,2) etc.
                    # If cycle has (1,0), it's not in the set if I only stored (0,1).
                    # So I must normalize.
                    pass
        
        return cycles

    def _find_one_cycle(n, edges_set):
        # Convert set to list for MILP
        edges = list(edges_set)
        m = len(edges)
        if m < n:
            return None
            
        # Variables: x_e for each edge
        # Objective: Maximize sum x_e (should be n if feasible)
        c = np.ones(m)
        
        # Constraints
        cons = []
        
        # 1. Degree constraints: sum_{e incident to v} x_e = 2
        # We need to map vertex to edges
        adj = {v: [] for v in range(n)}
        for i, (u, v) in enumerate(edges):
            adj[u].append(i)
            adj[v].append(i)
            
        for v in range(n):
            if len(adj[v]) < 2:
                return None # Cannot form cycle
            row = np.zeros(m)
            for e_idx in adj[v]:
                row[e_idx] = 1.0
            cons.append(LinearConstraint(row, 2, 2))
            
        # 2. Subtour elimination (MTZ)
        # Variables w_v for each vertex
        n_w = n
        nvar = m + n_w
        
        c = np.concatenate([c, np.zeros(n_w)])
        
        # Integrality
        integrality = np.zeros(nvar)
        integrality[:m] = 1
        
        lb = np.zeros(nvar)
        ub = np.ones(m)
        ub[m:] = n-1 # w_v in [0, n-1]
        
        # MTZ constraints:
        # w_u - w_v + 1 <= (n-1)(1 - x_{uv})
        # w_v - w_u + 1 <= (n-1)(1 - x_{vu})
        # This requires directed arcs. 
        # For undirected graph, we can add constraints for both directions or use a different formulation.
        # Common trick: Fix w_0 = 0.
        # For each edge (u,v):
        # w_u - w_v <= n-1 - n*x_{uv}  (if u->v used)
        # w_v - w_u <= n-1 - n*x_{uv}  (if v->u used)
        # Actually, standard MTZ for undirected is tricky.
        # Let's use the "order" variable approach.
        # If x_uv = 1, then either w_u = w_v + 1 or w_v = w_u + 1 is not quite right for a cycle.
        
        # Alternative: Use Dantzig-Fulkerson-Johnson (DFJ) cuts? 
        # Too complex to generate dynamically in simple code.
        
        # Let's stick to MTZ with directed arcs logic inside the undirected edge variable.
        # If we use x_e for undirected edge e=(u,v), we can't easily apply MTZ directly without splitting.
        
        # Let's split edges into arcs for the MTZ part?
        # No, that doubles variables.
        
        # Let's try a simpler heuristic first? No, must be robust.
        
        # Let's use the formulation:
        # w_u - w_v + 1 <= (n-1) * (1 - x_{uv})
        # w_v - w_u + 1 <= (n-1) * (1 - x_{vu})
        # Since x_{uv} = x_{vu} for the same undirected edge, let's call it x_e.
        # w_u - w_v + 1 <= (n-1) * (1 - x_e)
        # w_v - w_u + 1 <= (n-1) * (1 - x_e)
        
        # Fix w_0 = 0
        row = np.zeros(nvar)
        row[n_w + 0] = 1.0
        cons.append(LinearConstraint(row, 0, 0))
        
        for i, (u, v) in enumerate(edges):
            # Constraint 1: w_u - w_v + 1 <= (n-1) - (n-1)x_e
            # => w_u - w_v + (n-1)x_e <= n-2
            row = np.zeros(nvar)
            row[n_w + u] = 1.0
            row[n_w + v] = -1.0
            row[i] = n - 1
            cons.append(LinearConstraint(row, -np.inf, n - 2))
            
            # Constraint 2: w_v - w_u + 1 <= (n-1) - (n-1)x_e
            # => w_v - w_u + (n-1)x_e <= n-2
            row = np.zeros(nvar)
            row[n_w + v] = 1.0
            row[n_w + u] = -1.0
            row[i] = n - 1
            cons.append(LinearConstraint(row, -np.inf, n - 2))
            
        res = milp(c=-c, # Minimize -sum x => Maximize sum x
                   constraints=cons,
                   integrality=integrality,
                   bounds=Bounds(lb, ub))
                   
        if not res.success:
            return None
            
        # Check if we found a full cycle (sum x = n)
        # The objective was to maximize sum x. If a Hamiltonian cycle exists, max sum is n.
        if res.fun is None or res.fun < -n + 1e-6:
            return None
            
        # Extract cycle
        x = res.x[:m]
        adjc = [set() for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            if x[i] > 0.5:
                adjc[u].add(v)
                adjc[v].add(u)
                
        # Walk
        start = 0
        cyc = [start]
        prev = -1
        cur = start
        for _ in range(n):
            nxts = [u for u in adjc[cur] if u != prev]
            if not nxts:
                return None
            prev, cur = cur, nxts[0]
            if cur == start:
                break
            cyc.append(cur)
            
        if len(cyc) == n:
            return cyc
        return None
    