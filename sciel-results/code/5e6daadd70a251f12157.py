import time
import random
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_degrees(edges):
        degs = defaultdict(int)
        for u, v in edges:
            degs[u] += 1
            degs[v] += 1
        return degs

    def lower_bound(edges):
        if not edges:
            return 0
        degs = get_degrees(edges)
        odd_count = sum(1 for d in degs.values() if d % 2 == 1)
        return max(1, odd_count // 2)

    def solve_milp(edges, n, deadline):
        """
        Formulate path decomposition as MILP.
        Nodes: 0..n-1 are original vertices.
        We create a directed graph for flow.
        """
        if not edges:
            return []
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Line graph nodes:
        # For each v, we have v_in (index v) and v_out (index n+v)
        # Total nodes in flow network: 2*n + 2 (S, T)
        # S = 2*n, T = 2*n+1
        
        # Variables:
        # For each directed arc (u_out -> v_in) for edge (u,v), we have a variable.
        # For each v, we have variable for v_in -> v_out (pass through).
        # For each v, we have variable for S -> v_out (start).
        # For each v, we have variable for v_in -> T (end).
        
        # Let's index the variables.
        # var_idx = {}
        
        # 1. Edge arcs: (u, v) and (v, u)
        # We need to ensure x_uv + x_vu = 1.
        
        # 2. Pass-through arcs: v_in -> v_out
        # 3. Start arcs: S -> v_out
        # 4. End arcs: v_in -> T
        
        # We will use scipy.optimize.milp
        # c: objective coefficients
        # constraints: A_ub x <= b_ub, A_eq x = b_eq
        # bounds: (lb, ub)
        # integrality: 1 for integer variables
        
        # Let's build the variable list
        vars_list = []
        var_index = {}
        
        # Edge variables
        for u, v in edges:
            # Direction u -> v
            # u_out is n+u, v_in is v
            # Arc: (n+u) -> v
            idx = len(vars_list)
            var_index[(n+u, v)] = idx
            vars_list.append({'type': 'edge', 'u': u, 'v': v})
            
            # Direction v -> u
            # v_out is n+v, u_in is u
            # Arc: (n+v) -> u
            idx = len(vars_list)
            var_index[(n+v, u)] = idx
            vars_list.append({'type': 'edge', 'u': v, 'v': u})
            
        # Pass-through variables
        for v in range(n):
            # Arc: v_in (v) -> v_out (n+v)
            idx = len(vars_list)
            var_index[(v, n+v)] = idx
            vars_list.append({'type': 'pass', 'v': v})
            
        # Start variables
        for v in range(n):
            # Arc: S (2n) -> v_out (n+v)
            idx = len(vars_list)
            var_index[(2*n, n+v)] = idx
            vars_list.append({'type': 'start', 'v': v})
            
        # End variables
        for v in range(n):
            # Arc: v_in (v) -> T (2n+1)
            idx = len(vars_list)
            var_index[(v, 2*n+1)] = idx
            vars_list.append({'type': 'end', 'v': v})
            
        num_vars = len(vars_list)
        
        # Objective: Minimize number of paths = sum of start variables
        c = np.zeros(num_vars)
        for i, var in enumerate(vars_list):
            if var['type'] == 'start':
                c[i] = 1.0
            else:
                c[i] = 0.0
                
        # Constraints:
        # 1. Flow conservation at each node in the line graph.
        # Nodes: 0..n-1 (in), n..2n-1 (out), 2n (S), 2n+1 (T)
        # We only need conservation for 0..2n-1. S and T are sources/sinks.
        
        A_eq_rows = []
        b_eq_rows = []
        
        for node in range(2*n):
            inflow_vars = []
            outflow_vars = []
            
            for (u, v), idx in var_index.items():
                if v == node:
                    inflow_vars.append(idx)
                if u == node:
                    outflow_vars.append(idx)
                    
            # Inflow - Outflow = 0
            row = np.zeros(num_vars)
            for idx in inflow_vars:
                row[idx] = 1.0
            for idx in outflow_vars:
                row[idx] = -1.0
                
            A_eq_rows.append(row)
            b_eq_rows.append(0.0)
            
        # 2. Edge coverage: For each undirected edge {u, v}, x_uv + x_vu = 1
        for u, v in edges:
            idx1 = var_index[(n+u, v)] # u_out -> v_in
            idx2 = var_index[(n+v, u)] # v_out -> u_in
            
            row = np.zeros(num_vars)
            row[idx1] = 1.0
            row[idx2] = 1.0
            
            A_eq_rows.append(row)
            b_eq_rows.append(1.0)
            
        A_eq = np.array(A_eq_rows)
        b_eq = np.array(b_eq_rows)
        
        # Bounds: All variables are 0 or 1
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(num_vars)
        
        # Solve
        try:
            res = milp(c=c, constraints=LinearConstraint(A_eq, b_eq, b_eq), bounds=bounds, integrality=integrality)
        except Exception:
            return None
            
        if res.status != 0:
            return None
            
        # Extract paths
        # We need to trace the flow from S to T
        # Build graph of used arcs
        used_arcs = defaultdict(list)
        for i, var in enumerate(vars_list):
            if res.x[i] > 0.5:
                if var['type'] == 'edge':
                    u, v = var['u'], var['v']
                    used_arcs[(n+u, v)].append((u, v))
                elif var['type'] == 'pass':
                    v = var['v']
                    used_arcs[(v, n+v)].append(v)
                elif var['type'] == 'start':
                    v = var['v']
                    used_arcs[(2*n, n+v)].append(v)
                elif var['type'] == 'end':
                    v = var['v']
                    used_arcs[(v, 2*n+1)].append(v)
                    
        paths = []
        # Find all paths starting from S
        # S is 2*n
        # Neighbors of S are v_out (n+v)
        
        # Create adjacency for the flow graph
        flow_adj = defaultdict(list)
        for (u, v) in used_arcs.keys():
            flow_adj[u].append(v)
            
        # Traverse from S
        # Since it's a flow, we can just follow the unique path
        # But S might have multiple outgoing arcs.
        
        # Get all start vertices
        start_vertices = []
        for (u, v) in used_arcs.keys():
            if u == 2*n:
                # v is n+v_orig
                v_orig = v - n
                start_vertices.append(v_orig)
                
        for sv in start_vertices:
            # Trace path
            current_node = n + sv # Start at sv_out
            path = [sv]
            
            # We are at sv_out. Where do we go?
            # sv_out connects to sv_in (pass) or neighbor_in (edge)
            # Find next node
            next_node = None
            for nb in flow_adj[current_node]:
                next_node = nb
                break # Should be only one outgoing arc in a valid flow decomposition?
                # Actually, flow conservation ensures in=out.
                # At sv_out: Inflow from S (1). Outflow must be 1.
                # So yes, unique next node.
                
            while next_node is not None and next_node != 2*n + 1:
                # Determine the vertex
                if next_node < n:
                    # It's a v_in. Vertex is next_node.
                    path.append(next_node)
                    # Move to v_out
                    current_node = next_node
                    # Next step is from v_in to v_out or T
                    # Check if we go to T
                    if (current_node, 2*n+1) in used_arcs:
                         next_node = 2*n+1
                    else:
                         # Go to v_out
                         next_node = current_node + n
                         current_node = next_node
                         # Now at v_out, find next
                         # We need to find the outgoing arc from v_out
                         # We already handled the transition to v_out.
                         # Now we need to find where v_out goes.
                         # Wait, my loop logic is mixed.
                         
                # Let's re-structure the traversal
                break # Restart with cleaner logic
                
        # Cleaner traversal
        paths = []
        for sv in start_vertices:
            path = [sv]
            current_node = n + sv # Start at sv_out
            
            while True:
                # Find outgoing arc from current_node
                next_node = None
                for nb in flow_adj[current_node]:
                    next_node = nb
                    break
                    
                if next_node is None or next_node == 2*n + 1:
                    break
                    
                if next_node < n:
                    # Arrived at v_in. Vertex is next_node.
                    path.append(next_node)
                    # Move to v_out
                    current_node = next_node + n
                else:
                    # next_node is n+v (v_out). This shouldn't happen directly from v_out?
                    # v_out goes to v_in (pass) or u_in (edge).
                    # So next_node should always be < n (v_in) or T.
                    # If next_node >= n and != T, it's a v_out?
                    # That would mean v_out -> v_out, which is not in our graph.
                    # So this case should not occur.
                    pass
                    
        # Verify and return
        # Check if paths cover all edges and are simple
        # ... (verification logic)
        
        return paths

    def decompose_greedy(edges, deadline):
        # Keep the greedy logic from the original program as fallback
        if not edges:
            return []
            
        degs = get_degrees(edges)
        odd_vertices = [v for v, d in degs.items() if d % 2 != 0]
        K = max(1, len(odd_vertices) // 2)
        
        best_paths = None
        best_count = float('inf')
        
        rng = random.Random(42)
        
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            current_adj = defaultdict(list)
            for u, v in edges:
                current_adj[u].append(v)
                current_adj[v].append(u)
                
            remaining_edges = set(edges)
            paths = []
            
            while remaining_edges:
                active_vertices = set()
                for u, v in remaining_edges:
                    active_vertices.add(u)
                    active_vertices.add(v)
                    
                if not active_vertices:
                    break
                    
                start = rng.choice(list(active_vertices))
                path = [start]
                current = start
                visited = {start}
                
                while True:
                    avail = [nb for nb in current_adj[current] 
                             if tuple(sorted((current, nb))) in remaining_edges 
                             and nb not in visited]
                    
                    if not avail:
                        break
                    
                    avail.sort(key=lambda x: len([e for e in remaining_edges if x in e]), reverse=True)
                    
                    if len(avail) > 1 and rng.random() < 0.2:
                        next_v = rng.choice(avail[:3])
                    else:
                        next_v = avail[0]
                    
                    edge = tuple(sorted((current, next_v)))
                    remaining_edges.remove(edge)
                    
                    current_adj[current].remove(next_v)
                    current_adj[next_v].remove(current)
                    
                    path.append(next_v)
                    visited.add(next_v)
                    current = next_v
                    
                paths.append(path)
                
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = paths
                if best_count == K:
                    break
                    
        if best_paths is None:
            return [list(e) for e in edges]
        return best_paths

    def verify(paths, edges):
        edge_set = set(tuple(sorted(e)) for e in edges)
        used_edges = set()
        for p in paths:
            if len(p) < 2: continue
            if len(set(p)) != len(p): return False
            for i in range(len(p)-1):
                e = tuple(sorted((p[i], p[i+1])))
                if e not in edge_set: return False
                if e in used_edges: return False
                used_edges.add(e)
        if used_edges != edge_set: return False
        return True

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        # Try MILP first
        milp_paths = solve_milp(edges, n, deadline)
        
        if milp_paths and verify(milp_paths, edges):
            results.append(milp_paths)
        else:
            # Fallback to greedy
            greedy_paths = decompose_greedy(edges, deadline)
            results.append(greedy_paths)
            
    return results
