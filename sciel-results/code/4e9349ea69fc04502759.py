import time
import random
from collections import defaultdict, deque
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def decompose_graph_milp(edges, time_limit=40):
        if not edges:
            return []
            
        # Map undirected edges to indices
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        n_edges = len(edges)
        n_nodes = max(max(u, v) for u, v in edges) + 1 if edges else 0
        
        # Variables:
        # y[0..2*n_edges-1]: directed arcs. 
        # For edge i (u, v):
        # arc 2*i: u -> v
        # arc 2*i+1: v -> u
        # w[n_nodes]: endpoint indicators
        
        n_vars = 2 * n_edges + n_nodes
        y_offset = 0
        w_offset = 2 * n_edges
        
        # Cost: minimize sum(w)
        c = np.zeros(n_vars)
        c[w_offset:] = 1.0
        
        # Constraints
        A_eq = []
        b_eq = []
        A_ub = []
        b_ub = []
        
        # 1. y_uv + y_vu = 1
        for i, (u, v) in enumerate(edges):
            row = np.zeros(n_vars)
            row[y_offset + 2*i] = 1.0
            row[y_offset + 2*i + 1] = 1.0
            A_eq.append(row)
            b_eq.append(1.0)
            
        # 2. In-degree <= 1
        # For each node v, sum of y_u_v <= 1
        # We need to know which arcs point to v.
        in_arcs = defaultdict(list)
        out_arcs = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            # u -> v is 2*i
            in_arcs[v].append(2*i)
            out_arcs[u].append(2*i)
            # v -> u is 2*i+1
            in_arcs[u].append(2*i+1)
            out_arcs[v].append(2*i+1)
            
        for v in range(n_nodes):
            # In-degree constraint
            row = np.zeros(n_vars)
            for arc_idx in in_arcs[v]:
                row[y_offset + arc_idx] = 1.0
            A_ub.append(row)
            b_ub.append(1.0)
            
            # Out-degree constraint
            row = np.zeros(n_vars)
            for arc_idx in out_arcs[v]:
                row[y_offset + arc_idx] = 1.0
            A_ub.append(row)
            b_ub.append(1.0)
            
            # w_v >= out_v - in_v  =>  out_v - in_v - w_v <= 0
            row = np.zeros(n_vars)
            for arc_idx in out_arcs[v]:
                row[y_offset + arc_idx] = 1.0
            for arc_idx in in_arcs[v]:
                row[y_offset + arc_idx] = -1.0
            row[w_offset + v] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)
            
            # w_v >= in_v - out_v  =>  in_v - out_v - w_v <= 0
            row = np.zeros(n_vars)
            for arc_idx in in_arcs[v]:
                row[y_offset + arc_idx] = 1.0
            for arc_idx in out_arcs[v]:
                row[y_offset + arc_idx] = -1.0
            row[w_offset + v] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)
            
        if A_eq:
            A_eq = np.array(A_eq)
            b_eq = np.array(b_eq)
        else:
            A_eq = None
            b_eq = None
            
        if A_ub:
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
        else:
            A_ub = None
            b_ub = None
            
        # Integrality
        integrality = np.ones(n_vars)
        
        # Bounds
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        
        try:
            res = milp(c=c, constraints=LinearConstraint(A_ub, -np.inf, b_ub), integrality=integrality, bounds=bounds)
            # Note: LinearConstraint for equality is handled separately? 
            # No, milp takes a list of constraints or a single LinearConstraint.
            # We have both equality and inequality.
            # We can combine them or pass a list.
            # scipy.optimize.milp accepts `constraints` as a LinearConstraint or a list of LinearConstraint.
            
            # Let's re-structure to use list of constraints for clarity or combine.
            # Actually, LinearConstraint can handle both if we split?
            # No, LinearConstraint is for A @ x <= b or A @ x >= b or A @ x = b?
            # LinearConstraint(A, lb, ub) defines lb <= A @ x <= ub.
            # So we can use one LinearConstraint for all inequalities (A_ub, -inf, b_ub)
            # And one for equalities (A_eq, b_eq, b_eq).
            
            cons_list = []
            if A_ub is not None:
                cons_list.append(LinearConstraint(A_ub, -np.inf, b_ub))
            if A_eq is not None:
                cons_list.append(LinearConstraint(A_eq, b_eq, b_eq))
                
            res = milp(c=c, constraints=cons_list, integrality=integrality, bounds=bounds)
            
            if not res.success:
                # Fallback to greedy if MILP fails
                return decompose_graph_greedy(edges)
                
            x = res.x
            y = x[y_offset:y_offset + 2*n_edges]
            
            # Extract paths
            # Build directed adjacency
            dir_adj = defaultdict(list)
            for i, (u, v) in enumerate(edges):
                if y[2*i] > 0.5:
                    dir_adj[u].append(v)
                if y[2*i+1] > 0.5:
                    dir_adj[v].append(u)
                    
            paths = []
            visited_nodes = set()
            
            # Find all starts (out=1, in=0)
            # Or just iterate all nodes and if not visited, find a component.
            # Since it's a set of paths/cycles, we can start from any unvisited node.
            
            for start_node in range(n_nodes):
                if start_node in visited_nodes:
                    continue
                # Check if this node has any incident edge in the solution
                if not dir_adj[start_node]:
                    # Check if it has incoming only?
                    # If in=1, out=0, it's an end. We can start from its predecessor?
                    # Easier: Just find a node with out=1 (start of a path) or any node in a cycle.
                    # If we pick an end node (in=1, out=0), we can't follow it.
                    # So we should only start from nodes with out=1.
                    # If all nodes in a component are cycles, any node has out=1.
                    # If a path, the start has out=1.
                    # So, only start if out-degree > 0.
                    # How to check out-degree?
                    # We can compute it from dir_adj.
                    pass
                
                # Let's refine:
                # Find all nodes with out-degree > 0.
                # If we start from one, we trace the path.
                # If it's a cycle, we will return to start.
                # If it's a path, we will hit a node with out-degree 0.
                
                # But we must ensure we cover all edges.
                # Every edge is in exactly one component.
                # So iterating over all nodes and starting a trace if not visited is safe,
                # provided we handle the direction correctly.
                
                # If we start from an end node (out=0), we can't trace.
                # So we should only initiate trace from nodes with out > 0.
                
                # Let's collect starts.
                starts = [n for n in range(n_nodes) if len(dir_adj[n]) > 0]
                
                # But if we just iterate `for start_node in range(n_nodes)`, we might hit an end node first.
                # So let's do this:
                # 1. Identify all nodes with out > 0.
                # 2. For each such node, if not visited, trace.
                
                # Wait, a node in a cycle has out > 0.
                # A start node of a path has out > 0.
                # An end node has out = 0.
                # An internal node has out > 0.
                # So if we start from an internal node, we trace forward, hit the end, and stop.
                # We miss the part before the internal node.
                # So we MUST start from a node with in=0 (start of path) or any node in a cycle.
                
                # How to distinguish?
                # If we start from a node and eventually return to it, it's a cycle.
                # If we start from a node and hit a node with out=0, it's a path.
                # But if we start from an internal node of a path, we only get the suffix.
                
                # Correct approach:
                # Find all nodes with in=0. These are guaranteed starts.
                # Find all cycles.
                
                # Simpler:
                # For each node, if not visited:
                #   If out > 0:
                #     Trace forward.
                #     If we return to start: Cycle. Break it.
                #     If we hit dead end: Path.
                #     BUT, if we started from an internal node, we only got the suffix.
                #     The prefix is still unvisited.
                #     When we get to the prefix nodes later, they will be unvisited.
                #     But they have out > 0 (except the very start).
                #     So we will trace from the start of the path (which has in=0, out=1).
                #     So it will work if we ensure we visit ALL nodes.
                
                # Let's just iterate all nodes.
                # If node is visited, skip.
                # If node has out > 0:
                #   Trace forward until dead end or cycle.
                #   Mark all nodes in this trace as visited.
                #   Add path to list.
                # If node has out == 0:
                #   It must be an end of a path.
                #   It should have been visited when we traced from the start.
                #   If not, something is wrong (isolated? no, edges covered).
                #   So we can skip.
                
                if len(dir_adj[start_node]) == 0:
                    continue
                    
                path = [start_node]
                visited_nodes.add(start_node)
                current = start_node
                is_cycle = False
                
                while True:
                    nexts = dir_adj[current]
                    if not nexts:
                        break
                    next_node = nexts[0] # out <= 1, so only one
                    if next_node in visited_nodes:
                        is_cycle = True
                        break
                    path.append(next_node)
                    visited_nodes.add(next_node)
                    current = next_node
                    
                if is_cycle:
                    # We have a cycle. The last node points to the first node.
                    # We need to break it.
                    # Remove the edge from last to first.
                    # The path is path[0] ... path[-1].
                    # This is a valid simple path.
                    pass
                else:
                    # Simple path.
                    pass
                
                if len(path) > 1:
                    paths.append(path)
                    
            return paths
            
        except Exception as e:
            # Fallback
            return decompose_graph_greedy(edges)
            
    def decompose_graph_greedy(edges, time_limit=40):
        # Same as before but safer
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edges)
        paths = []
        
        def find_long_path():
            if not remaining_edges:
                return None
            
            rem_deg = defaultdict(int)
            for u, v in remaining_edges:
                rem_deg[u] += 1
                rem_deg[v] += 1
                
            if not rem_deg:
                return None
                
            start = max(rem_deg.keys(), key=lambda x: (rem_deg[x], -x))
            
            visited = {start}
            path = [start]
            current = start
            
            while True:
                neighbors = [n for n in adj[current] if (min(current, n), max(current, n)) in remaining_edges]
                if not neighbors:
                    break
                
                # Choose neighbor with min rem_deg
                next_node = min(neighbors, key=lambda n: rem_deg.get(n, 0))
                
                e = (min(current, next_node), max(current, next_node))
                remaining_edges.remove(e)
                rem_deg[current] -= 1
                rem_deg[next_node] -= 1
                
                if next_node in visited:
                    # Restore
                    remaining_edges.add(e)
                    rem_deg[current] += 1
                    rem_deg[next_node] += 1
                    neighbors.remove(next_node)
                    if not neighbors:
                        break
                    next_node = min(neighbors, key=lambda n: rem_deg.get(n, 0))
                    e = (min(current, next_node), max(current, next_node))
                    remaining_edges.remove(e)
                    rem_deg[current] -= 1
                    rem_deg[next_node] -= 1
                    if next_node in visited:
                        # Still in visited? Should not happen if we removed it from neighbors
                        # But if all neighbors are in visited, we broke.
                        # If we picked a new one, it shouldn't be in visited.
                        # Unless visited contains all neighbors?
                        # If so, we break.
                        pass
                
                path.append(next_node)
                visited.add(next_node)
                current = next_node
                
            return path if len(path) > 1 else None
            
        while remaining_edges:
            if time.time() - time_limit > time_limit:
                for u, v in remaining_edges:
                    paths.append([u, v])
                break
                
            path = find_long_path()
            if path is None:
                for u, v in remaining_edges:
                    paths.append([u, v])
                break
            paths.append(path)
            
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(*spec)
        paths = decompose_graph_milp(edges, time_limit=40)
        results.append(paths)
        
    return results
