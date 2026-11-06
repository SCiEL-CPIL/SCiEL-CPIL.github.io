import time
import random
from collections import defaultdict
import numpy as np

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []
    start_time = time.time()
    # Leave 10 seconds buffer for 150s limit
    deadline = start_time + 140.0
    
    for n, p, seed in specs:
        edges = instance(n, p, seed)
        if not edges:
            results.append([])
            continue
            
        # Build adjacency list and edge index
        adj = defaultdict(set)
        edge_index = {}
        for i, (u, v) in enumerate(edges):
            adj[u].add(v)
            adj[v].add(u)
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i
            
        m = len(edges)
        
        # --- Strategy: Generate candidate cycles and solve ILP ---
        
        candidate_cycles = []
        
        # 1. Find all 3-cycles
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    for w in adj[v]:
                        if w > u and w in adj[u]:
                            # Cycle u-v-w-u
                            # Ensure sorted representation for consistency?
                            # The cycle is [u, v, w]
                            candidate_cycles.append([u, v, w])
        
        # 2. Find all 4-cycles
        # u-v-w-x-u
        # To avoid duplicates, we can enforce u < v < w < x ? No, order matters for the cycle structure but not for the set of edges.
        # We just need unique sets of edges.
        seen_4_cycles = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    for w in adj[v]:
                        if w > u and w != v:
                            for x in adj[w]:
                                if x > u and x != v and x != w and x in adj[u]:
                                    # Cycle u-v-w-x
                                    # Check if we already have this edge set
                                    # Edges: (u,v), (v,w), (w,x), (x,u)
                                    key = frozenset([
                                        tuple(sorted((u, v))),
                                        tuple(sorted((v, w))),
                                        tuple(sorted((w, x))),
                                        tuple(sorted((x, u)))
                                    ])
                                    if key not in seen_4_cycles:
                                        seen_4_cycles.add(key)
                                        candidate_cycles.append([u, v, w, x])
        
        # 3. Generate random longer cycles (length 5 to 12)
        # Use randomized DFS to find cycles
        rng_local = random.Random(seed + 100)
        num_random_cycles = 500
        for _ in range(num_random_cycles):
            if time.time() > deadline - 10:
                break
            # Pick a random start node
            start = rng_local.choice(list(range(n)))
            path = [start]
            visited = {start}
            current = start
            
            # Try to extend path
            max_len = 12
            success = False
            for _depth in range(max_len - 1):
                neighbors = list(adj[current])
                # Filter neighbors that are not visited, or is start if depth >= 2
                next_neighbors = []
                for nb in neighbors:
                    if nb not in visited:
                        next_neighbors.append(nb)
                    elif nb == start and len(path) >= 3:
                        # Found a cycle
                        success = True
                        break
                
                if success:
                    break
                
                if not next_neighbors:
                    break
                
                # Choose next node randomly
                next_node = rng_local.choice(next_neighbors)
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            
            if success and len(path) >= 3:
                # Add to candidates
                # Check if this exact cycle (as set of edges) is already in candidates?
                # It might be a 3 or 4 cycle already found.
                # For simplicity, we just add it. ILP will handle duplicates (or we can dedupe).
                candidate_cycles.append(path)

        # Deduplicate cycles based on edge sets
        unique_cycles = []
        seen_edge_sets = set()
        for cyc in candidate_cycles:
            # Validate cycle
            valid = True
            edge_set = set()
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i + 1) % len(cyc)]
                if u == v:
                    valid = False
                    break
                e_key = (min(u, v), max(u, v))
                if e_key not in edge_index:
                    valid = False
                    break
                if e_key in edge_set:
                    valid = False
                    break
                edge_set.add(e_key)
            
            if valid and len(edge_set) == len(cyc):
                if edge_set not in seen_edge_sets:
                    seen_edge_sets.add(edge_set)
                    unique_cycles.append(cyc)
        
        # If we found cycles, try ILP
        if unique_cycles:
            try:
                from scipy.optimize import milp, LinearConstraint, Bounds
                import numpy as np
                
                num_cycles = len(unique_cycles)
                num_edges = m
                
                # Objective: Maximize sum(|c| - 1) * x_i
                # Minimize -sum(|c| - 1) * x_i
                c_obj = -np.array([len(c) - 1 for c in unique_cycles], dtype=float)
                
                # Constraints: For each edge e, sum(x_i for i where e in C_i) <= 1
                # A matrix of size m x num_cycles
                A = np.zeros((num_edges, num_cycles), dtype=float)
                
                for i, cyc in enumerate(unique_cycles):
                    for j in range(len(cyc)):
                        u = cyc[j]
                        v = cyc[(j + 1) % len(cyc)]
                        e_key = (min(u, v), max(u, v))
                        e_idx = edge_index[e_key]
                        A[e_idx, i] = 1.0
                
                constraints = LinearConstraint(A, ub=np.ones(num_edges))
                integrality = np.ones(num_cycles)
                bounds = Bounds(lb=0, ub=1)
                
                res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
                
                if res.success:
                    selected_cycles = []
                    for i in range(num_cycles):
                        if res.x[i] > 0.5:
                            selected_cycles.append(unique_cycles[i])
                    
                    # Construct pieces
                    pieces = []
                    used_edges = set()
                    for cyc in selected_cycles:
                        pieces.append(cyc)
                        for j in range(len(cyc)):
                            u = cyc[j]
                            v = cyc[(j + 1) % len(cyc)]
                            used_edges.add((min(u, v), max(u, v)))
                    
                    # Add remaining edges
                    for e in edges:
                        if e not in used_edges:
                            pieces.append(list(e))
                    
                    results.append(pieces)
                    continue
            except Exception:
                pass # Fallback to greedy if ILP fails
        
        # --- Fallback: Greedy Longest Cycle Removal ---
        remaining_edges = set(edges)
        pieces = []
        
        # Helper to check time
        def time_up():
            return time.time() > deadline - 5.0
        
        def find_greedy_cycle():
            if not remaining_edges:
                return None
            
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
            
            if not active_vertices:
                return None
            
            best_cycle = None
            max_attempts = 200
            
            for _ in range(max_attempts):
                if time_up():
                    break
                start = random.choice(list(active_vertices))
                path = [start]
                visited = {start}
                current = start
                
                # Greedy extension
                while True:
                    neighbors = [v for v in adj[current] if (min(current, v), max(current, v)) in remaining_edges]
                    if not neighbors:
                        break
                    
                    # Prefer unvisited
                    unvisited = [v for v in neighbors if v not in visited]
                    if unvisited:
                        # Pick neighbor with highest degree in remaining graph
                        # Or random for diversity
                        next_node = random.choice(unvisited)
                        path.append(next_node)
                        visited.add(next_node)
                        current = next_node
                    else:
                        # Try to close
                        if (min(current, start), max(current, start)) in remaining_edges:
                            if len(path) >= 3:
                                if best_cycle is None or len(path) > len(best_cycle):
                                    best_cycle = path[:]
                            break
                        else:
                            break
                
                if best_cycle and len(best_cycle) >= 10: # Heuristic: stop early if we found a decent one
                    break
            
            return best_cycle

        while remaining_edges and not time_up():
            cycle = find_greedy_cycle()
            if cycle:
                # Validate and add
                valid = True
                cycle_edges = set()
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    if e not in remaining_edges:
                        valid = False
                        break
                    cycle_edges.add(e)
                
                if valid:
                    pieces.append(cycle)
                    remaining_edges -= cycle_edges
                else:
                    # If invalid, break to avoid infinite loop, treat rest as edges
                    for e in remaining_edges:
                        pieces.append(list(e))
                    remaining_edges = set()
            else:
                # No cycle found
                for e in remaining_edges:
                    pieces.append(list(e))
                remaining_edges = set()
        
        results.append(pieces)
        
    return results

construct()
