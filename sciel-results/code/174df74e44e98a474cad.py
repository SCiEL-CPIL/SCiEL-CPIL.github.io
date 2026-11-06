"""Rewritten program using MILP for minimum path edge cover."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from itertools import combinations

def construct():
    def instance(n, p, seed):
        import random
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_milp(n, edge_list, seed):
        import random
        if not edge_list:
            return []
        
        m = len(edge_list)
        
        # Build adjacency for quick lookups
        adj = {v: set() for v in range(n)}
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        
        # Variables:
        # For each edge e = (u, v), we have 2 binary vars: y_{uv}, y_{vu}
        # Total arc vars: 2m
        # For each vertex v, we have 2 continuous vars: p_v (source), q_v (sink)
        # Total vars: 2m + 2n
        
        # Variable indices:
        # Arc vars: 0 to 2m-1
        #   For edge i = (u, v): arc u->v is 2*i, arc v->u is 2*i+1
        # Vertex vars: 2m to 2m+2n-1
        #   For vertex v: p_v is 2m + v, q_v is 2m + n + v
        
        # Objective: minimize sum of p_v
        c = np.zeros(2 * m + 2 * n)
        for v in range(n):
            c[2 * m + v] = 1.0  # p_v
        
        # Constraints:
        # 1. For each edge (u, v): y_{uv} + y_{vu} = 1
        # 2. For each vertex v: sum_{u in N(v)} y_{uv} - sum_{u in N(v)} y_{vu} = p_v - q_v
        #    i.e., out_degree - in_degree = p_v - q_v
        
        # Build constraint matrix
        A = []
        lb = []
        ub = []
        
        # Constraint 1: for each edge
        for i, (u, v) in enumerate(edge_list):
            row = np.zeros(2 * m + 2 * n)
            row[2 * i] = 1.0    # y_{uv}
            row[2 * i + 1] = 1.0  # y_{vu}
            A.append(row)
            lb.append(1.0)
            ub.append(1.0)
        
        # Constraint 2: for each vertex
        # out(v) - in(v) = p_v - q_v
        # sum_{u in N(v)} y_{vu} - sum_{u in N(v)} y_{uv} = p_v - q_v
        # Wait, let's be careful with orientation.
        # If edge is (u, v), then arc u->v is 2*i, arc v->u is 2*i+1.
        # For vertex v, incoming arcs are those where v is the target.
        # Outgoing arcs are those where v is the source.
        
        # Let's precompute for each vertex, the list of (arc_index, sign)
        # where sign is +1 for outgoing, -1 for incoming.
        
        vertex_arc_coeffs = {v: [] for v in range(n)}
        
        for i, (u, v) in enumerate(edge_list):
            # Arc u->v: index 2*i, outgoing from u, incoming to v
            vertex_arc_coeffs[u].append((2 * i, +1))
            vertex_arc_coeffs[v].append((2 * i, -1))
            # Arc v->u: index 2*i+1, outgoing from v, incoming to u
            vertex_arc_coeffs[v].append((2 * i + 1, +1))
            vertex_arc_coeffs[u].append((2 * i + 1, -1))
        
        for v in range(n):
            row = np.zeros(2 * m + 2 * n)
            for arc_idx, sign in vertex_arc_coeffs[v]:
                row[arc_idx] = sign
            # p_v - q_v on the right side, moved to left: ... - p_v + q_v = 0
            row[2 * m + v] = -1.0       # -p_v
            row[2 * m + n + v] = 1.0     # +q_v
            A.append(row)
            lb.append(0.0)
            ub.append(0.0)
        
        A = np.array(A)
        lb = np.array(lb)
        ub = np.array(ub)
        
        # Bounds: all vars in [0, 1]
        bounds = Bounds(lb=np.zeros(2 * m + 2 * n), ub=np.ones(2 * m + 2 * n))
        
        # Integrality: arc vars are binary, vertex vars are continuous (0 to 1 effectively, but let's say 0 to n)
        integrality = np.zeros(2 * m + 2 * n)
        for i in range(2 * m):
            integrality[i] = 1  # binary
        # Vertex vars are continuous, so integrality = 0
        
        # Solve
        res = milp(c=c, constraints=LinearConstraint(A, lb, ub), bounds=bounds, integrality=integrality)
        
        if not res.success:
            # Fallback to greedy
            return solve_greedy(n, edge_list, seed)
        
        # Extract solution
        x = res.x
        
        # Reconstruct directed edges
        directed_edges = []
        for i, (u, v) in enumerate(edge_list):
            y_uv = x[2 * i]
            y_vu = x[2 * i + 1]
            if y_uv > 0.5:
                directed_edges.append((u, v))
            elif y_vu > 0.5:
                directed_edges.append((v, u))
        
        # Build directed adjacency
        out_adj = {v: [] for v in range(n)}
        in_adj = {v: [] for v in range(n)}
        for u, v in directed_edges:
            out_adj[u].append(v)
            in_adj[v].append(u)
        
        # Find sources (out > in) and sinks (in > out)
        sources = []
        sinks = []
        for v in range(n):
            out_deg = len(out_adj[v])
            in_deg = len(in_adj[v])
            if out_deg > in_deg:
                sources.append(v)
            elif in_deg > out_deg:
                sinks.append(v)
        
        # Trace paths from sources to sinks
        # Also handle cycles
        
        paths = []
        visited_arcs = set()
        
        # First, trace from sources
        for s in sources:
            if not out_adj[s]:
                continue
            path = [s]
            cur = s
            while True:
                # Find an outgoing arc from cur that hasn't been visited
                next_node = None
                for nxt in out_adj[cur]:
                    if (cur, nxt) not in visited_arcs:
                        next_node = nxt
                        break
                if next_node is None:
                    break
                visited_arcs.add((cur, next_node))
                path.append(next_node)
                cur = next_node
                if not out_adj[cur]:
                    break
            if len(path) > 1:
                paths.append(path)
        
        # Then, find cycles among remaining arcs
        # Find all unvisited arcs
        remaining_arcs = [(u, v) for u, v in directed_edges if (u, v) not in visited_arcs]
        
        # Build a graph from remaining arcs
        rem_out = {v: [] for v in range(n)}
        for u, v in remaining_arcs:
            rem_out[u].append(v)
        
        # Find cycles
        visited_nodes = set()
        for start in range(n):
            if start in visited_nodes or not rem_out[start]:
                continue
            # Try to find a cycle starting from start
            stack = [(start, [start])]
            found_cycle = None
            while stack:
                node, path_so_far = stack.pop()
                if node in visited_nodes:
                    continue
                visited_nodes.add(node)
                for nxt in rem_out[node]:
                    if nxt == start and len(path_so_far) > 1:
                        # Found a cycle
                        found_cycle = path_so_far
                        break
                    elif nxt not in visited_nodes and (node, nxt) in set(remaining_arcs):
                        stack.append((nxt, path_so_far + [nxt]))
                if found_cycle:
                    break
            if found_cycle:
                # Mark arcs as visited
                for i in range(len(found_cycle)):
                    u = found_cycle[i]
                    v = found_cycle[(i + 1) % len(found_cycle)]
                    visited_arcs.add((u, v))
                paths.append(found_cycle)
        
        # If there are still unvisited arcs, something went wrong
        # But for now, return what we have
        return paths
    
    def solve_greedy(n, edge_list, seed):
        import random
        rng = random.Random(seed)
        if not edge_list:
            return []
        
        # Initialize paths: each edge is a path
        paths = []
        endpoint_map = {v: set() for v in range(n)}
        
        for u, v in edge_list:
            pid = len(paths)
            paths.append([u, v])
            endpoint_map[u].add(pid)
            endpoint_map[v].add(pid)
        
        active_paths = set(range(len(paths)))
        
        def is_simple(path):
            return len(set(path)) == len(path)
        
        max_iter = 10000
        iter_count = 0
        
        while iter_count < max_iter:
            iter_count += 1
            merged_any = False
            
            candidates = [v for v in range(n) if len(endpoint_map[v]) >= 2]
            
            if not candidates:
                break
                
            rng.shuffle(candidates)
            
            for v in candidates:
                if len(endpoint_map[v]) < 2:
                    continue
                    
                pids = list(endpoint_map[v])
                rng.shuffle(pids)
                
                merged = False
                for i in range(len(pids)):
                    for j in range(i + 1, len(pids)):
                        p1_id = pids[i]
                        p2_id = pids[j]
                        
                        if p1_id not in active_paths or p2_id not in active_paths:
                            continue
                            
                        p1 = paths[p1_id]
                        p2 = paths[p2_id]
                        
                        valid_merge = False
                        new_path = None
                        
                        if p1[-1] == v and p2[0] == v:
                            cand = p1 + p2[1:]
                            if is_simple(cand):
                                new_path = cand
                                valid_merge = True
                        if not valid_merge and p1[-1] == v and p2[-1] == v:
                            p2_rev = list(reversed(p2))
                            cand = p1 + p2_rev[1:]
                            if is_simple(cand):
                                new_path = cand
                                valid_merge = True
                        if not valid_merge and p1[0] == v and p2[0] == v:
                            p1_rev = list(reversed(p1))
                            cand = p1_rev + p2[1:]
                            if is_simple(cand):
                                new_path = cand
                                valid_merge = True
                        if not valid_merge and p1[0] == v and p2[-1] == v:
                            p1_rev = list(reversed(p1))
                            cand = p1_rev + p2[1:]
                            if is_simple(cand):
                                new_path = cand
                                valid_merge = True
                            
                        if valid_merge:
                            active_paths.remove(p1_id)
                            active_paths.remove(p2_id)
                            
                            if p1[0] == v:
                                end1 = p1[-1]
                            else:
                                end1 = p1[0]
                                
                            if p2[0] == v:
                                end2 = p2[-1]
                            else:
                                end2 = p2[0]
                                
                            endpoint_map[v].remove(p1_id)
                            endpoint_map[v].remove(p2_id)
                            endpoint_map[end1].remove(p1_id)
                            endpoint_map[end2].remove(p2_id)
                            
                            new_id = len(paths)
                            paths.append(new_path)
                            active_paths.add(new_id)
                            
                            endpoint_map[end1].add(new_id)
                            endpoint_map[end2].add(new_id)
                            
                            merged_any = True
                            merged = True
                            break
                    if merged:
                        break
                if merged:
                    break
            
            if not merged_any:
                break
        
        final_paths = [paths[i] for i in active_paths]
        return final_paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        # Try MILP first
        try:
            paths = solve_milp(n, edges, seed)
            if paths is None or len(paths) == 0 and len(edges) > 0:
                paths = solve_greedy(n, edges, seed)
        except Exception:
            paths = solve_greedy(n, edges, seed)
        
        # Also try greedy with multiple seeds and take the best
        best_paths = paths
        best_count = len(paths)
        
        for run_seed in range(10):
            g_paths = solve_greedy(n, edges, seed + 100 + run_seed)
            if len(g_paths) < best_count:
                best_count = len(g_paths)
                best_paths = g_paths
        
        results.append(best_paths)
    
    return results
# EVOLVE-BLOCK-END
