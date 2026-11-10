"""
Strategy: Minimum-Flow Edge-Disjoint Path Cover
Instead of Eulerian augmentation, we model the edge-disjoint path cover as a
flow problem. We want to cover all edges with the minimum number of paths.
This is equivalent to finding a flow of value E (number of edges) in a network
where each edge has capacity 1, and we minimize the "cost" associated with
starting a path.

A path cover of k paths corresponds to a flow where:
- Each original edge (u, v) is represented by a node u -> v with capacity 1, cost 0.
- To allow paths to start and end, we add a super source S and super sink T.
- However, the standard reduction for "minimum number of paths covering all edges"
  is slightly different. It is often mapped to:
  Minimize the number of paths = (E - M), where M is the maximum number of 
  "connections" between path segments? No.
  
  Actually, a simpler approach for sparse graphs (n <= 30, E ~ 50-100) is 
  Integer Linear Programming or Max Flow with costs.
  
  Let's use the standard reduction:
  To cover edges with minimum number of paths, we can view it as:
  Each path has a start and an end.
  Let x_{uv} be 1 if edge (u,v) is used in a path. We must use all edges.
  For each vertex v, let in_v be the number of paths ending at v, out_v be the number of paths starting at v.
  Flow conservation: sum(in_edges) + out_v = sum(out_edges) + in_v.
  Actually, for a path cover:
  For each vertex v, let d_v be its degree.
  Let s_v be the number of paths starting at v.
  Let e_v be the number of paths ending at v.
  Then s_v + e_v = d_v - 2 * (number of paths passing through v).
  This is complicated because "passing through" isn't a flow variable directly.
  
  Alternative:
  We can use a Min-Cost Max-Flow where we send 1 unit of flow for each edge.
  This is tricky because edges are the "items" to be covered, not the flow itself.
  
  Better approach for this specific size:
  Since n is small (<=30) and E is small (~50-100), we can use a greedy algorithm 
  guided by the "odd degree" matching but fixing the simplicity issue by 
  **checking simplicity and retrying with different matchings or using local repair**.
  
  However, the prompt asks for a MATERIALLY DIFFERENT strategy.
  The current program uses Eulerian augmentation.
  Inspiration 1 suggests "Minimum-flow (edge-disjoint simple path cover) formulation".
  
  Let's implement a Min-Cost Flow that actually solves for the minimum path cover.
  
  Problem: Minimum Edge-Disjoint Path Cover.
  This is NP-hard in general, but for small graphs, we can use a specific flow formulation.
  
  Actually, there is a known polynomial time algorithm for Minimum Path Cover in DAGs, 
  but our graph is undirected.
  
  For undirected graphs, the minimum number of edge-disjoint paths covering all edges 
  is at least max(1, odd_vertices / 2).
  Gallai's conjecture says it is at most n/2.
  
  Let's try a different constructive approach:
  **Greedy with Backtracking / Local Search on Path Merging**.
  
  Start with a valid cover (e.g., from Eulerian augmentation if it works, or greedy).
  Then, try to **merge** two paths into one if possible, or **re-route** edges to reduce the count.
  
  Wait, the current program's score is 5.16. The theoretical lower bound is sum(odd_v/2).
  For n=20, p=0.3, E~30. Odd vertices ~ 2 * (1 - (1-p)^{n-1} ... ) roughly even.
  If we can achieve the lower bound, the score would be much lower (better).
  Lower bound for 3 graphs:
  G1 (20, 0.3): E ~ 20*19/2 * 0.3 + 19 ~ 57 + 19 = 76? No.
  instance_graph generates a random graph G(n, p) plus a spanning tree path.
  n=20, p=0.3. Expected edges = 19 * 0.3 + 19 = 5.7 + 19 = 24.7.
  Odd vertices in random graph: roughly half. So ~10 odd vertices.
  Lower bound: 10/2 = 5 paths.
  Current score 5.16 implies total paths ~ 5.16 * 38? No, score is sum of paths.
  "score is the total number of paths".
  If score is 5.16, that's impossible if the lower bound is 5+6+7=18.
  
  Re-reading the prompt: "score is seed/value". 
  "search score is seed/value, so 1.0 equals the seed program".
  "The evaluator measures the raw value (lower is better)."
  "score is the total number of paths".
  
  If the current program has a score of 5.1636, and the raw value is the total number of paths,
  then the total number of paths is 5.1636? No, paths are integers.
  
  Ah, "score is seed/value". 
  If seed program value is V_seed, score = V_seed / V_child?
  Or score = V_child / V_seed?
  "1.0 equals the seed program and above 1.0 is an improvement".
  So Score = V_seed / V_child.
  If Score = 5.16, then V_child = V_seed / 5.16.
  This implies the child is 5x better than the seed.
  
  What is the seed program? "Initial program".
  If the seed program was just listing edges (value = E_total), and the current program achieves a much lower path count, the score would be high.
  
  Let's assume the "raw value" is the total number of paths.
  For the 3 graphs, the total paths might be around 15-20.
  If the seed (greedy) produced ~100 paths, and we produce ~20, score = 5.
  
  So the goal is to **minimize the total number of paths**.
  
  Strategy: **Minimum Cost Perfect Matching on Odd Vertices with Simple Path Verification and Repair**.
  
  The current program tries this but falls back to greedy if paths are not simple.
  The "inspiration" says: "Vertex-disjoint augmentation path enforcement ... directly targets the simplicity constraint ... potentially achieving the theoretical |odd|/2 path minimum".
  
  My new strategy:
  1. Compute odd vertices.
  2. Find min-cost perfect matching on odd vertices (using Hungarian algorithm on distance matrix).
  3. Construct the augmented graph.
  4. Find Eulerian circuit.
  5. Extract paths.
  6. **If any path is not simple, do NOT just fall back to greedy.**
     Instead, **local search**: 
     - Identify the conflict (vertex repeated in a path).
     - Try to swap edges in the matching or re-route the virtual path to avoid the conflict.
     - Or, use a **different matching** (2nd best, 3rd best) until a simple path cover is found.
     - Or, use **Simulated Annealing** on the set of paths to merge/split/re-route to reduce count while maintaining simplicity.
  
  Given the time limit (120s), and small graphs, we can afford to try multiple matchings.
  The number of odd vertices k is small (<= n).
  The number of perfect matchings can be large, but we can sample random matchings near the optimal one.
  
  Actually, a simpler robust approach for small graphs:
  **Integer Linear Programming (ILP) via scipy.optimize.milp?**
  SciPy has `milp` (mixed integer linear programming) in recent versions.
  If `scipy.optimize.milp` is available, we can formulate:
  Minimize sum of path_starts
  Subject to: flow conservation, edge usage = 1, integrality.
  
  Let's check if `scipy.optimize.milp` is available. It was added in SciPy 1.7.
  The environment likely has a recent SciPy.
  
  Formulation for Min Path Cover (Undirected):
  Variables:
  x_{uv} in {0, 1} for each directed edge (u, v) where (u, v) is an original edge.
  Actually, since edges are undirected, we can split each edge (u, v) into (u, v) and (v, u).
  We must choose exactly one direction for each edge? No, that's for Eulerian.
  
  For path cover:
  Let y_{uv} be 1 if edge (u, v) is traversed from u to v in a path.
  Let y_{vu} be 1 if edge (u, v) is traversed from v to u.
  Constraint: y_{uv} + y_{vu} = 1 for all edges {u, v}.
  
  Flow conservation at each node v:
  sum_{u} y_{uv} - sum_{u} y_{vu} = s_v - e_v
  where s_v is 1 if a path starts at v, 0 otherwise.
  e_v is 1 if a path ends at v, 0 otherwise.
  
  We want to minimize sum_v s_v (number of paths).
  Note: s_v and e_v are binary variables.
  Also, s_v + e_v <= 1? No, a vertex can be start of one and end of another?
  In a path cover, a vertex can be the start of one path and the end of another path?
  Yes, if the paths are disjoint in edges but not vertices.
  However, if a vertex is the start of one path and end of another, it's effectively a "through" vertex for two paths? No, it's two separate paths.
  
  Wait, if a vertex is the end of path A and start of path B, we can merge them into a single path A+B.
  So in a **minimum** path cover, no vertex is both the end of one path and the start of another.
  Thus, for each vertex v, s_v + e_v <= 1.
  And s_v, e_v are binary.
  
  Objective: Minimize sum_v s_v.
  
  This is a valid ILP formulation!
  
  Let's implement this using `scipy.optimize.milp`.
  
  Steps:
  1. Generate graph.
  2. Create directed arcs for each undirected edge.
  3. Define variables y_{uv}, y_{vu} for each edge.
  4. Define variables s_v, e_v for each vertex.
  5. Constraints:
     - y_{uv} + y_{vu} = 1 for all edges.
     - Flow conservation: sum_in(v) - sum_out(v) = e_v - s_v.
       (Note: if s_v=1, net outflow is 1. If e_v=1, net inflow is 1. If both 0, net 0).
     - s_v + e_v <= 1.
     - y, s, e are binary (or integer 0-1).
  6. Solve.
  7. Reconstruct paths from the flow.
  
  This should give the **exact minimum** number of paths for these small graphs.
  The number of variables is 2*E + 2*V.
  For n=30, E~50. Vars ~ 100 + 60 = 160.
  Constraints ~ E + V + V = 130.
  This is very small for an ILP solver. It should solve in milliseconds.
  
  Let's write the code.

"""

import time
from collections import defaultdict, deque
from random import Random
import numpy as np
import scipy.optimize as opt

def construct():
    _start = time.time()
    _DEADLINE = 140.0

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_min_path_cover_ilp(n, edges):
        """
        Solve the minimum edge-disjoint path cover using ILP.
        Returns list of paths (vertex lists).
        """
        if not edges:
            return []
        
        # Map vertices to 0..n-1
        # n is given
        
        # Create directed arcs
        # For each undirected edge (u, v), we have variables y_uv and y_vu
        # We need to map (u, v) to a variable index.
        
        arc_vars = {} # (u, v) -> var index
        var_idx = 0
        for (u, v) in edges:
            arc_vars[(u, v)] = var_idx
            var_idx += 1
            arc_vars[(v, u)] = var_idx
            var_idx += 1
            
        # s_v and e_v variables
        s_vars = {}
        e_vars = {}
        for v in range(n):
            s_vars[v] = var_idx
            var_idx += 1
            e_vars[v] = var_idx
            var_idx += 1
            
        total_vars = var_idx
        
        # Objective: Minimize sum(s_v)
        c = np.zeros(total_vars)
        for v in range(n):
            c[s_vars[v]] = 1.0
            
        # Constraints
        # 1. y_uv + y_vu = 1 for each edge
        # 2. Flow conservation: sum(y_uv) - sum(y_vu) = e_v - s_v
        #    => sum(y_uv) - sum(y_vu) - e_v + s_v = 0
        # 3. s_v + e_v <= 1
        
        A_rows = []
        b_vals = []
        
        # Constraint type: equality or inequality
        # scipy.optimize.milp takes c, constraints (linear), integrality, bounds
        
        # Build linear constraints for milp
        # milp expects:
        # c: objective
        # constraints: LinearConstraint(A, lb, ub)
        # integrality: array of 0 (continuous) or 1 (integer)
        # bounds: list of (lb, ub)
        
        # All variables are binary (0 or 1)
        integrality = np.ones(total_vars)
        bounds = [(0, 1)] * total_vars
        
        # A matrix is sparse, but total_vars is small (~200), so dense is fine.
        A = np.zeros((2 * len(edges) + n + n, total_vars)) # Rough estimate of rows
        # Let's build list of rows first
        rows = []
        lbs = []
        ubs = []
        
        # 1. Edge constraints: y_uv + y_vu = 1
        for (u, v) in edges:
            row = np.zeros(total_vars)
            row[arc_vars[(u, v)]] = 1.0
            row[arc_vars[(v, u)]] = 1.0
            rows.append(row)
            lbs.append(1.0)
            ubs.append(1.0)
            
        # 2. Flow conservation: sum_in - sum_out - e_v + s_v = 0
        # sum(y_wv) - sum(y_vw) - e_v + s_v = 0
        for v in range(n):
            row = np.zeros(total_vars)
            # Inflow: y_wv where (w, v) is an edge
            for (u, w) in edges:
                if w == v:
                    row[arc_vars[(u, v)]] = 1.0 # y_uv where v is second? 
                    # Wait, arc_vars keys are (start, end).
                    # So inflow to v is from u to v. Key (u, v).
                if u == v:
                    row[arc_vars[(v, w)]] = -1.0 # Outflow from v to w. Key (v, w).
            
            # Correction:
            # Inflow to v: edges (u, v). Variable y_{uv}.
            # Outflow from v: edges (v, w). Variable y_{vw}.
            # Equation: sum_{u:(u,v) in E} y_{uv} - sum_{w:(v,w) in E} y_{vw} - e_v + s_v = 0
            
            row = np.zeros(total_vars)
            for (u, w) in edges:
                if w == v and u != v:
                    row[arc_vars[(u, v)]] = 1.0
                if u == v and w != v:
                    row[arc_vars[(v, w)]] = -1.0
            
            row[e_vars[v]] = -1.0
            row[s_vars[v]] = 1.0
            
            rows.append(row)
            lbs.append(0.0)
            ubs.append(0.0)
            
        # 3. s_v + e_v <= 1
        for v in range(n):
            row = np.zeros(total_vars)
            row[s_vars[v]] = 1.0
            row[e_vars[v]] = 1.0
            rows.append(row)
            lbs.append(0.0)
            ubs.append(1.0)
            
        A = np.array(rows)
        lb = np.array(lbs)
        ub = np.array(ubs)
        
        # Solve
        try:
            res = opt.milp(c, constraints=opt.LinearConstraint(A, lb, ub), 
                           integrality=integrality, bounds=bounds)
            if not res.success:
                return None
            
            # Extract solution
            x = res.x
            
            # Reconstruct paths
            # Get directed edges used
            directed_edges = []
            for (u, v) in edges:
                if x[arc_vars[(u, v)]] > 0.5:
                    directed_edges.append((u, v))
                elif x[arc_vars[(v, u)]] > 0.5:
                    directed_edges.append((v, u))
            
            # Build adjacency for directed graph
            adj = defaultdict(list)
            for u, v in directed_edges:
                adj[u].append(v)
                
            # Find paths
            # Starts are vertices with s_v > 0.5
            starts = [v for v in range(n) if x[s_vars[v]] > 0.5]
            
            paths = []
            used_edges = set()
            
            for start in starts:
                path = [start]
                current = start
                while current in adj:
                    # Find next node
                    nexts = adj[current]
                    if not nexts:
                        break
                    # Since it's a path cover, out-degree of current should be 1 
                    # unless it's a start? 
                    # Actually, in the flow solution, each node has out-degree 0 or 1 
                    # (except starts which have out-degree 1, ends 0, intermediates 1).
                    # Wait, flow conservation ensures this.
                    # So we just follow the unique outgoing edge.
                    next_node = nexts[0]
                    edge_key = (current, next_node)
                    if edge_key in used_edges:
                        break # Should not happen
                    used_edges.add(edge_key)
                    path.append(next_node)
                    current = next_node
                paths.append(path)
                
            return paths
            
        except Exception as e:
            return None

    def greedy_path_cover(edges):
        """Fallback greedy."""
        if not edges:
            return []
        edge_set = set()
        for e in edges:
            u, v = e
            edge_set.add((u, v) if u <= v else (v, u))
        adj = defaultdict(list)
        for (u, v) in edge_set:
            adj[u].append(v)
            adj[v].append(u)
        used_edges = set()
        paths = []
        while edge_set - used_edges:
            remaining = edge_set - used_edges
            if not remaining: break
            rem_adj = defaultdict(list)
            for (u, v) in remaining:
                rem_adj[u].append(v)
                rem_adj[v].append(u)
            start = max(rem_adj.keys(), key=lambda x: len(rem_adj[x])) if rem_adj else None
            if start is None: break
            path = [start]
            visited_in_path = {start}
            current = start
            while True:
                neighbors = [n for n in rem_adj[current] if n not in visited_in_path]
                if not neighbors: break
                best_neighbor = max(neighbors, key=lambda nb: len([x for x in rem_adj[nb] if x not in visited_in_path]))
                path.append(best_neighbor)
                visited_in_path.add(best_neighbor)
                current = best_neighbor
            path_edges = set()
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                key = (u, v) if u <= v else (v, u)
                path_edges.add(key)
            valid = all(e in remaining for e in path_edges)
            if valid and path_edges:
                used_edges |= path_edges
                paths.append(path)
            else:
                e = next(iter(remaining))
                used_edges.add(e)
                paths.append([e[0], e[1]])
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > _DEADLINE - 10:
            edges = instance(*spec)
            results.append(greedy_path_cover(edges))
            continue

        n, p, seed = spec
        edges = instance(n, p, seed)
        
        # Try ILP
        paths = solve_min_path_cover_ilp(n, edges)
        
        # Validate
        if paths is None:
            paths = greedy_path_cover(edges)
        else:
            edge_set = set()
            for (u, v) in edges:
                edge_set.add((u, v) if u <= v else (v, u))
            covered = set()
            valid = True
            for pth in paths:
                if len(pth) < 2:
                    valid = False
                    break
                if len(set(pth)) != len(pth):
                    valid = False
                    break
                for i in range(len(pth) - 1):
                    u, v = pth[i], pth[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    if key not in edge_set:
                        valid = False
                        break
                    if key in covered:
                        valid = False
                        break
                    covered.add(key)
                if not valid:
                    break
            if not valid or covered != edge_set:
                paths = greedy_path_cover(edges)

        results.append(paths)

    return results
