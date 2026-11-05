import time
import random
from random import Random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

DEADLINE_MARGIN = 10  # seconds before 150s

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def build_adj(edges):
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj

def find_2_factor_ilp(edges, adj, max_time=20.0):
    """
    Find a maximum cardinality 2-factor (union of edge-disjoint cycles)
    using Integer Linear Programming.
    """
    if not edges:
        return []
    
    # Map edges to indices
    edge_list = list(edges)
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    m = len(edge_list)
    n_vertices = max([v for e in edge_list for v in e]) + 1 if edge_list else 0
    
    # Variables: x_i = 1 if edge i is in the 2-factor
    # Objective: Maximize sum(x_i)
    c = -np.ones(m)  # Minimize -sum(x_i)
    
    # Constraints:
    # 1. For each vertex v, sum of x_i for edges incident to v must be <= 2.
    #    (Actually, for a 2-factor, it must be exactly 2 for vertices in the factor,
    #     but since we want a subset, <= 2 is correct for the "subgraph" part,
    #     but we need to ensure no isolated vertices or paths.
    #     A 2-factor is a spanning 2-regular subgraph. However, we might not be able
    #     to cover all vertices. So we seek a 2-regular subgraph on a subset of vertices.
    #     Constraint: degree(v) <= 2.
    #     Additionally, we need to prevent subtours (cycles smaller than the whole component?
    #     No, we WANT cycles. We want a union of cycles.
    #     The constraint deg(v) <= 2 allows paths and cycles.
    #     To ensure only cycles, we need to forbid paths.
    #     A standard trick: If we maximize the number of edges, and we have deg(v)<=2,
    #     the optimal solution will be a union of cycles and possibly some paths?
    #     Actually, if there is a path, we can often extend it or close it if possible.
    #     But ILP doesn't know that.
    #     However, any path component has two endpoints with degree 1.
    #     If we maximize edges, we prefer cycles (degree 2) over paths (degree 1 endpoints).
    #     So the objective naturally pushes towards cycles.
    #     Let's check: A path of length L has L edges and L+1 vertices.
    #     A cycle of length L has L edges and L vertices.
    #     For the same number of edges, a cycle uses fewer vertices? No.
    #     But a cycle is "more efficient" in terms of closing the loop.
    #     The key is: A 2-factor is a set of cycles.
    #     If we just constrain deg(v) <= 2, we might get a path.
    #     To strictly enforce cycles, we can add constraints that forbid degree 1?
    #     No, that would force a spanning 2-factor which might not exist.
    #     
    #     Let's stick to deg(v) <= 2. The objective max sum(x) will prefer
    #     configurations where degrees are 2 because they allow more edges to be added
    #     locally without violating the degree constraint as soon as possible?
    #     Actually, consider a path u-v-w. deg(u)=1, deg(v)=2, deg(w)=1.
    #     If we have an edge u-w, we can form a cycle u-v-w-u.
    #     The ILP will prefer the cycle if the edge exists because it adds an edge
    #     without increasing the degree of any vertex beyond what was already used?
    #     Wait, in the path u-v-w, edges (u,v) and (v,w) are used.
    #     If we add (u,w), deg(u) becomes 2, deg(w) becomes 2.
    #     So the cycle uses 3 edges, the path uses 2.
    #     Maximize edges -> prefers cycle.
    #     What if no closing edge exists? Then it stays a path.
    #     A path component is not a valid "cycle piece".
    #     We need to extract only the cycle components from the ILP solution.
    
    # Build incidence matrix A for vertex-edge constraints
    # A is n_vertices x m
    A = np.zeros((n_vertices, m))
    for i, (u, v) in enumerate(edge_list):
        A[u, i] = 1
        A[v, i] = 1
        
    # Constraints: A @ x <= 2
    # x is binary
    integrality = np.ones(m)
    
    # Bounds for x: [0, 1]
    bounds = Bounds(lb=0, ub=1)
    
    # Linear constraints: A @ x <= 2
    cons = LinearConstraint(A, lb=None, ub=2)
    
    start_time = time.time()
    
    try:
        res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds, 
                   options={"time_limit": max_time, "mip_rel_gap": 0.0})
        
        if time.time() - start_time > max_time:
            return []
            
        if res.status == 0 or res.status == 1: # Optimal or best found
            x = res.x
            selected_edges = set()
            for i in range(m):
                if x[i] > 0.5:
                    selected_edges.add(edge_list[i])
            
            # Extract cycles from selected_edges
            # Build adjacency of selected edges
            sel_adj = {}
            for (u, v) in selected_edges:
                sel_adj.setdefault(u, set()).add(v)
                sel_adj.setdefault(v, set()).add(u)
            
            visited = set()
            cycles = []
            
            for node in sel_adj:
                if node in visited:
                    continue
                # BFS to find component
                component = []
                stack = [node]
                visited.add(node)
                while stack:
                    n = stack.pop()
                    component.append(n)
                    for nb in sel_adj[n]:
                        if nb not in visited:
                            visited.add(nb)
                            stack.append(nb)
                
                # Check if component is a cycle
                is_cycle = True
                for v in component:
                    if len(sel_adj[v]) != 2:
                        is_cycle = False
                        break
                
                if is_cycle and len(component) >= 3:
                    # Reconstruct cycle order
                    cycle = [component[0]]
                    prev = None
                    curr = component[0]
                    while True:
                        cycle.append(curr)
                        nexts = [nb for nb in sel_adj[curr] if nb != prev]
                        if not nexts:
                            break
                        next_node = nexts[0]
                        if next_node == component[0]:
                            break
                        prev = curr
                        curr = next_node
                    
                    if len(cycle) > 1 and cycle[0] == cycle[-1]:
                        cycle = cycle[:-1]
                    if len(cycle) >= 3:
                        cycles.append(cycle)
                        
            return cycles
    except Exception:
        return []

def decompose_graph(edges, max_time=40.0):
    """
    Decompose edges by:
    1. Finding a max 2-factor via ILP.
    2. Removing those cycles.
    3. Repeating on remaining edges until time runs out or no more 2-factors.
    4. Greedy short cycle extraction for leftovers.
    5. Single edges for the rest.
    """
    start = time.time()
    remaining = set(tuple(sorted(e)) for e in edges)
    pieces = []
    rng = Random(42)
    
    # Phase 1: ILP based 2-factor extraction
    while remaining:
        cur_time = time.time()
        if cur_time - start > max_time * 0.7: # Leave some time for phase 2
            break
        
        adj = build_adj(list(remaining))
        # ILP can be slow if graph is large, but n<=30 is small.
        # We limit time per ILP call.
        cycles = find_2_factor_ilp(remaining, adj, max_time=5.0)
        
        if cycles:
            for cycle in cycles:
                # Verify cycle edges are in remaining
                cyc_edges = set()
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (u, v) if u < v else (v, u)
                    cyc_edges.add(e)
                
                if cyc_edges <= remaining:
                    pieces.append(cycle[:])
                    for e in cyc_edges:
                        remaining.discard(e)
        else:
            # No more 2-factors found by ILP (or timeout)
            break
            
    # Phase 2: Greedy extraction of short cycles (length 3-5) from remaining
    # This is fast and handles leftovers that don't form a 2-factor together
    if remaining:
        adj = build_adj(list(remaining))
        nodes = list(set([u for (u, v) in remaining] + [v for (u, v) in remaining]))
        
        # Try to find cycles of length 3, 4, 5
        # We use a randomized DFS with a depth limit
        while remaining and (time.time() - start < max_time * 0.9):
            found = False
            # Randomize node order
            rng.shuffle(nodes)
            for start_node in nodes:
                if len(remaining) < 3:
                    break
                # DFS with depth limit 6
                visited = {start_node}
                path = [start_node]
                stack = [(start_node, iter(sorted(adj.get(start_node, []), key=lambda x: rng.random())))]
                while stack and not found:
                    node, neighbors = stack[-1]
                    advanced = False
                    for nb in neighbors:
                        e = (node, nb) if node < nb else (nb, node)
                        if e not in remaining:
                            continue
                        if nb == start_node and len(path) >= 3:
                            # Found a cycle
                            cyc_edges = set()
                            for i in range(len(path)):
                                u = path[i]
                                v = path[(i + 1) % len(path)]
                                ce = (u, v) if u < v else (v, u)
                                cyc_edges.add(ce)
                            if cyc_edges <= remaining:
                                pieces.append(path[:])
                                for ce in cyc_edges:
                                    remaining.discard(ce)
                                found = True
                                break
                        if nb not in visited and len(path) < 6: # Depth limit
                            visited.add(nb)
                            path.append(nb)
                            stack.append((nb, iter(sorted(adj.get(nb, []), key=lambda x: rng.random()))))
                            advanced = True
                            break
                    if found:
                        break
                    if not advanced:
                        stack.pop()
                        if path:
                            last = path.pop()
                            visited.discard(last)
                if found:
                    # Update adj for remaining
                    adj = build_adj(list(remaining))
                    break
            
            if not found:
                # No short cycles found with current heuristic, break to single edges
                break

    # Phase 3: Single edges
    for e in remaining:
        pieces.append([e[0], e[1]])
        
    return pieces

def construct():
    start = time.time()
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    per_graph_time = max(1.0, (DEADLINE_MARGIN - 5) / 3)
    
    for i, spec in enumerate(specs):
        edges = instance(*spec)
        cur_time = time.time()
        remaining_time = min(per_graph_time, DEADLINE_MARGIN - (cur_time - start))
        if remaining_time < 1.0:
            # No time left, return all edges as single pieces
            results.append([list(e) for e in edges])
            continue
        
        pieces = decompose_graph(edges, max_time=remaining_time)
        results.append(pieces)
        
        # Check deadline
        cur_time = time.time()
        if cur_time - start > DEADLINE_MARGIN - 3:
            # Return remaining as individual edges
            for remaining_spec in specs[i + 1:]:
                e = instance(*remaining_spec)
                results.append([list(e) for e in e])
            break
    
    # Ensure we have exactly 3 results
    while len(results) < 3:
        results.append([])
    
    return results[:3]
