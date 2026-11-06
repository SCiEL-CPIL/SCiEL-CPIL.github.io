"""
Snake-in-the-Box in Q9 using MILP (Mixed Integer Linear Programming) formulation.

Structural change: instead of heuristic local search / DFS backtracking,
formulate the problem as a maximum induced path (snake) in Q_9 as an MILP:
  - x[v] = 1 if vertex v is in the snake
  - y[e] = 1 if edge e is in the snake
  - Degree constraints: each included vertex has degree 0 or 2 (endpoints have degree 1)
  - Connectivity enforced via MTZ (Miller-Tucker-Zemlin) subtour elimination
  - Objective: maximize number of edges = sum(y[e])

Solve with scipy.optimize.milp (HiGHS solver) with a time limit.
Then extract the path from the solution and extend greedily if possible.
"""
import time
import random
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

N = 9
TOTAL_VERTS = 1 << N  # 512

def _precompute_neighbors():
    neighbors = [[] for _ in range(TOTAL_VERTS)]
    for v in range(TOTAL_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    return neighbors

_NEIGHBORS = _precompute_neighbors()

def _build_milp():
    """
    Build MILP for maximum induced path in Q_9.
    
    Variables:
      x[v] for v in 0..511: binary, 1 if vertex v is in the snake
      y[e] for each undirected edge e: binary, 1 if edge is in the snake
      t[v] for v in 0..511: continuous, position in path (MTZ)
    
    For 512 vertices and 9*512/2 = 2304 edges:
      Total variables: 512 + 2304 + 512 = 3328
    
    Constraints:
      1. Degree: for each vertex v, sum of incident y[e] = 2*x[v] - endpoint_indicator
         Actually: degree(v) <= 2*x[v], and degree(v) >= x[v] (if included, at least 1 edge unless single vertex)
         More precisely for a path:
           - If x[v]=1, degree is 1 or 2
           - If x[v]=0, degree is 0
         So: degree(v) <= 2*x[v] and degree(v) >= x[v]
         But we also need exactly two endpoints with degree 1.
         For simplicity: degree(v) <= 2*x[v] and degree(v) >= x[v]
         This allows cycles if not for MTZ.
      
      2. MTZ subtour elimination:
         For each directed edge (u,v) where u!=v:
           If y[u,v]=1 then t[v] = t[u] + 1
           Enforced by: t[v] - t[u] >= 1 - M*(1 - y[u,v])
           and t[v] - t[u] <= M*(1 - y[u,v])
           where M is an upper bound on path length (use TOTAL_VERTS)
         Also: 0 <= t[v] <= TOTAL_VERTS - 1
         And t[v] <= (TOTAL_VERTS-1)*x[v] (if v not in path, t[v]=0)
      
      3. Induced path constraint: non-consecutive vertices must not be adjacent.
         For each pair of vertices (u,v) that are adjacent (edge e), if both are in the path
         but the edge is NOT selected, that violates the induced path property.
         So: x[u] + x[v] <= 1 + y[u,v] for each edge (u,v)
         This ensures: if both u and v are in the snake, the edge between them must be in the snake.
    
    Objective: maximize sum of y[e] (number of edges)
    """
    # Enumerate edges (undirected)
    edges = []
    edge_index = {}
    for v in range(TOTAL_VERTS):
        for w in _NEIGHBORS[v]:
            if v < w:
                idx = len(edges)
                edges.append((v, w))
                edge_index[(v, w)] = idx
    
    num_edges = len(edges)
    # Variable ordering: x[0..511], y[0..num_edges-1], t[0..511]
    num_x = TOTAL_VERTS
    num_y = num_edges
    num_t = TOTAL_VERTS
    total_vars = num_x + num_y + num_t
    
    # Objective: maximize sum(y) -> minimize -sum(y)
    c = np.zeros(total_vars)
    for e in range(num_y):
        c[num_x + e] = -1.0  # minimize -y[e]
    
    # Bounds
    lb = np.zeros(total_vars)
    ub = np.zeros(total_vars)
    for i in range(num_x):
        ub[i] = 1.0  # x binary
    for i in range(num_y):
        ub[num_x + i] = 1.0  # y binary
    for i in range(num_t):
        ub[num_x + num_y + i] = float(TOTAL_VERTS - 1)  # t continuous
    
    # Integrality: x and y are binary (1), t is continuous (0)
    integrality = np.zeros(total_vars)
    for i in range(num_x):
        integrality[i] = 1
    for i in range(num_y):
        integrality[num_x + i] = 1
    
    # Build constraint matrix
    # We'll build constraints as a list of rows
    A_rows = []
    lb_rows = []
    ub_rows = []
    
    # Constraint 1: degree(v) <= 2*x[v] for each v
    # sum_{e incident to v} y[e] - 2*x[v] <= 0
    for v in range(TOTAL_VERTS):
        row = np.zeros(total_vars)
        for w in _NEIGHBORS[v]:
            u, w2 = (v, w) if v < w else (w, v)
            e_idx = edge_index[(u, w2)]
            row[num_x + e_idx] = 1.0
        row[v] = -2.0
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(0.0)
    
    # Constraint 2: degree(v) >= x[v] for each v (if included, at least 1 edge)
    # sum_{e incident to v} y[e] - x[v] >= 0
    for v in range(TOTAL_VERTS):
        row = np.zeros(total_vars)
        for w in _NEIGHBORS[v]:
            u, w2 = (v, w) if v < w else (w, v)
            e_idx = edge_index[(u, w2)]
            row[num_x + e_idx] = 1.0
        row[v] = -1.0
        A_rows.append(row)
        lb_rows.append(0.0)
        ub_rows.append(np.inf)
    
    # Constraint 3: Induced path: x[u] + x[v] <= 1 + y[u,v] for each edge
    # x[u] + x[v] - y[u,v] <= 1
    for e_idx, (u, v) in enumerate(edges):
        row = np.zeros(total_vars)
        row[u] = 1.0
        row[v] = 1.0
        row[num_x + e_idx] = -1.0
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(1.0)
    
    # Constraint 4: MTZ subtour elimination
    # For each directed pair (u,v) where there's an edge:
    #   t[v] - t[u] >= 1 - M*(1 - y[u,v])  =>  t[v] - t[u] + M*y[u,v] >= M
    #   t[v] - t[u] <= M*(1 - y[u,v])     =>  t[v] - t[u] + M*y[u,v] <= M
    # Also: t[v] <= (TOTAL_VERTS-1)*x[v]  =>  t[v] - (TOTAL_VERTS-1)*x[v] <= 0
    # And: t[v] >= 0 (already in bounds)
    
    M = float(TOTAL_VERTS)
    
    # t[v] - (TOTAL_VERTS-1)*x[v] <= 0
    for v in range(TOTAL_VERTS):
        row = np.zeros(total_vars)
        row[num_x + num_y + v] = 1.0
        row[v] = -(TOTAL_VERTS - 1)
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(0.0)
    
    # For each undirected edge (u,v), we have two directed constraints:
    # Direction u->v: t[v] - t[u] + M*y[u,v] >= M  and  t[v] - t[u] + M*y[u,v] <= M
    # Direction v->u: t[u] - t[v] + M*y[u,v] >= M  and  t[u] - t[v] + M*y[u,v] <= M
    # But we only need one direction per edge for MTZ. Let's use:
    # For edge (u,v) with u<v: 
    #   t[v] - t[u] >= 1 - M*(1-y)  =>  t[v] - t[u] + M*y >= M
    #   t[u] - t[v] >= 1 - M*(1-y)  =>  t[u] - t[v] + M*y >= M
    # This enforces that if y=1, then |t[u]-t[v]| >= 1, and combined with the upper bound, |t[u]-t[v]|=1
    
    for e_idx, (u, v) in enumerate(edges):
        # t[v] - t[u] + M*y >= M
        row = np.zeros(total_vars)
        row[num_x + num_y + v] = 1.0
        row[num_x + num_y + u] = -1.0
        row[num_x + e_idx] = M
        A_rows.append(row)
        lb_rows.append(M)
        ub_rows.append(np.inf)
        
        # t[u] - t[v] + M*y >= M
        row = np.zeros(total_vars)
        row[num_x + num_y + u] = 1.0
        row[num_x + num_y + v] = -1.0
        row[num_x + e_idx] = M
        A_rows.append(row)
        lb_rows.append(M)
        ub_rows.append(np.inf)
    
    A = np.array(A_rows)
    lb_arr = np.array(lb_rows)
    ub_arr = np.array(ub_rows)
    
    return c, A, lb_arr, ub_arr, integrality, lb, ub, edges, edge_index

def _extract_path_from_solution(sol, edges, edge_index):
    """Extract the snake path from MILP solution."""
    if sol.status != 0:
        return None
    
    x = sol.x[:TOTAL_VERTS]
    y = sol.x[TOTAL_VERTS:TOTAL_VERTS + len(edges)]
    
    # Get selected vertices
    selected = [v for v in range(TOTAL_VERTS) if x[v] > 0.5]
    if len(selected) < 2:
        return None
    
    # Get selected edges
    selected_edges = set()
    for i, (u, v) in enumerate(edges):
        if y[i] > 0.5:
            selected_edges.add((u, v))
            selected_edges.add((v, u))
    
    # Build adjacency from selected edges
    adj = {v: [] for v in selected}
    for (u, v) in selected_edges:
        if u in adj and v in adj:
            adj[u].append(v)
            adj[v].append(u)
    
    # Find endpoints (degree 1)
    endpoints = [v for v in selected if len(adj[v]) == 1]
    if len(endpoints) != 2:
        # Try to find a path anyway
        endpoints = [selected[0]]
    
    # Traverse from one endpoint
    if not endpoints:
        return None
    
    start = endpoints[0]
    path = [start]
    visited = {start}
    cur = start
    while True:
        nexts = [w for w in adj[cur] if w not in visited]
        if not nexts:
            break
        cur = nexts[0]
        path.append(cur)
        visited.add(cur)
    
    return path

def _greedy_extend(path, used_set=None):
    """Extend a path greedily from both ends."""
    if used_set is None:
        used_set = set(path)
    
    # Try extending from the end
    while True:
        cur = path[-1]
        found = False
        for b in range(N):
            v = cur ^ (1 << b)
            if v not in used_set:
                valid = True
                for w in _NEIGHBORS[v]:
                    if w in used_set and w != cur:
                        valid = False
                        break
                if valid:
                    path.append(v)
                    used_set.add(v)
                    found = True
                    break
        if not found:
            break
    
    # Try extending from the beginning
    while True:
        cur = path[0]
        found = False
        for b in range(N):
            v = cur ^ (1 << b)
            if v not in used_set:
                valid = True
                for w in _NEIGHBORS[v]:
                    if w in used_set and w != cur:
                        valid = False
                        break
                if valid:
                    path.insert(0, v)
                    used_set.add(v)
                    found = True
                    break
        if not found:
            break
    
    return path

def _try_insert(path, vertex):
    """Try to insert a vertex into the path."""
    used_set = set(path)
    if vertex in used_set:
        return None
    
    # Try inserting between consecutive vertices
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        # Check if vertex is adjacent to both u and v
        if (vertex ^ u).bit_count() == 1 and (vertex ^ v).bit_count() == 1:
            # Check induced path property
            valid = True
            for w in _NEIGHBORS[vertex]:
                if w in used_set and w != u and w != v:
                    valid = False
                    break
            if valid:
                new_path = path[:i+1] + [vertex] + path[i+1:]
                return new_path
    
    # Try prepending
    if (vertex ^ path[0]).bit_count() == 1:
        valid = True
        for w in _NEIGHBORS[vertex]:
            if w in used_set and w != path[0]:
                valid = False
                break
        if valid:
            return [vertex] + path
    
    # Try appending
    if (vertex ^ path[-1]).bit_count() == 1:
        valid = True
        for w in _NEIGHBORS[vertex]:
            if w in used_set and w != path[-1]:
                valid = False
                break
        if valid:
            return path + [vertex]
    
    return None

def construct():
    deadline = time.time() + 145.0
    rng = random.Random(42)
    
    # First, get a good initial solution via greedy
    greedy_path = [0]
    greedy_used = {0}
    while True:
        cur = greedy_path[-1]
        found = False
        for b in range(N):
            v = cur ^ (1 << b)
            if v not in greedy_used:
                valid = True
                for w in _NEIGHBORS[v]:
                    if w in greedy_used and w != cur:
                        valid = False
                        break
                if valid:
                    greedy_path.append(v)
                    greedy_used.add(v)
                    found = True
                    break
        if not found:
            break
    
    best_path = list(greedy_path)
    best_len = len(best_path)
    
    # Try MILP
    try:
        c, A, lb_arr, ub_arr, integrality, lb, ub, edges, edge_index = _build_milp()
        
        constraints = LinearConstraint(A, lb_arr, ub_arr)
        bounds = Bounds(lb, ub)
        
        # Solve with time limit
        time_limit = min(120.0, deadline - time.time() - 5.0)
        if time_limit > 10.0:
            sol = milp(c, constraints=constraints, integrality=integrality, bounds=bounds,
                       options={"time_limit": time_limit})
            
            if sol.status == 0:
                path = _extract_path_from_solution(sol, edges, edge_index)
                if path and len(path) > best_len:
                    # Greedy extend
                    path = _greedy_extend(path)
                    best_path = path
                    best_len = len(path)
    except Exception as e:
        pass
    
    # Local search: try to improve via insertions
    if time.time() < deadline - 5.0:
        used_set = set(best_path)
        for _ in range(10000):
            if time.time() > deadline - 5.0:
                break
            # Pick a random vertex not in path
            v = rng.randint(0, TOTAL_VERTS - 1)
            if v in used_set:
                continue
            new_path = _try_insert(best_path, v)
            if new_path:
                best_path = new_path
                used_set = set(new_path)
                best_len = len(new_path)
    
    # Final greedy extension
    best_path = _greedy_extend(best_path)
    best_len = len(best_path)
    
    return best_path

# EVOLVE-BLOCK-START
# The construct function is the entry point.
# EVOLVE-BLOCK-END
