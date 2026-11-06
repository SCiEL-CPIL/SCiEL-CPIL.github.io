"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations
from scipy.optimize import milp, LinearConstraint, Bounds

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

# Precompute edge position lookup
_edge_pos = {}
for i, e in enumerate(P['edge_order']):
    _edge_pos[(min(e[0], e[1]), max(e[0], e[1]))] = i + 1  # 1-based

def _cycle_score(cycle):
    """Compute the score of a Hamilton cycle: max edge position in the cycle."""
    n = len(cycle)
    max_pos = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        key = (min(u, v), max(u, v))
        pos = _edge_pos[key]
        if pos > max_pos:
            max_pos = pos
    return max_pos

def _find_hamilton_cycle_ilp(adj, n, time_limit=10.0):
    """
    Find a Hamilton cycle in a graph using ILP (scipy.optimize.milp).
    adj: list of sets, adj[i] = set of neighbors of vertex i
    Returns a list of vertices forming the cycle, or None if not found.
    """
    # Build edge list
    edges = []
    edge_idx = {}
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
                edge_idx[(u, v)] = len(edges) - 1
    
    m = len(edges)
    if m < n:
        return None
    
    # Variables: x_e for each edge (binary), y_v for each vertex (binary, but we fix y_0 = 1)
    # Actually, for TSP ILP, we need subtour elimination.
    # Simplified: use degree constraints + MTZ subtour elimination
    # x_e: binary, 1 if edge e is in the cycle
    # u_v: continuous, position of vertex v in the cycle (MTZ)
    
    # Decision variables: x_0, ..., x_{m-1} (binary), u_1, ..., u_{n-1} (continuous)
    # u_0 is fixed to 0
    # Total variables: m + (n-1)
    
    # Objective: minimize sum x_e (any Hamilton cycle has exactly n edges, so this is constant)
    # Actually, we just need feasibility. Set objective to 0.
    c = np.zeros(m + (n - 1))
    
    # Constraints:
    # 1. Degree constraints: for each vertex v, sum of x_e over edges incident to v = 2
    # 2. MTZ: for each edge (u,v), u_u - u_v + n * x_{uv} <= n - 1 (if u != 0)
    #    and u_v - u_u + n * x_{uv} <= n - 1 (if v != 0)
    #    Actually standard MTZ: for edge (i,j), u_i - u_j + n*x_ij <= n-1
    #    with u_0 = 0, u_i in [1, n-1] for i != 0
    
    # Build A matrix for LinearConstraint
    # We'll use separate constraint blocks
    
    # Degree constraints: n rows, m + (n-1) columns
    A_deg = np.zeros((n, m + n - 1))
    for e_idx, (u, v) in enumerate(edges):
        A_deg[u, e_idx] = 1
        A_deg[v, e_idx] = 1
    lb_deg = np.full(n, 2)
    ub_deg = np.full(n, 2)
    
    # MTZ constraints: for each edge (u,v), two constraints (one for each direction)
    # u_u - u_v + n*x_uv <= n-1  =>  x_uv + u_u - u_v <= n-1
    # u_v - u_u + n*x_uv <= n-1  =>  x_uv - u_u + u_v <= n-1
    # But we only need one per edge if we enforce u ordering.
    # Standard: for edge (i,j), if i != 0 and j != 0:
    #   u_i - u_j + n*x_ij <= n-1
    # If i == 0: u_j >= 1 (handled by bounds)
    # If j == 0: u_i <= n-1 (handled by bounds)
    
    A_mt = []
    lb_mt = []
    ub_mt = []
    
    for e_idx, (u, v) in enumerate(edges):
        # u_u - u_v + n*x_uv <= n-1
        # x_uv + u_u - u_v <= n-1
        row = np.zeros(m + n - 1)
        row[e_idx] = 1
        if u != 0:
            row[m + u - 1] = 1  # u_u
        if v != 0:
            row[m + v - 1] = -1  # -u_v
        A_mt.append(row)
        lb_mt.append(-np.inf)
        ub_mt.append(n - 1)
        
        # u_v - u_u + n*x_uv <= n-1
        # x_uv - u_u + u_v <= n-1
        row2 = np.zeros(m + n - 1)
        row2[e_idx] = 1
        if u != 0:
            row2[m + u - 1] = -1
        if v != 0:
            row2[m + v - 1] = 1
        A_mt.append(row2)
        lb_mt.append(-np.inf)
        ub_mt.append(n - 1)
    
    A_mt = np.array(A_mt) if A_mt else np.zeros((0, m + n - 1))
    
    # Combine constraints
    A = np.vstack([A_deg, A_mt])
    lb = np.concatenate([lb_deg, lb_mt])
    ub = np.concatenate([ub_deg, ub_mt])
    
    # Bounds: x_e in [0, 1], u_v in [0, n-1]
    lb_var = np.zeros(m + n - 1)
    ub_var = np.ones(m + n - 1)
    ub_var[m:] = n - 1  # u_v bounds
    
    # Integrality: x_e are binary (integrality=1), u_v are continuous (integrality=0)
    integrality = np.concatenate([np.ones(m, dtype=int), np.zeros(n - 1, dtype=int)])
    
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, lb, ub),
            integrality=integrality,
            bounds=Bounds(lb_var, ub_var)
        )
    except Exception:
        return None
    
    if not res.success:
        return None
    
    # Extract cycle from x variables
    x = res.x[:m]
    cycle_edges = []
    for i, (u, v) in enumerate(edges):
        if x[i] > 0.5:
            cycle_edges.append((u, v))
    
    if len(cycle_edges) != n:
        return None
    
    # Reconstruct cycle from edges
    adj_cycle = [[] for _ in range(n)]
    for u, v in cycle_edges:
        adj_cycle[u].append(v)
        adj_cycle[v].append(u)
    
    # Verify it's a single cycle
    for v in range(n):
        if len(adj_cycle[v]) != 2:
            return None
    
    # Traverse the cycle starting from 0
    cycle = [0]
    visited = {0}
    current = 0
    for _ in range(n - 1):
        nexts = adj_cycle[current]
        nxt = None
        for nb in nexts:
            if nb not in visited:
                nxt = nb
                break
        if nxt is None:
            return None
        cycle.append(nxt)
        visited.add(nxt)
        current = nxt
    
    # Verify closing edge
    if 0 not in adj_cycle[current]:
        return None
    
    return cycle

def _two_opt_improve(cycle, max_iter=50000, time_limit=20.0):
    """Improve a cycle using best-improvement 2-opt moves."""
    n = len(cycle)
    best = cycle[:]
    best_score = _cycle_score(best)
    
    deadline = time.time() + time_limit
    improved = True
    iters = 0
    while improved and iters < max_iter and time.time() < deadline:
        improved = False
        iters += 1
        best_move = None
        best_move_score = best_score
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                u = best[i]
                v = best[i + 1]
                w = best[j]
                x = best[(j + 1) % n]
                
                # Compute new score efficiently
                # Edges removed: (u,v) and (w,x)
                # Edges added: (u,w) and (v,x)
                key_uv = (min(u, v), max(u, v))
                key_wx = (min(w, x), max(w, x))
                key_1 = (min(u, w), max(u, w))
                key_2 = (min(v, x), max(v, x))
                
                pos_uv = _edge_pos[key_uv]
                pos_wx = _edge_pos[key_wx]
                pos_1 = _edge_pos[key_1]
                pos_2 = _edge_pos[key_2]
                
                # New max = max of all other edges in cycle
                # We need to know if the removed edges contained the max
                # Quick estimate: new_score <= max(best_score, pos_1, pos_2)
                # But we need exact. For speed, compute full score only if promising.
                if max(pos_1, pos_2) >= best_score:
                    # Could be worse or same, skip unless it removes the max edge
                    # Check if pos_uv or pos_wx == best_score
                    if pos_uv < best_score and pos_wx < best_score:
                        # Removing non-max edges, adding edges with pos >= best_score
                        # New score will be >= best_score, skip
                        continue
                
                new_cycle = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                new_score = _cycle_score(new_cycle)
                if new_score < best_move_score:
                    best_move_score = new_score
                    best_move = (i, j)
        
        if best_move is not None:
            i, j = best_move
            best = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
            best_score = best_move_score
            improved = True
    
    return best, best_score

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin before 90s kill
    
    # Start with identity permutation
    best_cycle = list(range(n))
    best_score = _cycle_score(best_cycle)
    
    # Binary search for minimum threshold K such that G_K contains a Hamilton cycle
    lo = n
    hi = best_score
    
    # Binary search using ILP for feasibility
    while lo < hi and time.time() < deadline - 15.0:
        mid = (lo + hi) // 2
        # Build adjacency list for edges with position <= mid
        adj = [set() for _ in range(n)]
        for i, e in enumerate(P['edge_order']):
            if i + 1 > mid:
                break
            u, v = e
            adj[u].add(v)
            adj[v].add(u)
        
        # Check if every vertex has degree >= 2
        if any(len(adj[v]) < 2 for v in range(n)):
            lo = mid + 1
            continue
        
        # Try to find Hamilton cycle using ILP
        cycle = _find_hamilton_cycle_ilp(adj, n, time_limit=15.0)
        if cycle is not None:
            score = _cycle_score(cycle)
            if score < best_score:
                best_score = score
                best_cycle = cycle
            hi = mid
        else:
            lo = mid + 1
    
    # After binary search, try to improve with local search
    if time.time() < deadline - 5.0:
        improved_cycle, improved_score = _two_opt_improve(best_cycle, max_iter=50000, time_limit=deadline - time.time() - 3.0)
        if improved_score < best_score:
            best_score = improved_score
            best_cycle = improved_cycle
    
    return best_cycle
# EVOLVE-BLOCK-END
