import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']

_edge_pos = {}
for _i, (_u, _v) in enumerate(EDGE_ORDER, 1):
    _edge_pos[(min(_u, _v), max(_u, _v))] = _i

def _edge_idx(u, v):
    return _edge_pos[(min(u, v), max(u, v))]

def _cycle_score(cycle):
    return max(_edge_idx(cycle[i], cycle[(i+1) % N]) for i in range(N))

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # Safety margin for 90s limit
    
    # Precompute edge indices matrix for fast lookup
    idx_mat = [[0]*N for _ in range(N)]
    for u in range(N):
        for v in range(N):
            if u != v:
                idx_mat[u][v] = _edge_idx(u, v)
    
    # Trivial cycle
    trivial = list(range(N))
    best_score = _cycle_score(trivial)
    best_cycle = trivial[:]
    
    def find_cycle_milp(K, time_limit=5.0):
        """Find a Hamilton cycle using only edges with index <= K via MILP."""
        n = N
        # Build adjacency: which edges are available
        adj = [[False]*n for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u != v and idx_mat[u][v] <= K:
                    adj[u][v] = True
        
        # Check necessary conditions: each vertex must have degree >= 2
        for u in range(n):
            deg = sum(adj[u][v] for v in range(n) if v != u)
            if deg < 2:
                return None
        
        # MILP formulation:
        # x[u,v] = 1 if edge (u,v) is in the cycle
        # Constraints:
        # 1. Flow conservation: for each vertex u, sum_v x[u,v] = sum_v x[v,u] = 1
        # 2. No subtours: for each subset S of vertices with 2 <= |S| < n,
        #    sum_{u in S, v not in S} x[u,v] >= 1 (cut constraint)
        #    We'll add these lazily or use MTZ formulation instead.
        
        # Use MTZ (Miller-Tucker-Zemlin) formulation to avoid subtour constraints:
        # x[u,v] binary for u != v
        # y[u] integer, 1 <= y[u] <= n-1, y[start] = 1
        # y[u] - y[v] + n*x[u,v] <= n-1 for all u != v, u != start, v != start
        
        # Variables: x[u,v] for all ordered pairs (u,v), u != v
        # Plus y[u] for each vertex u
        # Total: n*(n-1) + n variables
        
        start = 0
        # Index for x[u,v]: u*n + v
        # Index for y[u]: n*n + u
        num_x = n * n
        num_y = n
        total_vars = num_x + num_y
        
        # Objective: minimize sum of x (should be exactly n for a Hamilton cycle)
        c = np.zeros(total_vars)
        for u in range(n):
            for v in range(n):
                if u != v:
                    c[u*n + v] = 1.0
        
        # Bounds: x binary, y integer in [1, n-1]
        lb = np.zeros(total_vars)
        ub = np.zeros(total_vars)
        for u in range(n):
            for v in range(n):
                if u != v:
                    lb[u*n + v] = 0
                    ub[u*n + v] = 1
            lb[num_x + u] = 1
            ub[num_x + u] = n - 1
        
        # Fix y[start] = 1
        lb[num_x + start] = 1
        ub[num_x + start] = 1
        
        # Constraints:
        # 1. Out-degree = 1 for each vertex: sum_v x[u,v] = 1
        # 2. In-degree = 1 for each vertex: sum_v x[v,u] = 1
        # 3. MTZ: y[u] - y[v] + n*x[u,v] <= n-1 for u != start, v != start
        # 4. Only allow edges that exist in G_K: x[u,v] = 0 if not adj[u,v]
        
        # Build constraint matrix A and bounds
        A_rows = []
        A_lbs = []
        A_ubs = []
        
        # Out-degree constraints
        for u in range(n):
            row = np.zeros(total_vars)
            for v in range(n):
                if u != v and adj[u][v]:
                    row[u*n + v] = 1.0
            A_rows.append(row)
            A_lbs.append(1.0)
            A_ubs.append(1.0)
        
        # In-degree constraints
        for u in range(n):
            row = np.zeros(total_vars)
            for v in range(n):
                if u != v and adj[v][u]:
                    row[v*n + u] = 1.0
            A_rows.append(row)
            A_lbs.append(1.0)
            A_ubs.append(1.0)
        
        # MTZ constraints: y[u] - y[v] + n*x[u,v] <= n-1
        # for u != start, v != start, u != v
        for u in range(n):
            if u == start:
                continue
            for v in range(n):
                if v == start or v == u:
                    continue
                row = np.zeros(total_vars)
                row[num_x + u] = 1.0
                row[num_x + v] = -1.0
                row[u*n + v] = float(n)
                A_rows.append(row)
                A_lbs.append(-np.inf)
                A_ubs.append(float(n - 1))
        
        # Also need: y[u] - y[v] + n*x[v,u] <= n-1 (for the reverse direction)
        # Actually MTZ only needs one direction per pair, but let's be safe
        # The standard MTZ is: y[u] - y[v] + n*x[u,v] <= n-1 for all u,v != start
        # This prevents subtours.
        
        # Force x[u,v] = 0 for non-edges
        for u in range(n):
            for v in range(n):
                if u != v and not adj[u][v]:
                    row = np.zeros(total_vars)
                    row[u*n + v] = 1.0
                    A_rows.append(row)
                    A_lbs.append(0.0)
                    A_ubs.append(0.0)
        
        if not A_rows:
            return None
        
        A = np.array(A_rows)
        lbs = np.array(A_lbs)
        ubs = np.array(A_ubs)
        
        # Integer variables: x are binary, y are integer
        integrality = np.ones(total_vars, dtype=int)
        for u in range(n):
            for v in range(n):
                if u != v:
                    integrality[u*n + v] = 1  # binary
            integrality[num_x + u] = 1  # integer
        
        try:
            res = milp(
                c=c,
                constraints=LinearConstraint(A, lbs, ubs),
                integrality=integrality,
                bounds=Bounds(lb, ub),
                options={"time_limit": time_limit, "presolve": True}
            )
            
            if res.status != 0:
                return None
            
            # Extract solution
            x_sol = res.x[:num_x].reshape(n, n)
            cycle = [start]
            cur = start
            for _ in range(n - 1):
                # Find next vertex
                found = False
                for v in range(n):
                    if v != cur and v != start and x_sol[cur, v] > 0.5:
                        cycle.append(v)
                        cur = v
                        found = True
                        break
                if not found:
                    # Try including start as last
                    if x_sol[cur, start] > 0.5:
                        break
                    return None
            
            if len(cycle) != n:
                return None
            
            # Verify it's a valid Hamilton cycle
            if len(set(cycle)) != n:
                return None
            
            # Check all edges are in G_K
            for i in range(n):
                u = cycle[i]
                v = cycle[(i+1) % n]
                if idx_mat[u][v] > K:
                    return None
            
            return cycle
        
        except Exception:
            return None
    
    # Binary search for minimal K
    lo = N  # At least N edges needed
    hi = best_score
    
    while lo <= hi and time.time() < deadline - 15.0:
        mid = (lo + hi) // 2
        cycle = find_cycle_milp(mid, time_limit=5.0)
        
        if cycle is not None:
            sc = _cycle_score(cycle)
            if sc <= mid:
                best_cycle = cycle
                best_score = sc
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            lo = mid + 1
    
    # 2-opt polish to reduce max edge index
    def two_opt(cycle, time_limit):
        n = N
        improved = True
        deadline_local = time.time() + time_limit
        best = cycle[:]
        best_sc = _cycle_score(best)
        
        while improved and time.time() < deadline_local:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    best[i:j+1] = reversed(best[i:j+1])
                    new_sc = _cycle_score(best)
                    if new_sc < best_sc:
                        best_sc = new_sc
                        improved = True
                    else:
                        best[i:j+1] = reversed(best[i:j+1])
        
        return best, best_sc
    
    time_left = deadline - time.time() - 5.0
    if time_left > 1.0:
        best_cycle, best_score = two_opt(best_cycle, time_left)
    
    # Random restarts with 2-opt
    rng = random.Random(42)
    while time.time() < deadline - 2.0:
        perm = list(range(N))
        rng.shuffle(perm)
        improved_cycle, improved_score = two_opt(perm, 1.0)
        if improved_score < best_score:
            best_cycle = improved_cycle
            best_score = improved_score
    
    return best_cycle
