import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edge_order = P['edge_order']

# Precompute arrival index (0-based) for each edge
arrival = {}
edge_indices = []
for idx, (u, v) in enumerate(edge_order):
    key = (min(u, v), max(u, v))
    arrival[key] = idx
    edge_indices.append((u, v, idx))

def get_arrival(u, v):
    key = (min(u, v), max(u, v))
    return arrival[key]

def check_feasibility(K, time_limit=5.0):
    """
    Check if a Hamiltonian cycle exists using only edges with index <= K.
    Returns the cycle as a list of vertices if found, else None.
    """
    start_time = time.time()
    
    # Filter edges
    valid_edges = [(u, v) for u, v, idx in edge_indices if idx <= K]
    
    if len(valid_edges) < n:
        return None
        
    # Build adjacency for quick check
    adj = {i: set() for i in range(n)}
    for u, v in valid_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) < 2:
            return None
            
    # MILP Formulation
    # Variables:
    # x_{uv} for each edge in valid_edges (binary)
    # y_u for each vertex u (continuous, 0..n-1)
    
    # Map edge to variable index
    edge_var_idx = {}
    for i, (u, v) in enumerate(valid_edges):
        edge_var_idx[(u, v)] = i
        edge_var_idx[(v, u)] = i
        
    num_edges = len(valid_edges)
    num_vars = num_edges + n
    
    # Objective: Minimize 0 (feasibility)
    c = np.zeros(num_vars)
    
    # Bounds
    # x_{uv} in {0, 1}
    # y_u in [0, n-1]
    lb = np.zeros(num_vars)
    ub = np.ones(num_vars)
    # y vars are the last n vars
    lb[num_edges:] = 0
    ub[num_edges:] = n - 1
    
    # Integrality: x vars are integer (binary), y vars are continuous
    integrality = np.zeros(num_vars)
    integrality[:num_edges] = 1
    # integrality[num_edges:] = 0 (continuous)
    
    # Constraints
    A_list = []
    lb_list = []
    ub_list = []
    
    # 1. Degree constraints: sum_{v} x_{uv} = 2
    for u in range(n):
        row = np.zeros(num_vars)
        for v in range(n):
            if v > u:
                if (u, v) in edge_var_idx:
                    row[edge_var_idx[(u, v)]] = 1
            elif v < u:
                if (v, u) in edge_var_idx:
                    row[edge_var_idx[(v, u)]] = 1
        A_list.append(row)
        lb_list.append(2)
        ub_list.append(2)
        
    # 2. MTZ Subtour Elimination
    # y_u - y_v + n * x_{uv} <= n - 1 for all u != v
    # Note: This constraint is only needed if x_{uv} can be 1.
    # If edge (u,v) is not in valid_edges, x_{uv} is effectively 0, and the constraint
    # y_u - y_v <= n - 1 is always true given bounds.
    # So we only add constraints for valid edges.
    for u, v in valid_edges:
        if u == v: continue
        row = np.zeros(num_vars)
        row[edge_var_idx[(u, v)]] = n
        row[num_edges + u] = 1
        row[num_edges + v] = -1
        A_list.append(row)
        lb_list.append(-np.inf)
        ub_list.append(n - 1)
        
    # Fix y_0 = 0 to break symmetry
    row = np.zeros(num_vars)
    row[num_edges + 0] = 1
    A_list.append(row)
    lb_list.append(0)
    ub_list.append(0)
    
    A = np.array(A_list)
    lb_con = np.array(lb_list)
    ub_con = np.array(ub_list)
    
    constraints = LinearConstraint(A, lb_con, ub_con)
    bounds = Bounds(lb, ub)
    
    # Solve
    try:
        res = milp(c, integrality=integrality, constraints=constraints, bounds=bounds)
    except Exception:
        return None
        
    if time.time() - start_time > time_limit:
        return None
        
    if res.status != 0:
        return None
        
    # Extract cycle
    x = res.x[:num_edges]
    edges_in_cycle = []
    for i, (u, v) in enumerate(valid_edges):
        if x[i] > 0.5:
            edges_in_cycle.append((u, v))
            
    # Verify it's a single cycle
    if len(edges_in_cycle) != n:
        return None
        
    adj_cycle = {i: [] for i in range(n)}
    for u, v in edges_in_cycle:
        adj_cycle[u].append(v)
        adj_cycle[v].append(u)
        
    # Check connectivity and degree
    for i in range(n):
        if len(adj_cycle[i]) != 2:
            return None
            
    visited = [False] * n
    visited[0] = True
    path = [0]
    current = 0
    for _ in range(n-1):
        nexts = adj_cycle[current]
        # Choose the one not visited, unless it's the last one which must be 0
        next_node = None
        for nb in nexts:
            if not visited[nb]:
                next_node = nb
                break
        if next_node is None:
            # Must be the last step to 0
            if current != 0 and adj_cycle[current][0] == 0 or adj_cycle[current][1] == 0:
                 # Check if 0 is the only unvisited? No, 0 is visited.
                 # If we are at the last node, the only unvisited neighbor should be 0?
                 # Actually, if we've visited n-1 nodes, the current node is the last one.
                 # Its neighbors are the previous node and 0.
                 # If we didn't pick the unvisited one, it means both are visited.
                 # This implies we closed the loop early or got stuck.
                 return None
            return None
        visited[next_node] = True
        path.append(next_node)
        current = next_node
        
    # Check if path ends at a neighbor of 0
    if current not in adj_cycle[0]:
        return None
        
    return path

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    # Default cycle
    default_cycle = list(range(n))
    default_score = max(get_arrival(default_cycle[i], default_cycle[(i+1) % n]) + 1 for i in range(n))
    
    best_score = default_score
    best_cycle = default_cycle
    
    # Binary search for minimal K
    lo, hi = 24, 276
    
    # We know a cycle exists for hi=276 (complete graph)
    # Let's start searching
    
    while lo < hi and time.time() < deadline - 5:
        mid = (lo + hi) // 2
        cycle = check_feasibility(mid, time_limit=5.0)
        if cycle is not None:
            score = max(get_arrival(cycle[i], cycle[(i+1) % n]) + 1 for i in range(n))
            if score < best_score:
                best_score = score
                best_cycle = cycle
            hi = mid
        else:
            lo = mid + 1
            
    return best_cycle
# EVOLVE-BLOCK-END
