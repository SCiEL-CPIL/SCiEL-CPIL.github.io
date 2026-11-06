import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = 100.0 # Strictly under 120s limit
    
    n = 20
    m = n * (n - 1) // 2
    edge_to_idx = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_idx[(u, v)] = len(edge_to_idx)
    
    # Precompute for each vertex u, the list of edges incident to u
    # and for each pair of vertices (u, v), the list of vertices w
    # that are common neighbors of u and v (i.e., w is adjacent to both u and v)
    
    # For MILP formulation:
    # Variables: x_e for each edge e
    # Objective: max sum x_e
    # Constraints: for each pair (u, v), sum_{w in N(u) cap N(v)} x_{uw} x_{vw} <= 2
    # This is quadratic. We linearize using y_{u,v,w} = x_{uw} * x_{vw}
    # y_{u,v,w} <= x_{uw}, y_{u,v,w} <= x_{vw}, y_{u,v,w} >= x_{uw} + x_{vw} - 1
    # y_{u,v,w} >= 0
    
    # Number of y variables: for each pair (u,v), and each w != u,v
    # But we only need y for w such that (u,w) and (v,w) are potential edges.
    # Actually, for any triple (u, v, w) distinct, we might need y.
    # But we only constrain pairs (u,v). For a fixed pair (u,v), the common neighbors w are all other vertices.
    # So for each pair (u,v), we have n-2 potential common neighbors w.
    # Total y variables: C(n,2) * (n-2) = 190 * 18 = 3420.
    
    # Let's index y variables by (u, v, w) with u < v, w != u, w != v.
    y_to_idx = {}
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(n):
                if w == u or w == v:
                    continue
                y_to_idx[(u, v, w)] = len(y_to_idx)
    
    num_y = len(y_to_idx)
    total_vars = m + num_y
    
    # Objective: maximize sum x_e
    # c = [-1, ..., -1] for x, [0, ..., 0] for y
    c = np.zeros(total_vars)
    c[:m] = -1.0
    
    # Constraints:
    # 1. 0 <= x_e <= 1
    # 2. 0 <= y <= 1
    # 3. y_{u,v,w} <= x_{uw}
    # 4. y_{u,v,w} <= x_{vw}
    # 5. y_{u,v,w} >= x_{uw} + x_{vw} - 1
    # 6. sum_{w} y_{u,v,w} <= 2 for each pair (u,v)
    
    # We will build the constraint matrix A and bounds lb, ub.
    
    # Let's count the number of constraints first.
    # Type 1 & 2: 2 * total_vars (but we can use Bounds)
    # Type 3: num_y constraints
    # Type 4: num_y constraints
    # Type 5: num_y constraints
    # Type 6: C(n,2) constraints
    
    # Using Bounds for x and y:
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)
    
    # We will add the linear constraints for y linearization and the K2,3 constraint.
    
    A_rows = []
    A_lb = []
    A_ub = []
    
    # Helper to get x index
    def get_x_idx(u, v):
        if u > v: u, v = v, u
        return edge_to_idx[(u, v)]
    
    # Type 3: y_{u,v,w} - x_{uw} <= 0
    # Type 4: y_{u,v,w} - x_{vw} <= 0
    # Type 5: -y_{u,v,w} + x_{uw} + x_{vw} >= 1  =>  y_{u,v,w} - x_{uw} - x_{vw} <= -1
    
    # Type 6: sum_{w} y_{u,v,w} <= 2
    
    # Let's build the matrix row by row.
    
    # We'll use a list of lists for rows, then convert to array.
    # But for efficiency, we can build a sparse matrix or just a dense one if not too big.
    # 3420 y vars + 190 x vars = 3610 vars.
    # Constraints:
    # 3 * 3420 + 190 = 10260 + 190 = 10450 constraints.
    # Dense matrix 10450 x 3610 is about 37 million floats. 300 MB. Might be okay.
    # But SciPy MILP prefers sparse.
    
    from scipy.sparse import csr_matrix
    
    # We will construct the sparse matrix.
    
    # Let's iterate through all y variables and add the 3 linearization constraints.
    # Then iterate through all pairs (u,v) and add the sum constraint.
    
    # We'll store rows as lists of (col, val) and then build the matrix.
    # Or use scipy.sparse.lil_matrix or similar.
    
    # Let's use lil_matrix for construction.
    
    # Number of constraints
    num_constraints = 3 * num_y + (n * (n - 1) // 2)
    
    A = np.zeros((num_constraints, total_vars)) # Dense for simplicity first, might be slow but let's try.
    # Actually, 10450 * 3610 * 8 bytes = 301 MB. It's borderline.
    # Let's try sparse.
    
    # Using scipy.sparse
    A = np.zeros((num_constraints, total_vars)) # Let's stick to dense if it fits in memory and time.
    # If it's too slow, we can switch to sparse.
    
    lb_arr = np.zeros(num_constraints)
    ub_arr = np.zeros(num_constraints)
    
    row_idx = 0
    
    # Linearization constraints
    for (u, v, w), y_idx in y_to_idx.items():
        idx_uw = get_x_idx(u, w)
        idx_vw = get_x_idx(v, w)
        
        # y - x_uw <= 0
        A[row_idx, y_idx] = 1
        A[row_idx, idx_uw] = -1
        lb_arr[row_idx] = -np.inf
        ub_arr[row_idx] = 0
        row_idx += 1
        
        # y - x_vw <= 0
        A[row_idx, y_idx] = 1
        A[row_idx, idx_vw] = -1
        lb_arr[row_idx] = -np.inf
        ub_arr[row_idx] = 0
        row_idx += 1
        
        # y - x_uw - x_vw <= -1
        A[row_idx, y_idx] = 1
        A[row_idx, idx_uw] = -1
        A[row_idx, idx_vw] = -1
        lb_arr[row_idx] = -np.inf
        ub_arr[row_idx] = -1
        row_idx += 1
        
    # K2,3 constraints
    for u in range(n):
        for v in range(u + 1, n):
            # sum_{w != u, v} y_{u,v,w} <= 2
            for w in range(n):
                if w == u or w == v:
                    continue
                y_idx = y_to_idx[(u, v, w)]
                A[row_idx, y_idx] = 1
            lb_arr[row_idx] = -np.inf
            ub_arr[row_idx] = 2
            row_idx += 1
            
    # Ensure we didn't exceed row count
    assert row_idx == num_constraints
    
    # Integrality: x must be integer (0 or 1), y must be integer (0 or 1)
    integrality = np.ones(total_vars)
    
    # Solve MILP
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, lb_arr, ub_arr),
            integrality=integrality,
            bounds=Bounds(lb, ub),
            options={"time_limit": 80, "mip_rel_gap": 0.0}
        )
        
        if res.status == 0: # Optimal
            x_sol = res.x[:m]
            # Round to nearest integer
            x_sol = np.round(x_sol)
            
            # Construct graph
            best_adj = [0] * n
            for (u, v), idx in edge_to_idx.items():
                if x_sol[idx] > 0.5:
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
                    
            best_edges = sum(bin(x).count('1') for x in best_adj) // 2
            
            # Validate
            def is_valid(adj):
                for u in range(n):
                    for v in range(u + 1, n):
                        common = 0
                        for w in range(n):
                            if w == u or w == v: continue
                            if (adj[u] & (1 << w)) and (adj[v] & (1 << w)):
                                common += 1
                        if common >= 3:
                            return False
                return True
            
            if not is_valid(best_adj):
                # Fallback to heuristic if MILP fails or gives invalid result
                pass # Handled below
                
        else:
            # Fallback
            pass
            
    except Exception as e:
        # Fallback
        pass

    # Heuristic Polish / Fallback
    # If MILP didn't produce a valid result, or to improve it, we run a heuristic.
    # We use the MILP solution as a starting point if valid, otherwise random.
    
    adj = [0] * n
    common = [[0] * n for _ in range(n)]
    
    # Try to use MILP solution if it was valid
    if 'best_adj' in locals() and is_valid(best_adj):
        adj = best_adj[:]
        # Rebuild common matrix
        for u in range(n):
            for v in range(n):
                if u == v: continue
                for w in range(n):
                    if w == u or w == v: continue
                    if (adj[u] & (1 << w)) and (adj[v] & (1 << w)):
                        common[u][v] += 1
        current_edges = sum(bin(x).count('1') for x in adj) // 2
    else:
        # Random initialization
        random.seed(42)
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            # Simple greedy add
            if not (adj[u] & (1 << v)):
                # Check validity
                valid = True
                for x in range(n):
                    if x == u: continue
                    if adj[v] & (1 << x):
                        if common[u][x] >= 2:
                            valid = False
                            break
                if valid:
                    for y in range(n):
                        if y == v: continue
                        if adj[u] & (1 << y):
                            if common[v][y] >= 2:
                                valid = False
                                break
                if valid:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    for x in range(n):
                        if x == u: continue
                        if adj[v] & (1 << x):
                            common[u][x] += 1
                            common[x][u] += 1
                    for y in range(n):
                        if y == v: continue
                        if adj[u] & (1 << y):
                            common[v][y] += 1
                            common[y][v] += 1
        current_edges = sum(bin(x).count('1') for x in adj) // 2
        
    best_adj = adj[:]
    best_edges = current_edges
    
    # Local Search Polish
    def add_edge(u, v):
        nonlocal adj, common
        if u == v or (adj[u] & (1 << v)):
            return False
        for x in range(n):
            if x == u: continue
            if adj[v] & (1 << x):
                if common[u][x] >= 2:
                    return False
        for y in range(n):
            if y == v: continue
            if adj[u] & (1 << y):
                if common[v][y] >= 2:
                    return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        for x in range(n):
            if x == u: continue
            if adj[v] & (1 << x):
                common[u][x] += 1
                common[x][u] += 1
        for y in range(n):
            if y == v: continue
            if adj[u] & (1 << y):
                common[v][y] += 1
                common[y][v] += 1
        return True
        
    def remove_edge(u, v):
        nonlocal adj, common
        if not (adj[u] & (1 << v)):
            return
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        for x in range(n):
            if x == u: continue
            if adj[v] & (1 << x):
                common[u][x] -= 1
                common[x][u] -= 1
        for y in range(n):
            if y == v: continue
            if adj[u] & (1 << y):
                common[v][y] -= 1
                common[y][v] -= 1
                
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    while time.time() - start_time < deadline:
        improved = False
        random.shuffle(all_edges)
        for u, v in all_edges:
            if not (adj[u] & (1 << v)):
                if add_edge(u, v):
                    improved = True
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                    break
        if improved:
            continue
        
        # 2-swap
        found_swap = False
        for _ in range(100):
            e1, e2 = random.sample(all_edges, 2)
            u1, v1 = e1
            u2, v2 = e2
            if not (adj[u1] & (1 << v1)) or not (adj[u2] & (1 << v2)):
                continue
            if len({u1, v1, u2, v2}) < 4:
                continue
            new_u1, new_v1 = u1, u2
            new_u2, new_v2 = v1, v2
            if (adj[new_u1] & (1 << new_v1)) or (adj[new_u2] & (1 << new_v2)):
                continue
            old_adj = adj[:]
            old_common = [row[:] for row in common]
            remove_edge(u1, v1)
            remove_edge(u2, v2)
            ok1 = add_edge(new_u1, new_v1)
            ok2 = add_edge(new_u2, new_v2) if ok1 else False
            if ok1 and ok2:
                found_swap = True
                break
            else:
                adj = old_adj
                common = old_common
        if not found_swap:
            # Restart
            adj = [0] * n
            common = [[0] * n for _ in range(n)]
            random.shuffle(all_pairs)
            for u, v in all_pairs:
                if not (adj[u] & (1 << v)):
                    valid = True
                    for x in range(n):
                        if x == u: continue
                        if adj[v] & (1 << x):
                            if common[u][x] >= 2:
                                valid = False
                                break
                    if valid:
                        for y in range(n):
                            if y == v: continue
                            if adj[u] & (1 << y):
                                if common[v][y] >= 2:
                                    valid = False
                                    break
                    if valid:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        for x in range(n):
                            if x == u: continue
                            if adj[v] & (1 << x):
                                common[u][x] += 1
                                common[x][u] += 1
                        for y in range(n):
                            if y == v: continue
                            if adj[u] & (1 << y):
                                common[v][y] += 1
                                common[y][v] += 1
            current_edges = sum(bin(x).count('1') for x in adj) // 2
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj[:]

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
