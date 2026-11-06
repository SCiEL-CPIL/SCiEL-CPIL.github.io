"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We stop at 145s.
    deadline = start_time + 145.0
    
    n = 20
    
    # Variables:
    # x_e for each edge e in E_complete
    # We map edge (i, j) with i < j to index idx
    edges_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(edges_list)
    
    # Objective: Maximize sum(x_e)
    # min c^T x, so c = -1 for all variables
    c = np.full(m, -1.0)
    
    # Bounds: 0 <= x_e <= 1
    lb = np.zeros(m)
    ub = np.ones(m)
    integrality = np.ones(m)  # All variables are binary
    
    # Constraints:
    # For each pair of vertices (u, v), u < v:
    # sum_{w not in {u,v}} x_{u,w} * x_{v,w} <= 2
    # This is quadratic. We linearize it.
    # Let y_{u,v,w} = x_{u,w} * x_{v,w}
    # We need constraints:
    # y_{u,v,w} <= x_{u,w}
    # y_{u,v,w} <= x_{v,w}
    # y_{u,v,w} >= x_{u,w} + x_{v,w} - 1
    # y_{u,v,w} >= 0
    # And the main constraint: sum_w y_{u,v,w} <= 2
    
    # This creates a huge number of variables and constraints.
    # Number of pairs (u, v): n*(n-1)/2 = 190
    # For each pair, w ranges over n-2 = 18 vertices.
    # Total y variables: 190 * 18 = 3420
    # Total x variables: 190
    # Total variables: 3610
    
    # Constraints per pair (u, v):
    # 1. Sum constraint: sum_w y_{u,v,w} <= 2 (1 constraint)
    # 2. Linking constraints for each w:
    #    y_{u,v,w} - x_{u,w} <= 0
    #    y_{u,v,w} - x_{v,w} <= 0
    #    -y_{u,v,w} + x_{u,w} + x_{v,w} >= 1  =>  y_{u,v,w} - x_{u,w} - x_{v,w} <= -1
    #    So 3 constraints per w.
    # Total constraints per pair: 1 + 18*3 = 55
    # Total constraints: 190 * 55 = 10450
    
    # This is a very large MILP. SciPy's milp uses HiGHS.
    # 3610 variables and 10450 constraints might be solvable within 145s?
    # Let's try. If it's too slow, we might need to fall back to a heuristic 
    # or a simpler model. But the prompt asks for a MILP strategy.
    
    # Let's build the matrix A.
    # Row indices for constraints
    # Col indices for variables
    # x vars: 0 .. m-1
    # y vars: m .. m + num_y - 1
    
    num_pairs = 0
    pair_indices = []
    for u in range(n):
        for v in range(u + 1, n):
            pair_indices.append((u, v))
            num_pairs += 1
            
    num_y_vars_per_pair = n - 2
    total_y_vars = num_pairs * num_y_vars_per_pair
    total_vars = m + total_y_vars
    
    # Precompute indices for x variables
    x_idx = {}
    for idx, (i, j) in enumerate(edges_list):
        x_idx[(i, j)] = idx
        if i != j:
            x_idx[(j, i)] = idx # Symmetric access, though we only use i<j in construction
    
    # Build A matrix (sparse or dense? 10450 x 3610 is ~37M entries. Dense is ~300MB. Sparse is better.)
    # SciPy milp accepts dense or sparse. Let's use sparse (COO format) for efficiency.
    # Actually, constructing sparse matrix in Python loop is slow. 
    # Let's check if we can reduce the problem size or if HiGHS is fast enough.
    # 145 seconds is a lot.
    
    # Let's try to build the arrays for scipy.sparse
    row_indices = []
    col_indices = []
    values = []
    
    constraint_counter = 0
    
    for p_idx, (u, v) in enumerate(pair_indices):
        # Determine which w are not u or v
        w_list = [w for w in range(n) if w != u and w != v]
        
        # 1. Sum constraint: sum_y <= 2
        # Row: constraint_counter
        # Cols: y vars for this pair
        y_base_idx = m + p_idx * num_y_vars_per_pair
        for k, w in enumerate(w_list):
            row_indices.append(constraint_counter)
            col_indices.append(y_base_idx + k)
            values.append(1.0)
        
        # Lower bound for this row: -inf (default), Upper bound: 2
        # We need to store bounds for each row.
        
        constraint_counter += 1
        
        # 2. Linking constraints for each w
        for k, w in enumerate(w_list):
            y_var_idx = y_base_idx + k
            
            # Find x indices for (u, w) and (v, w)
            # Ensure u < w and v < w for lookup
            if u < w:
                x_uw_idx = x_idx[(u, w)]
            else:
                x_uw_idx = x_idx[(w, u)]
                
            if v < w:
                x_vw_idx = x_idx[(v, w)]
            else:
                x_vw_idx = x_idx[(w, v)]
            
            # Constraint A: y - x_uw <= 0
            row_indices.append(constraint_counter)
            col_indices.append(y_var_idx)
            values.append(1.0)
            row_indices.append(constraint_counter)
            col_indices.append(x_uw_idx)
            values.append(-1.0)
            constraint_counter += 1
            
            # Constraint B: y - x_vw <= 0
            row_indices.append(constraint_counter)
            col_indices.append(y_var_idx)
            values.append(1.0)
            row_indices.append(constraint_counter)
            col_indices.append(x_vw_idx)
            values.append(-1.0)
            constraint_counter += 1
            
            # Constraint C: y - x_uw - x_vw <= -1
            row_indices.append(constraint_counter)
            col_indices.append(y_var_idx)
            values.append(1.0)
            row_indices.append(constraint_counter)
            col_indices.append(x_uw_idx)
            values.append(-1.0)
            row_indices.append(constraint_counter)
            col_indices.append(x_vw_idx)
            values.append(-1.0)
            constraint_counter += 1
    
    total_constraints = constraint_counter
    
    # Build sparse matrix
    from scipy.sparse import coo_matrix
    A = coo_matrix((values, (row_indices, col_indices)), shape=(total_constraints, total_vars))
    
    # Bounds for constraints
    # Rows 0, 1+3*18, 2+... are the sum constraints (every 55 rows? No, 1 sum + 18*3 linking = 55 rows per pair)
    # Let's identify which rows are sum constraints.
    # For pair p, sum constraint is at row p * 55
    # Linking constraints are at rows p * 55 + 1 .. p * 55 + 54
    
    lb_con = np.full(total_constraints, -np.inf)
    ub_con = np.full(total_constraints, np.inf)
    
    for p in range(num_pairs):
        sum_row = p * 55
        ub_con[sum_row] = 2.0
        
        # Linking constraints A and B have upper bound 0
        for k in range(num_y_vars_per_pair):
            base_link = sum_row + 1 + k * 3
            ub_con[base_link] = 0.0
            ub_con[base_link + 1] = 0.0
            # Constraint C has upper bound -1
            ub_con[base_link + 2] = -1.0

    # Variable bounds
    lb_vars = np.zeros(total_vars)
    ub_vars = np.ones(total_vars)
    
    # Integrality
    integrality = np.ones(total_vars)
    
    # Objective
    c_full = np.zeros(total_vars)
    c_full[:m] = -1.0  # Maximize x
    
    # Solve
    try:
        res = milp(c=c_full,
                   constraints=LinearConstraint(A, lb_con, ub_con),
                   integrality=integrality,
                   bounds=Bounds(lb_vars, ub_vars))
        
        if res.status == 0:  # Optimal
            x_solution = res.x[:m]
            # Extract edges
            edges = []
            for idx, (u, v) in enumerate(edges_list):
                if x_solution[idx] > 0.5:
                    edges.append([u, v])
            return edges
        else:
            # If MILP fails or times out, fallback to a simple greedy?
            # The prompt says "solve the resulting Integer Linear Program".
            # If it fails, we should probably return a valid graph.
            # Let's implement a quick greedy fallback to ensure validity.
            pass
    except Exception:
        pass
    
    # Fallback: Greedy construction
    # This ensures we return *something* valid if MILP is too slow or fails.
    adj = np.zeros((n, n), dtype=bool)
    edges = []
    
    def is_valid_add(adj, u, v):
        common_uv = np.sum(adj[u] & adj[v])
        if common_uv > 2:
            return False
        neighbors_v = np.where(adj[v])[0]
        for w in neighbors_v:
            if w == u: continue
            if np.sum(adj[u] & adj[w]) > 1:
                return False
        neighbors_u = np.where(adj[u])[0]
        for w in neighbors_u:
            if w == v: continue
            if np.sum(adj[v] & adj[w]) > 1:
                return False
        return True

    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() > deadline - 1:
            break
        if is_valid_add(adj, u, v):
            adj[u, v] = True
            adj[v, u] = True
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
