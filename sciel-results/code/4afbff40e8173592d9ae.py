"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 20
    # Number of possible edges in a simple graph on n vertices
    m = n * (n - 1) // 2
    
    # Map edge index to (u, v)
    edge_list = []
    for u in range(n):
        for v in range(u + 1, n):
            edge_list.append((u, v))
    
    # For each pair of vertices (u, v) with u < v, we need to ensure
    # that the number of common neighbors is at most 2.
    # Let x_e be the indicator variable for edge e.
    # Common neighbors of u and v are vertices w such that (u,w) and (v,w) are edges.
    # We need sum_{w in N(u) \cap N(v)} 1 <= 2.
    # Note: w cannot be u or v.
    
    # Build constraint matrix A for MILP
    # Each constraint is: sum_{e in E(u,v)} x_e <= 2
    # where E(u,v) is the set of edges (u,w) and (v,w) for w != u,v? No.
    # The common neighbors are w such that edge (u,w) exists AND edge (v,w) exists.
    # This is a product of variables, which is not linear.
    
    # Wait, the constraint is on the NUMBER of common neighbors.
    # Let c_{uv} = number of common neighbors of u and v.
    # c_{uv} = sum_{w != u,v} [x_{uw} * x_{vw}]
    # This is quadratic. MILP requires linear constraints.
    
    # We need a different formulation or a heuristic.
    # The directive says "solve the resulting Integer Linear Program using scipy.optimize.milp".
    # But the natural formulation is quadratic.
    
    # Alternative: Maybe the directive implies a relaxation or a specific linearization?
    # Or maybe I should use a different approach that fits in MILP?
    
    # Actually, K2,3-free means no two vertices have 3 or more common neighbors.
    # This is equivalent to: for any u, v, |N(u) ∩ N(v)| <= 2.
    
    # Since the problem size is small (20 vertices, 190 edges), we could try to solve a QP or use a heuristic.
    # But the directive specifically asks for MILP.
    
    # Perhaps we can use the fact that for dense graphs, we can start from a known dense graph and verify/improve?
    # Or maybe we can formulate it as: maximize sum x_e subject to for all u<v, sum_{w} (x_{uw} + x_{vw} - 2*x_{uw}*x_{vw}) ... no.
    
    # Let's reconsider. Is there a linear way to express "at most 2 common neighbors"?
    # No, because it involves products of edge variables.
    
    # However, if we fix the degrees or use a specific structure, maybe?
    # Or maybe we can use a heuristic search guided by MILP relaxations?
    
    # Given the time limit (120s) and the requirement to use MILP, perhaps the intended solution is to
    # use a local search or greedy algorithm, but the directive is specific.
    
    # Let's try a different interpretation: Maybe we can use MILP to solve a subproblem or to verify a candidate?
    # Or maybe we can use a linear approximation of the quadratic term?
    
    # Actually, let's look at the current seed: 5 copies of K4.
    # Edges = 5 * 6 = 30.
    # Can we do better?
    # Ex(20, K2,3) is known to be larger than 30.
    # For n=20, a random graph with p=0.5 has ~100 edges, but likely contains K2,3.
    # The extremal number ex(n, K2,3) is O(n^{3/2}). For n=20, sqrt(20)*20 ≈ 89.
    # So we should be able to get close to 80-90 edges.
    
    # Since direct MILP is not straightforward due to quadratic constraints, I will implement a
    # heuristic that is likely to find a dense K2,3-free graph, and then perhaps use MILP
    # in a local improvement step if possible, or just use a strong heuristic.
    
    # But the directive says "solve the resulting Integer Linear Program using scipy.optimize.milp".
    # This suggests there IS a linear formulation.
    
    # Wait, if we define y_{uvw} = x_{uw} * x_{vw}, then the constraint is sum_w y_{uvw} <= 2.
    # And we have constraints y_{uvw} <= x_{uw}, y_{uvw} <= x_{vw}, y_{uvw} >= x_{uw} + x_{vw} - 1.
    # This is a standard linearization of the product of two binary variables!
    # So we can add variables y_{uvw} for each triple (u,v,w) with u<v, w not in {u,v}.
    # Number of triples: C(20,2) * 18 = 190 * 18 = 3420 variables for y.
    # Plus 190 variables for x.
    # Total variables: 3610.
    # Constraints:
    # 1. For each u<v: sum_{w != u,v} y_{uvw} <= 2. (190 constraints)
    # 2. For each u<v, w not in {u,v}:
    #    y_{uvw} <= x_{uw}
    #    y_{uvw} <= x_{vw}
    #    y_{uvw} >= x_{uw} + x_{vw} - 1
    #    (3 * 3420 = 10260 constraints)
    # 3. 0 <= x_e <= 1, 0 <= y_{uvw} <= 1.
    
    # This is a valid MILP formulation!
    # The objective is to maximize sum x_e.
    
    # Let's implement this.
    
    # Variables:
    # x[0..189] for edges
    # y[0..3419] for triples (u, v, w) where u < v and w not in {u, v}
    
    # We need to map (u, v, w) to an index for y.
    # And (u, w) to an index for x.
    
    # Edge index mapping:
    def edge_index(u, v):
        if u > v:
            u, v = v, u
        # u < v
        # Number of edges before u: u*(n-1) - u*(u-1)/2 ... no
        # Simpler: use a dictionary or a 2D array.
        pass
    
    # Let's use a 2D array for edge indices.
    edge_idx = [[-1] * n for _ in range(n)]
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_idx[u][v] = idx
            edge_idx[v][u] = idx
            idx += 1
    
    # Triple index mapping:
    # We iterate over all pairs (u, v) with u < v, and for each w not in {u, v}.
    triple_idx = {}
    y_count = 0
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(n):
                if w == u or w == v:
                    continue
                triple_idx[(u, v, w)] = y_count
                y_count += 1
    
    total_vars = m + y_count
    
    # Objective: maximize sum x_e -> minimize -sum x_e
    c = np.zeros(total_vars)
    for i in range(m):
        c[i] = -1.0
    
    # Constraints:
    # We will build the constraint matrix A and bounds lb, ub.
    # A is a list of linear expressions.
    
    # Since the matrix is sparse and large, we should use scipy.sparse.
    # But milp accepts a dense array or a sparse array?
    # scipy.optimize.milp accepts a sparse array for A.
    
    from scipy.sparse import csr_matrix
    
    # Let's build the rows of A.
    # Row 1: For each pair (u, v), sum_{w} y_{uvw} <= 2
    # Row 2-4: For each triple (u, v, w), the linearization constraints.
    
    # Number of constraint rows:
    num_rows = 0
    # 190 rows for the common neighbor constraint
    num_rows += n * (n - 1) // 2
    # 3 * y_count rows for the linearization
    num_rows += 3 * y_count
    
    # Initialize sparse matrix
    rows = []
    cols = []
    vals = []
    
    # Constraint type 1: sum_{w} y_{uvw} <= 2 for each u < v
    row_idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(n):
                if w == u or w == v:
                    continue
                y_var_idx = m + triple_idx[(u, v, w)]
                rows.append(row_idx)
                cols.append(y_var_idx)
                vals.append(1.0)
            row_idx += 1
    
    # Constraint type 2: y_{uvw} <= x_{uw}
    # y_{uvw} - x_{uw} <= 0
    # Constraint type 3: y_{uvw} <= x_{vw}
    # y_{uvw} - x_{vw} <= 0
    # Constraint type 4: y_{uvw} >= x_{uw} + x_{vw} - 1
    # -y_{uvw} + x_{uw} + x_{vw} >= -1  =>  y_{uvw} - x_{uw} - x_{vw} <= 1
    
    for (u, v, w), y_i in triple_idx.items():
        y_var_idx = m + y_i
        x_uw_idx = edge_idx[u][w]
        x_vw_idx = edge_idx[v][w]
        
        # y <= x_uw
        rows.append(row_idx); cols.append(y_var_idx); vals.append(1.0)
        rows.append(row_idx); cols.append(x_uw_idx); vals.append(-1.0)
        row_idx += 1
        
        # y <= x_vw
        rows.append(row_idx); cols.append(y_var_idx); vals.append(1.0)
        rows.append(row_idx); cols.append(x_vw_idx); vals.append(-1.0)
        row_idx += 1
        
        # y - x_uw - x_vw <= 1
        rows.append(row_idx); cols.append(y_var_idx); vals.append(1.0)
        rows.append(row_idx); cols.append(x_uw_idx); vals.append(-1.0)
        rows.append(row_idx); cols.append(x_vw_idx); vals.append(-1.0)
        row_idx += 1
    
    A = csr_matrix((vals, (rows, cols)), shape=(num_rows, total_vars))
    
    # Bounds for variables
    # x: [0, 1]
    # y: [0, 1]
    integrality = np.ones(total_vars)
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)
    
    # Bounds for constraints
    # Type 1: <= 2
    # Type 2-4: <= 0, <= 0, <= 1
    cons_lb = np.full(num_rows, -np.inf)
    cons_ub = np.zeros(num_rows)
    
    # First 190 constraints: ub = 2
    num_pairs = n * (n - 1) // 2
    cons_ub[:num_pairs] = 2.0
    
    # Remaining constraints:
    # They come in groups of 3 for each triple.
    # The pattern is:
    # Row i (relative to start of linearization): y - x_uw <= 0 -> ub=0
    # Row i+1: y - x_vw <= 0 -> ub=0
    # Row i+2: y - x_uw - x_vw <= 1 -> ub=1
    
    start_lin = num_pairs
    for i in range(y_count):
        base = start_lin + 3 * i
        cons_ub[base] = 0.0
        cons_ub[base + 1] = 0.0
        cons_ub[base + 2] = 1.0
    
    linear_constraint = LinearConstraint(A, cons_lb, cons_ub)
    bounds = Bounds(lb, ub)
    
    # Solve
    res = milp(c, constraints=linear_constraint, integrality=integrality, bounds=bounds)
    
    if res.status != 0:
        # Fallback to seed if MILP fails
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges
    
    # Extract solution
    edges = []
    for i in range(m):
        if res.x[i] > 0.5:
            u, v = edge_list[i]
            edges.append([u, v])
    
    return edges

# EVOLVE-BLOCK-END
