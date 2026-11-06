"""Rewritten program for dense K2,3-free graphs on 20 vertices."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 20
    # Variables: x_{u,v} for 0 <= u < v < n
    num_vars = n * (n - 1) // 2
    
    # Map edge index to (u, v)
    edge_to_uv = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_uv.append((u, v))
            idx += 1
    
    # Objective: maximize sum of x_{u,v} -> minimize -sum
    c = -np.ones(num_vars)
    
    # Constraints: for every pair (u, v) of distinct vertices,
    # sum over w != u, v of (A_{u,w} * A_{v,w}) <= 2
    # But A_{u,w} * A_{v,w} is bilinear. We need to linearize.
    #
    # Alternative approach: For each triple (u, v, w) with u < v and w != u, v,
    # we can't directly express the product. Instead, let's think differently.
    #
    # The constraint "no K2,3" means: for every pair of vertices {u, v},
    # |N(u) ∩ N(v)| <= 2.
    #
    # This is a quadratic constraint on the edge variables. MILP requires linear
    # constraints. We can introduce auxiliary variables y_{u,v,w} for each
    # triple (u, v, w) where w is a common neighbor candidate, but that's complex.
    #
    # Let's use a different formulation. For each unordered pair {u, v} and each
    # vertex w distinct from u and v, define z_{u,v,w} = 1 iff both edges (u,w) and
    # (v,w) exist. Then sum_w z_{u,v,w} <= 2 for each pair {u,v}.
    #
    # z_{u,v,w} <= x_{u,w}, z_{u,v,w} <= x_{v,w}, z_{u,v,w} >= x_{u,w} + x_{v,w} - 1
    #
    # This is a valid linearization for binary variables.
    
    # Number of triples: for each pair (u,v) with u<v, and each w not in {u,v}
    # But z_{u,v,w} is symmetric in u,v, so we fix u<v.
    # For each pair (u,v) with u<v, w ranges over all vertices except u and v: n-2 values.
    
    # Let's index z variables:
    # z_idx for (u, v, w) where u < v and w != u, w != v
    z_to_idx = {}
    z_count = 0
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(n):
                if w != u and w != v:
                    z_to_idx[(u, v, w)] = z_count
                    z_count += 1
    
    # Total variables: num_vars (edges) + z_count (auxiliary)
    total_vars = num_vars + z_count
    
    # Objective: maximize edges, z variables don't contribute
    c_full = np.zeros(total_vars)
    c_full[:num_vars] = c  # -1 for edge variables
    
    # Build constraint matrix A and bounds
    # Constraints:
    # 1. For each (u, v, w): z_{u,v,w} - x_{u,w} <= 0
    # 2. For each (u, v, w): z_{u,v,w} - x_{v,w} <= 0
    # 3. For each (u, v, w): z_{u,v,w} - x_{u,w} - x_{v,w} >= -1
    # 4. For each pair (u, v): sum_{w} z_{u,v,w} <= 2
    
    num_constraints = 3 * z_count + n * (n - 1) // 2
    
    # Use sparse representation via lists
    rows = []
    cols = []
    vals = []
    
    def add_constraint(row_idx, col, val):
        rows.append(row_idx)
        cols.append(col)
        vals.append(val)
    
    constraint_bounds = []
    constraint_lb = []
    constraint_ub = []
    
    # Constraints 1-3 for each (u, v, w)
    for (u, v, w), z_idx in z_to_idx.items():
        # Find edge indices for (u,w) and (v,w)
        # Edge (a,b) where a < b
        if u < w:
            euw_idx = u * (n - 1) - u * (u + 1) // 2 + (w - u - 1)
        else:
            euw_idx = w * (n - 1) - w * (w + 1) // 2 + (u - w - 1)
        
        if v < w:
            evw_idx = v * (n - 1) - v * (v + 1) // 2 + (w - v - 1)
        else:
            evw_idx = w * (n - 1) - w * (w + 1) // 2 + (v - w - 1)
        
        # Constraint 1: z_{u,v,w} - x_{u,w} <= 0
        ci = len(constraint_lb)
        add_constraint(ci, num_vars + z_idx, 1)
        add_constraint(ci, euw_idx, -1)
        constraint_lb.append(0)
        constraint_ub.append(0)
        
        # Constraint 2: z_{u,v,w} - x_{v,w} <= 0
        ci = len(constraint_lb)
        add_constraint(ci, num_vars + z_idx, 1)
        add_constraint(ci, evw_idx, -1)
        constraint_lb.append(0)
        constraint_ub.append(0)
        
        # Constraint 3: z_{u,v,w} - x_{u,w} - x_{v,w} >= -1
        # i.e., -z_{u,v,w} + x_{u,w} + x_{v,w} <= 1
        ci = len(constraint_lb)
        add_constraint(ci, num_vars + z_idx, -1)
        add_constraint(ci, euw_idx, 1)
        add_constraint(ci, evw_idx, 1)
        constraint_lb.append(0)
        constraint_ub.append(1)
    
    # Constraint 4: for each pair (u, v) with u < v, sum_w z_{u,v,w} <= 2
    for u in range(n):
        for v in range(u + 1, n):
            ci = len(constraint_lb)
            for w in range(n):
                if w != u and w != v:
                    z_idx = z_to_idx[(u, v, w)]
                    add_constraint(ci, num_vars + z_idx, 1)
            constraint_lb.append(0)
            constraint_ub.append(2)
    
    A = (np.array(rows), np.array(cols), np.array(vals))
    
    # Variable bounds: all binary [0, 1]
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)
    
    # Integrality: all binary
    integrality = np.ones(total_vars)
    
    # Solve
    res = milp(c=c_full, constraints=LinearConstraint(A, lb=np.array(constraint_lb), ub=np.array(constraint_ub)),
               integrality=integrality, bounds=Bounds(lb, ub))
    
    if not res.success:
        # Fallback: return the seed solution
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges
    
    # Extract edges
    edges = []
    for i in range(num_vars):
        if res.x[i] > 0.5:
            u, v = edge_to_uv[i]
            edges.append([u, v])
    
    return edges
