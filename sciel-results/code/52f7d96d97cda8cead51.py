import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n = 20
    # Number of possible edges
    m = n * (n - 1) // 2
    
    # Map edge index to (u, v)
    edge_map = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_map.append((i, j))
            idx += 1
    
    # Variable x_e for each edge e, x_e in {0, 1}
    # Objective: maximize sum(x_e)
    c = -np.ones(m)  # Minimize negative sum
    
    # Constraints: For every pair of distinct vertices (u, v), 
    # the number of common neighbors <= 2.
    # Common neighbors of u and v are vertices w such that (u,w) and (v,w) are edges.
    # Note: The edge (u,v) itself is not a common neighbor.
    # So for each pair (u, v), sum_{w != u, v} x_{u,w} * x_{v,w} <= 2?
    # No, that's quadratic.
    # The constraint is: |N(u) ∩ N(v)| <= 2.
    # This is equivalent to: for any distinct u, v, the number of vertices w 
    # such that (u,w) in E and (v,w) in E is at most 2.
    
    # This is a quadratic constraint. MILP requires linear constraints.
    # However, we can reformulate.
    # Let y_{u,v,w} indicate if w is a common neighbor of u and v.
    # y_{u,v,w} = 1 iff x_{u,w}=1 and x_{v,w}=1.
    # This is equivalent to:
    # y_{u,v,w} <= x_{u,w}
    # y_{u,v,w} <= x_{v,w}
    # y_{u,v,w} >= x_{u,w} + x_{v,w} - 1
    # And sum_{w != u, v} y_{u,v,w} <= 2.
    
    # This introduces a huge number of variables: O(n^3).
    # n=20 -> 20*19*18/6 = 1140 triples (u,v,w) with u<v<w? No, u,v distinct, w distinct from both.
    # Actually, for each pair (u,v), we sum over w.
    # Total pairs (u,v) is 190.
    # For each pair, ~18 w's.
    # Total y variables ~ 190 * 18 = 3420.
    # Plus 190 x variables.
    # Total ~ 3610 variables.
    # Constraints:
    # 3 per y variable: ~ 10260 constraints.
    # 1 per pair (u,v): 190 constraints.
    # Total ~ 10450 constraints.
    
    # This is a very large MILP for a 150s timeout. 
    # SciPy's milp uses HiGHS. It might be too slow.
    # Let's check if there's a simpler way or if we should just try it.
    # If it times out, we fall back to a heuristic.
    
    # Alternative: 
    # The problem is small enough that a clever heuristic might be better, 
    # but the directive asks for MILP.
    # Let's try to construct the MILP.
    
    # Variables:
    # x[0..m-1]
    # y[k] for each triple (u, v, w) where u < v and w not in {u, v}.
    # We need to map (u, v, w) to an index.
    
    # Actually, we can be smarter.
    # The constraint is on common neighbors.
    # For a fixed pair (u, v), let S_{u,v} be the set of w != u, v.
    # We need sum_{w in S_{u,v}} x_{u,w} x_{v,w} <= 2.
    
    # Let's build the matrix A for the constraints.
    # This might be memory intensive if done naively, but 10k constraints x 3.6k vars is ~ 36M entries.
    # Sparse matrix should handle it.
    
    # Let's define the mapping for y variables.
    # y_index[u][v][w]
    # We only need y for w that could be a common neighbor.
    
    # Let's list all valid (u, v, w) triples where u < v and w not in {u, v}.
    # Note: The constraint for pair (u, v) is symmetric to (v, u).
    # We can just iterate u < v.
    
    y_triples = []
    y_index_map = {}
    y_idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(n):
                if w != u and w != v:
                    y_index_map[(u, v, w)] = y_idx
                    y_triples.append((u, v, w))
                    y_idx += 1
    
    num_y = len(y_triples)
    num_vars = m + num_y
    
    # Objective: maximize sum(x)
    # c[0..m-1] = -1
    # c[m..] = 0
    c = np.zeros(num_vars)
    c[:m] = -1
    
    # Constraints:
    # 1. y_{u,v,w} <= x_{u,w}
    # 2. y_{u,v,w} <= x_{v,w}
    # 3. y_{u,v,w} >= x_{u,w} + x_{v,w} - 1  =>  -x_{u,w} - x_{v,w} + y_{u,v,w} >= -1
    # 4. sum_{w} y_{u,v,w} <= 2 for each (u,v)
    
    # We need to build sparse matrix A.
    # Rows correspond to constraints.
    # Let's count constraints.
    # For each y variable (u,v,w):
    #   Row A: y <= x_uw  =>  -x_uw + y <= 0
    #   Row B: y <= x_vw  =>  -x_vw + y <= 0
    #   Row C: y >= x_uw + x_vw - 1 => -x_uw - x_vw + y >= -1 => x_uw + x_vw - y <= 1
    # So 3 constraints per y.
    # For each pair (u,v):
    #   sum_{w} y_{u,v,w} <= 2
    
    num_constraints = 3 * num_y + (n * (n - 1) // 2)
    
    # Use scipy.sparse
    from scipy.sparse import lil_matrix
    
    A = lil_matrix((num_constraints, num_vars), dtype=float)
    lb = np.zeros(num_constraints)
    ub = np.zeros(num_constraints)
    
    row_idx = 0
    
    # Constraints 1, 2, 3 for each y
    for (u, v, w), y_i in y_index_map.items():
        var_y = m + y_i
        
        # Find x indices for (u, w) and (v, w)
        # Need to handle ordering u<w or w<u
        def get_x_idx(i, j):
            if i > j:
                i, j = j, i
            # Calculate index
            # This is O(n) if we do it naively. 
            # Precompute x_index_map
            pass
        
    # Precompute x_index_map
    x_index_map = {}
    for i, (u, v) in enumerate(edge_map):
        x_index_map[(u, v)] = i
        x_index_map[(v, u)] = i
        
    for (u, v, w), y_i in y_index_map.items():
        var_y = m + y_i
        x_uw = x_index_map[(u, w)]
        x_vw = x_index_map[(v, w)]
        
        # Constraint 1: -x_uw + y <= 0
        A[row_idx, x_uw] = -1
        A[row_idx, var_y] = 1
        lb[row_idx] = -np.inf
        ub[row_idx] = 0
        row_idx += 1
        
        # Constraint 2: -x_vw + y <= 0
        A[row_idx, x_vw] = -1
        A[row_idx, var_y] = 1
        lb[row_idx] = -np.inf
        ub[row_idx] = 0
        row_idx += 1
        
        # Constraint 3: x_uw + x_vw - y <= 1
        A[row_idx, x_uw] = 1
        A[row_idx, x_vw] = 1
        A[row_idx, var_y] = -1
        lb[row_idx] = -np.inf
        ub[row_idx] = 1
        row_idx += 1
        
    # Constraints 4: sum_{w} y_{u,v,w} <= 2
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(n):
                if w != u and w != v:
                    y_i = y_index_map[(u, v, w)]
                    var_y = m + y_i
                    A[row_idx, var_y] = 1
            lb[row_idx] = -np.inf
            ub[row_idx] = 2
            row_idx += 1
            
    A = A.tocsr()
    
    # Integrality: x are integers (binary), y are integers (binary)
    integrality = np.ones(num_vars)
    
    # Bounds: 0 <= x, y <= 1
    bounds = Bounds(lb=np.zeros(num_vars), ub=np.ones(num_vars))
    
    # Solve
    try:
        res = milp(c=c, constraints=LinearConstraint(A, lb, ub), integrality=integrality, bounds=bounds, options={"time_limit": 140})
        if res.status == 0 or res.status == 1:  # Optimal or Local Optimal
            x_sol = res.x[:m]
            # Round to nearest integer
            x_sol = np.round(x_sol).astype(int)
            
            # Construct edge list
            edges = []
            for i, (u, v) in enumerate(edge_map):
                if x_sol[i] == 1:
                    edges.append([u, v])
            
            # Verify validity
            adj = [0] * n
            for u, v in edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    cn = adj[i] & adj[j]
                    if bin(cn).count('1') > 2:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return edges
            else:
                # If invalid due to numerical issues, try to fix or return empty
                # But usually MILP should be exact if bounds are tight.
                # If it fails, we fall through to heuristic
                pass
    except Exception as e:
        pass
        
    # Fallback: Heuristic
    # Since MILP might be too slow or fail, we use a fast heuristic as fallback.
    # This ensures we return something valid.
    
    def init_adj():
        return [0] * n
    
    def common_count(adj, u, v):
        return bin(adj[u] & adj[v]).count('1')
        
    def is_valid_add(adj, u, v):
        if common_count(adj, u, v) > 2:
            return False
        nv = adj[v]
        while nv:
            low = nv & (-nv)
            w = low.bit_length() - 1
            if w != u:
                if common_count(adj, u, w) > 1:
                    return False
            nv ^= low
        nu = adj[u]
        while nu:
            low = nu & (-nu)
            w = low.bit_length() - 1
            if w != v:
                if common_count(adj, v, w) > 1:
                    return False
            nu ^= low
        return True
        
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    best_adj = init_adj()
    best_count = 0
    
    # Quick greedy
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_pairs)
    adj = init_adj()
    for u, v in all_pairs:
        if is_valid_add(adj, u, v):
            add_edge(adj, u, v)
            
    count = sum(bin(a).count('1') for a in adj) // 2
    if count > best_count:
        best_count = count
        best_adj = adj[:]
        
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
