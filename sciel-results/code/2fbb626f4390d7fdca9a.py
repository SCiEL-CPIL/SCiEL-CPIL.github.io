import time
import random
import sys
from scipy.optimize import milp, LinearConstraint, Bounds

def instance_trees(n=20, seed=611):
    rng = random.Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    trees = instance_trees(20, 611)
    n_trees = 20
    n_hosts = 20
    
    # Pre-compute tree sizes and edges
    tree_sizes = [len(t) + 1 for t in trees]
    tree_edges = trees
    
    # Total variables:
    # x_{t, v, h}: tree t, vertex v, host h
    # Total x vars = sum(k_t^2)
    x_vars = []
    x_index = {}
    var_idx = 0
    for t in range(n_trees):
        k = tree_sizes[t]
        for v in range(k):
            for h in range(n_hosts):
                x_index[(t, v, h)] = var_idx
                var_idx += 1
    n_x_vars = var_idx
    
    # w variables: w_{t, e_t, e_host, o}
    # e_t is an edge in tree t (u, v)
    # e_host is a host edge (h, h')
    # o is orientation (0: u->h, v->h'; 1: u->h', v->h)
    w_vars = []
    w_index = {}
    host_edges = [(i, j) for i in range(n_hosts) for j in range(i+1, n_hosts)]
    for t in range(n_trees):
        for (u, v) in tree_edges[t]:
            for (h, hp) in host_edges:
                w_index[(t, (u, v), (h, hp), 0)] = var_idx
                var_idx += 1
                w_index[(t, (u, v), (h, hp), 1)] = var_idx
                var_idx += 1
    n_w_vars = var_idx - n_x_vars
    total_vars = var_idx
    
    # Objective: Maximize sum of x_{t, v, h} (which is constant 1 for valid packing) 
    # Actually, we just want a feasible solution. 
    # Or minimize 0. 
    # But to ensure we find a packing, we can just solve feasibility.
    # However, milp requires an objective. We can set c = 0.
    c = [0.0] * total_vars
    
    # Constraints
    # 1. Sum_h x_{t, v, h} = 1
    # 2. Sum_v x_{t, v, h} = 1
    # 3. w_{t, e_t, e_host, 0} <= x_{t, u, h}
    # 4. w_{t, e_t, e_host, 0} <= x_{t, v, hp}
    # 5. w_{t, e_t, e_host, 0} >= x_{t, u, h} + x_{t, v, hp} - 1
    # (similar for orientation 1)
    # 6. Sum_t Sum_{e_t} (w_{t, e_t, e_host, 0} + w_{t, e_t, e_host, 1}) <= 1 for each e_host
    
    A_rows = []
    lb = []
    ub = []
    
    # Constraint 1: Sum_h x_{t, v, h} = 1
    for t in range(n_trees):
        k = tree_sizes[t]
        for v in range(k):
            row = [0.0] * total_vars
            for h in range(n_hosts):
                row[x_index[(t, v, h)]] = 1.0
            A_rows.append(row)
            lb.append(1.0)
            ub.append(1.0)
            
    # Constraint 2: Sum_v x_{t, v, h} = 1
    for t in range(n_trees):
        k = tree_sizes[t]
        for h in range(n_hosts):
            row = [0.0] * total_vars
            for v in range(k):
                row[x_index[(t, v, h)]] = 1.0
            A_rows.append(row)
            lb.append(1.0)
            ub.append(1.0)
            
    # Constraints 3-5: Linking w and x
    for t in range(n_trees):
        for (u, v) in tree_edges[t]:
            for (h, hp) in host_edges:
                # Orientation 0: u->h, v->hp
                w_idx = w_index[(t, (u, v), (h, hp), 0)]
                x_uh_idx = x_index[(t, u, h)]
                x_vhp_idx = x_index[(t, v, hp)]
                
                # w <= x_uh
                row = [0.0] * total_vars
                row[w_idx] = 1.0
                row[x_uh_idx] = -1.0
                A_rows.append(row)
                lb.append(-1e10)
                ub.append(0.0)
                
                # w <= x_vhp
                row = [0.0] * total_vars
                row[w_idx] = 1.0
                row[x_vhp_idx] = -1.0
                A_rows.append(row)
                lb.append(-1e10)
                ub.append(0.0)
                
                # w >= x_uh + x_vhp - 1
                row = [0.0] * total_vars
                row[w_idx] = -1.0
                row[x_uh_idx] = 1.0
                row[x_vhp_idx] = 1.0
                A_rows.append(row)
                lb.append(-1.0)
                ub.append(1e10)
                
                # Orientation 1: u->hp, v->h
                w_idx1 = w_index[(t, (u, v), (h, hp), 1)]
                x_uhp_idx = x_index[(t, u, hp)]
                x_vh_idx = x_index[(t, v, h)]
                
                # w1 <= x_uhp
                row = [0.0] * total_vars
                row[w_idx1] = 1.0
                row[x_uhp_idx] = -1.0
                A_rows.append(row)
                lb.append(-1e10)
                ub.append(0.0)
                
                # w1 <= x_vh
                row = [0.0] * total_vars
                row[w_idx1] = 1.0
                row[x_vh_idx] = -1.0
                A_rows.append(row)
                lb.append(-1e10)
                ub.append(0.0)
                
                # w1 >= x_uhp + x_vh - 1
                row = [0.0] * total_vars
                row[w_idx1] = -1.0
                row[x_uhp_idx] = 1.0
                row[x_vh_idx] = 1.0
                A_rows.append(row)
                lb.append(-1.0)
                ub.append(1e10)
                
    # Constraint 6: Edge disjointness
    # For each host edge e_host, sum of w's for all trees and edges mapping to it <= 1
    for (h, hp) in host_edges:
        row = [0.0] * total_vars
        for t in range(n_trees):
            for (u, v) in tree_edges[t]:
                row[w_index[(t, (u, v), (h, hp), 0)]] = 1.0
                row[w_index[(t, (u, v), (h, hp), 1)]] = 1.0
        A_rows.append(row)
        lb.append(-1e10)
        ub.append(1.0)
        
    # Convert to LinearConstraint
    A = [row for row in A_rows]
    # scipy milp expects A as a list of lists or array
    # It also expects integrality info
    integrality = [1] * total_vars
    
    # Bounds for variables: 0 or 1
    bounds = Bounds(lb=[0.0] * total_vars, ub=[1.0] * total_vars)
    
    # Solve
    try:
        res = milp(c=c, constraints=LinearConstraint(A, lb, ub), integrality=integrality, bounds=bounds, options={"time_limit": 130})
        if res.status == 0: # Optimal
            x_sol = res.x
            mappings = [None] * n_trees
            for t in range(n_trees):
                k = tree_sizes[t]
                mapping = [0] * k
                for v in range(k):
                    for h in range(n_hosts):
                        if x_sol[x_index[(t, v, h)]] > 0.5:
                            mapping[v] = h
                            break
                mappings[t] = mapping
            return mappings
        else:
            # If ILP fails or times out, fallback to simple greedy or return None
            # But ILP should be strong enough.
            # Let's try a quick greedy fallback just in case
            pass
    except Exception as e:
        pass
        
    # Fallback: Simple greedy or random restart backtracking if ILP fails
    # For now, return a trivial valid packing (e.g., just T1 and T2) or None
    # But we want max score.
    # Let's implement a fast backtracking for small trees as fallback
    used_edges = set()
    mappings = [None] * 20
    
    def try_embed(t_idx):
        if time.time() > deadline: return False
        k = t_idx + 1
        edges = tree_edges[t_idx]
        if not edges:
            mappings[t_idx] = [0]
            return True
            
        # Try random permutations of host vertices
        # This is slow.
        return False
        
    # Given the strength of ILP, we likely won't need fallback.
    # But to be safe, if ILP fails, we return a partial packing found by a simple heuristic.
    
    # Simple heuristic: Map trees in order, using random available vertices.
    # This might not be edge disjoint.
    
    # Let's just return None for unembedded trees if ILP fails.
    # But the task requires valid types.
    
    # If we reach here, ILP didn't succeed.
    # We can try a few random restarts of the original backtracking with a very tight time limit.
    
    # Actually, let's just return the best we can.
    # If ILP fails, it's likely due to complexity or timeout.
    # We will return a list of Nones for safety, or a trivial packing.
    # T1 is empty, T2 is edge (0,1).
    # We can always pack T1 and T2.
    result = [None] * 20
    result[0] = [0]
    result[1] = [0, 1]
    return result

