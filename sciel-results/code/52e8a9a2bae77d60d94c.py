import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 24
    rng = random.Random(614)
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((i, j))
    rng.shuffle(all_edges)
    
    edge_pos = {}
    for idx, (u, v) in enumerate(all_edges, 1):
        edge_pos[(u, v)] = idx
        edge_pos[(v, u)] = idx

    # Pre-compute indices for all directed pairs
    # We will use a matrix or list of lists for x variables
    # Number of directed edges is n*(n-1)
    # Let's index x[i][j] by i*n + j
    
    def find_hamiltonian_cycle_ilp(K):
        """
        Returns a list of vertices representing the cycle (starting with 0) or None.
        """
        # Determine which undirected edges are present
        # present[i][j] = True if edge (i,j) has pos <= K
        # We can just check on the fly or build a set
        valid_edges = set()
        for (u, v) in all_edges:
            if edge_pos[(u, v)] <= K:
                valid_edges.add((u, v))
                valid_edges.add((v, u))
        
        # Variables:
        # x[i*n + j] for i != j
        # u[i] for i in 0..n-1
        # Total vars: n*(n-1) + n
        
        num_x = n * (n - 1)
        num_u = n
        num_vars = num_x + num_u
        
        # Objective: Minimize sum of x_ij for i < j (each undirected edge counted once)
        # Actually, we can just minimize sum of all x_ij / 2 or sum of x_ij for i<j.
        # Since sum x_ij for i!=j will be 2*n if a cycle exists.
        # Let's minimize sum_{i<j} (x_{ij} + x_{ji}).
        c = np.zeros(num_vars)
        for i in range(n):
            for j in range(i+1, n):
                idx_ij = i * n + j
                idx_ji = j * n + i
                c[idx_ij] = 1
                c[idx_ji] = 1
                # u variables have 0 cost
        # We want to minimize this. If a cycle exists, min value is n.
        
        # Constraints
        # 1. Degree constraints
        # For each i: sum_{j!=i} x_{ij} = 1
        # For each i: sum_{j!=i} x_{ji} = 1
        A_rows = []
        b_lower = []
        b_upper = []
        
        # Out-degree
        for i in range(n):
            row = np.zeros(num_vars)
            for j in range(n):
                if i != j:
                    row[i * n + j] = 1
            A_rows.append(row)
            b_lower.append(1)
            b_upper.append(1)
            
        # In-degree
        for i in range(n):
            row = np.zeros(num_vars)
            for j in range(n):
                if i != j:
                    row[j * n + i] = 1
            A_rows.append(row)
            b_lower.append(1)
            b_upper.append(1)
            
        # 2. MTZ Constraints
        # u_i - u_j + n * x_{ij} <= n - 1 for all i != j
        # Rearranged: u_i - u_j + n * x_{ij} <= n - 1
        # Note: u variables are at index num_x + i
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                row = np.zeros(num_vars)
                row[num_x + i] = 1
                row[num_x + j] = -1
                row[i * n + j] = n
                A_rows.append(row)
                b_lower.append(-np.inf)
                b_upper.append(n - 1)
                
        # 3. u_0 = 0
        row = np.zeros(num_vars)
        row[num_x + 0] = 1
        A_rows.append(row)
        b_lower.append(0)
        b_upper.append(0)
        
        # 4. Bounds for u_i
        # 0 <= u_i <= n-1 for i != 0
        # u_0 is fixed by constraint, but we can set bounds too.
        
        A = np.array(A_rows)
        b_l = np.array(b_lower)
        b_u = np.array(b_upper)
        
        constraints = LinearConstraint(A, b_l, b_u)
        
        # Variable bounds
        # x_ij in {0, 1}
        # u_i in [0, n-1]
        integrality = np.zeros(num_vars)
        for i in range(n):
            for j in range(n):
                if i != j:
                    integrality[i * n + j] = 1
        # u variables are continuous, so integrality 0
        
        lb = np.zeros(num_vars)
        ub = np.zeros(num_vars)
        # x bounds
        for i in range(n):
            for j in range(n):
                if i != j:
                    ub[i * n + j] = 1
                    # If edge not in valid_edges, force x=0
                    if (i, j) not in valid_edges:
                        lb[i * n + j] = 0
                        ub[i * n + j] = 0
                    else:
                        lb[i * n + j] = 0
                        ub[i * n + j] = 1
                        
        # u bounds
        for i in range(n):
            if i == 0:
                lb[num_x + i] = 0
                ub[num_x + i] = 0
            else:
                lb[num_x + i] = 0
                ub[num_x + i] = n - 1
                
        bounds = Bounds(lb, ub)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status != 0: # 0 is success
            return None
            
        # Extract solution
        x_vals = res.x[:num_x]
        
        # Reconstruct cycle
        path = [0]
        current = 0
        for _ in range(n):
            next_node = -1
            for j in range(n):
                if j != current:
                    if x_vals[current * n + j] > 0.5:
                        next_node = j
                        break
            if next_node == -1:
                return None
            if next_node == 0 and len(path) < n:
                # Should not happen before all nodes visited if MTZ works correctly
                # But if it does, it's a subtour or error
                return None
            path.append(next_node)
            current = next_node
            
        if path[-1] != 0:
            return None
            
        # Remove the last 0 to get a permutation of 0..n-1
        # The cycle is 0 -> ... -> 0. The permutation is 0 -> ...
        # The task asks for a permutation of vertices 0..23 describing the cycle.
        # Usually this means v0, v1, ..., v_{n-1} such that edges (v_i, v_{i+1}) and (v_{n-1}, v_0) exist.
        # Our path is [0, v1, ..., v_{n-1}, 0].
        # So the permutation is path[:-1].
        return path[:-1]

    # Binary search for minimum K
    lo = n
    hi = n * (n - 1) // 2
    
    # Check feasibility at hi
    if find_hamiltonian_cycle_ilp(hi) is None:
        return list(range(n))
        
    while lo < hi:
        mid = (lo + hi) // 2
        if find_hamiltonian_cycle_ilp(mid) is not None:
            hi = mid
        else:
            lo = mid + 1
            
    K = lo
    cycle = find_hamiltonian_cycle_ilp(K)
    
    if cycle is None:
        return list(range(n))
        
    return cycle
