"""Program using ILP to find the minimum threshold K for a Hamiltonian cycle."""
import random
import itertools
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 24
    # Generate the pinned edge list with random.Random(614)
    rng = random.Random(614)
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((i, j))
    rng.shuffle(all_edges)
    
    # Build edge index lookup: edge -> position (1-based)
    edge_pos = {}
    for idx, (u, v) in enumerate(all_edges, 1):
        edge_pos[(u, v)] = idx
        edge_pos[(v, u)] = idx
    
    # For each threshold K, check if G_K contains a Hamiltonian cycle
    # We'll do a binary search on K, but since we want the minimum,
    # let's try increasing K from a reasonable lower bound.
    # A Hamiltonian cycle on n vertices has n edges.
    # The minimum possible K is at least n (need at least n edges).
    # Let's start from n and go up.
    
    # Actually, let's use binary search for efficiency.
    # Lower bound: n (need n edges for a cycle)
    # Upper bound: total edges = n*(n-1)//2 = 276
    
    def has_hamiltonian_cycle(K):
        """Check if graph with edges having position <= K contains a Hamiltonian cycle."""
        # Build adjacency: for each vertex, which other vertices are connected
        # and what's the edge position
        adj = {v: {} for v in range(n)}
        for (u, v) in all_edges:
            pos = edge_pos[(u, v)]
            if pos <= K:
                adj[u][v] = pos
                adj[v][u] = pos
        
        # Use ILP to find a Hamiltonian cycle
        # Variables: x_{ij} = 1 if edge (i,j) is in the cycle
        # For undirected graph, we can use directed variables x_{ij} for i != j
        # but we need to ensure symmetry or just use one direction.
        
        # Let's use directed variables: x_{ij} for all i != j
        # Constraints:
        # 1. Flow conservation: sum_j x_{ij} = 1 for all i (out-degree 1)
        #    sum_j x_{ji} = 1 for all i (in-degree 1)
        # 2. Subtour elimination (MTZ formulation):
        #    u_i - u_j + n * x_{ij} <= n - 1 for all i != j, i != 0, j != 0
        #    where u_i are auxiliary variables, 1 <= u_i <= n-1 for i != 0, u_0 = 0
        
        # Number of directed edges
        m = n * (n - 1)
        
        # Variable indices: x_{ij} is at index i*n + j
        def x_idx(i, j):
            return i * n + j
        
        # u_i is at index m + i
        def u_idx(i):
            return m + i
        
        total_vars = m + n
        
        # Objective: minimize 0 (feasibility)
        c = [0.0] * total_vars
        
        # Constraints
        # 1. Out-degree: sum_j x_{ij} = 1 for each i
        A_rows = []
        b_lower = []
        b_upper = []
        
        for i in range(n):
            row = [0.0] * total_vars
            for j in range(n):
                if i != j:
                    row[x_idx(i, j)] = 1.0
            A_rows.append(row)
            b_lower.append(1.0)
            b_upper.append(1.0)
        
        # 2. In-degree: sum_j x_{ji} = 1 for each i
        for i in range(n):
            row = [0.0] * total_vars
            for j in range(n):
                if i != j:
                    row[x_idx(j, i)] = 1.0
            A_rows.append(row)
            b_lower.append(1.0)
            b_upper.append(1.0)
        
        # 3. MTZ subtour elimination
        # For all i != j, i != 0, j != 0:
        # u_i - u_j + n * x_{ij} <= n - 1
        for i in range(1, n):
            for j in range(1, n):
                if i == j:
                    continue
                row = [0.0] * total_vars
                row[u_idx(i)] = 1.0
                row[u_idx(j)] = -1.0
                row[x_idx(i, j)] = float(n)
                A_rows.append(row)
                b_lower.append(-float('inf'))
                b_upper.append(float(n - 1))
        
        # Also need: u_i - u_j + n * x_{ji} <= n - 1 for i != j, i != 0, j != 0
        # Wait, the MTZ constraint is: for all i, j with i != j, i != 0, j != 0:
        # u_i - u_j + n * x_{ij} <= n - 1
        # This is sufficient.
        
        # Bounds
        # x_{ij} in {0, 1}
        # u_0 = 0
        # u_i in [1, n-1] for i != 0
        
        integrality = [1] * m + [0] * n  # x variables are integer (binary), u variables are continuous
        
        # Actually, for binary variables, we need to set them to 0 or 1
        # milp uses integrality: 1 for integer, 0 for continuous, -1 for free
        # For binary, we set integrality=1 and bounds [0,1]
        
        lb = [0.0] * total_vars
        ub = [0.0] * total_vars
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    ub[x_idx(i, j)] = 1.0
        
        # u_0 = 0
        lb[u_idx(0)] = 0.0
        ub[u_idx(0)] = 0.0
        
        # u_i in [1, n-1] for i != 0
        for i in range(1, n):
            lb[u_idx(i)] = 1.0
            ub[u_idx(i)] = float(n - 1)
        
        # Check which edges are available (position <= K)
        # If edge (i,j) is not available, set x_{ij} and x_{ji} to 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if (i, j) not in adj:
                    # Edge not available
                    ub[x_idx(i, j)] = 0.0
                    lb[x_idx(i, j)] = 0.0
        
        A = A_rows
        constraints = LinearConstraint(A, b_lower, b_upper)
        bounds = Bounds(lb, ub)
        
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            return res.status == 0  # 0 means success
        except Exception:
            return False
    
    # Binary search for minimum K
    lo = n  # at least n edges needed
    hi = n * (n - 1) // 2  # all edges
    
    # First check if it's feasible with all edges
    if not has_hamiltonian_cycle(hi):
        # Should not happen, but fallback
        return list(range(n))
    
    while lo < hi:
        mid = (lo + hi) // 2
        if has_hamiltonian_cycle(mid):
            hi = mid
        else:
            lo = mid + 1
    
    K = lo
    
    # Now find the actual Hamiltonian cycle using K
    # Re-run ILP to get the solution
    n_vars = n * (n - 1) + n
    m = n * (n - 1)
    
    def x_idx(i, j):
        return i * n + j
    
    def u_idx(i):
        return m + i
    
    total_vars = m + n
    
    c = [0.0] * total_vars
    
    A_rows = []
    b_lower = []
    b_upper = []
    
    for i in range(n):
        row = [0.0] * total_vars
        for j in range(n):
            if i != j:
                row[x_idx(i, j)] = 1.0
        A_rows.append(row)
        b_lower.append(1.0)
        b_upper.append(1.0)
    
    for i in range(n):
        row = [0.0] * total_vars
        for j in range(n):
            if i != j:
                row[x_idx(j, i)] = 1.0
        A_rows.append(row)
        b_lower.append(1.0)
        b_upper.append(1.0)
    
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                continue
            row = [0.0] * total_vars
            row[u_idx(i)] = 1.0
            row[u_idx(j)] = -1.0
            row[x_idx(i, j)] = float(n)
            A_rows.append(row)
            b_lower.append(-float('inf'))
            b_upper.append(float(n - 1))
    
    lb = [0.0] * total_vars
    ub = [0.0] * total_vars
    
    for i in range(n):
        for j in range(n):
            if i != j:
                ub[x_idx(i, j)] = 1.0
    
    lb[u_idx(0)] = 0.0
    ub[u_idx(0)] = 0.0
    
    for i in range(1, n):
        lb[u_idx(i)] = 1.0
        ub[u_idx(i)] = float(n - 1)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pos = edge_pos.get((i, j), float('inf'))
            if pos > K:
                ub[x_idx(i, j)] = 0.0
                lb[x_idx(i, j)] = 0.0
    
    integrality = [1] * m + [0] * n
    
    A = A_rows
    constraints = LinearConstraint(A, b_lower, b_upper)
    bounds = Bounds(lb, ub)
    
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    # Extract the cycle from the solution
    # x_{ij} > 0.5 means edge (i,j) is in the cycle
    # Build the cycle
    next_v = [None] * n
    for i in range(n):
        for j in range(n):
            if i != j and res.x[x_idx(i, j)] > 0.5:
                next_v[i] = j
                break
    
    # Reconstruct the permutation
    # Start from vertex 0, follow next_v
    cycle = [0]
    for _ in range(n - 1):
        cur = cycle[-1]
        nxt = next_v[cur]
        if nxt is None:
            # Fallback
            return list(range(n))
        cycle.append(nxt)
    
    return cycle
