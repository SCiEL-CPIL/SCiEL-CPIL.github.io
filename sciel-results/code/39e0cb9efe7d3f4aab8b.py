"""Program using ILP to find the minimum threshold K for a Hamiltonian cycle."""
import random
import sys
import numpy as np
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

    def check_hamiltonian_cycle(K):
        """
        Check if there is a Hamiltonian cycle using only edges with position <= K.
        Returns the cycle as a list of vertices or None.
        """
        # Collect edges available in G_K
        available_edges = []
        for (u, v) in all_edges:
            if edge_pos[(u, v)] <= K:
                available_edges.append((u, v))
        
        # Quick check: degrees
        degrees = {v: 0 for v in range(n)}
        for u, v in available_edges:
            degrees[u] += 1
            degrees[v] += 1
        for v in range(n):
            if degrees[v] < 2:
                return None

        m = len(available_edges)
        if m < n:
            return None

        # Variables:
        # x_i for i in range(m) : binary, 1 if edge i is in cycle
        # u_v for v in range(n) : integer, position in cycle (0 to n-1)
        
        # Total variables: m + n
        # Indices:
        # 0 .. m-1 : x
        # m .. m+n-1 : u
        
        # Objective: Minimize 0 (feasibility problem)
        c = np.zeros(m + n)
        
        # Constraints:
        # 1. Degree constraints: sum of x for edges incident to v == 2
        #    For each v, sum_{e incident to v} x_e = 2
        A_rows = []
        b_lows = []
        b_highs = []
        
        # Degree constraints
        for v in range(n):
            row = np.zeros(m + n)
            for i, (u, w) in enumerate(available_edges):
                if u == v or w == v:
                    row[i] = 1.0
            A_rows.append(row)
            b_lows.append(2.0)
            b_highs.append(2.0)
            
        # 2. MTZ constraints: u_i - u_j + n * x_ij <= n - 1
        #    for all directed pairs (i, j) such that (i, j) is in available_edges
        #    Note: available_edges are undirected. We need to add constraints for both directions if the edge exists.
        #    x_ij is the same variable for (i,j) and (j,i).
        
        # We need to map edge index to variable index
        # Let's create a dictionary for edge to var index
        edge_to_var = {}
        for i, (u, v) in enumerate(available_edges):
            edge_to_var[(u, v)] = i
            edge_to_var[(v, u)] = i
            
        for i, (u, v) in enumerate(available_edges):
            # Constraint for direction u -> v
            row1 = np.zeros(m + n)
            row1[m + u] = 1.0
            row1[m + v] = -1.0
            row1[i] = float(n)
            A_rows.append(row1)
            b_lows.append(-np.inf)
            b_highs.append(float(n - 1))
            
            # Constraint for direction v -> u
            row2 = np.zeros(m + n)
            row2[m + v] = 1.0
            row2[m + u] = -1.0
            row2[i] = float(n)
            A_rows.append(row2)
            b_lows.append(-np.inf)
            b_highs.append(float(n - 1))

        # 3. Fix u_0 = 0
        row_fix = np.zeros(m + n)
        row_fix[m + 0] = 1.0
        A_rows.append(row_fix)
        b_lows.append(0.0)
        b_highs.append(0.0)
        
        A = np.array(A_rows)
        b_low = np.array(b_lows)
        b_high = np.array(b_highs)
        
        constraints = LinearConstraint(A, b_low, b_high)
        
        # Bounds
        # x_i in [0, 1]
        # u_v in [0, n-1]
        lb = np.zeros(m + n)
        ub = np.zeros(m + n)
        for i in range(m):
            lb[i] = 0
            ub[i] = 1
        for v in range(n):
            lb[m + v] = 0
            ub[m + v] = n - 1
            
        # Integrality
        # x_i are binary (integer)
        # u_v are integer
        integrality = np.ones(m + n)
        
        res = milp(c, constraints=constraints, integrality=integrality, bounds=Bounds(lb, ub))
        
        if res.status != 0:
            return None
            
        # Extract cycle
        # Get x values
        x_vals = res.x[:m]
        cycle_edges = []
        for i in range(m):
            if x_vals[i] > 0.5:
                cycle_edges.append(available_edges[i])
                
        # Reconstruct path from edges
        # Build adjacency list
        adj = {v: [] for v in range(n)}
        for u, v in cycle_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Start from 0
        path = [0]
        visited = {0}
        current = 0
        while len(path) < n:
            nexts = adj[current]
            next_unvisited = [x for x in nexts if x not in visited]
            if not next_unvisited:
                return None
            # Should be exactly 1 if it's a simple cycle
            current = next_unvisited[0]
            path.append(current)
            visited.add(current)
            
        # Verify closure
        if 0 not in adj[current]:
            return None
            
        return path

    # Binary search for minimum K
    lo = n
    hi = n * (n - 1) // 2
    
    # Ensure hi is feasible (should be, as complete graph has HC)
    if check_hamiltonian_cycle(hi) is None:
        return list(range(n))
        
    while lo < hi:
        mid = (lo + hi) // 2
        if check_hamiltonian_cycle(mid) is not None:
            hi = mid
        else:
            lo = mid + 1
            
    K = lo
    cycle = check_hamiltonian_cycle(K)
    
    if cycle is None:
        return list(range(n))
        
    return cycle
