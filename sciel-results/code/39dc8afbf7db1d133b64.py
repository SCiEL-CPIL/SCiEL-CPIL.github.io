"""Program using MILP (scipy.optimize.milp) to find the minimum threshold K for a Hamiltonian cycle."""
import random
import sys
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

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

    def check_feasibility(K):
        """
        Check if there exists a Hamiltonian cycle in the graph formed by edges with position <= K.
        Returns the arc selection vector y (length 2*m) if feasible, else None.
        Uses MTZ formulation with MILP.
        """
        # Build list of arcs (u,v) where edge (u,v) has pos <= K
        arcs = []
        for (u, v) in all_edges:
            if edge_pos[(u, v)] <= K:
                arcs.append((u, v))
                arcs.append((v, u))
        
        m_arcs = len(arcs)
        if m_arcs == 0:
            return None
        
        # Map arc index to (u, v)
        # We'll build constraint matrices
        
        # Variables:
        # y[0..m_arcs-1]: binary, arc selected
        # x[0..n-1]: integer, position in cycle (1..n)
        # Total variables: m_arcs + n
        
        # Objective: minimize sum(y) (should be n for a Hamiltonian cycle)
        c = np.zeros(m_arcs + n)
        c[:m_arcs] = 1.0  # minimize number of arcs; optimal will be n
        
        # Constraints:
        # 1. Flow conservation: for each vertex v, sum of incoming arcs = 1 and sum of outgoing arcs = 1
        #    For each vertex v:
        #    sum_{(u,v) in arcs} y[idx(u,v)] = 1  (in-degree)
        #    sum_{(v,w) in arcs} y[idx(v,w)] = 1  (out-degree)
        
        # Build incidence structure
        in_arcs = {v: [] for v in range(n)}
        out_arcs = {v: [] for v in range(n)}
        for idx, (u, v) in enumerate(arcs):
            in_arcs[v].append(idx)
            out_arcs[u].append(idx)
        
        # 2. MTZ constraints: for each arc (u,v), x[u] - x[v] + n*y[idx] <= n - 1
        #    And x[u] - x[v] >= 1 - (n-1)*(1 - y[idx]) => x[u] - x[v] + (n-1)*(1-y[idx]) >= 1
        #    Actually, the standard MTZ is:
        #    x[u] - x[v] + n*y[idx] <= n - 1  for all arcs (u,v)
        #    This prevents subtours.
        
        # Let's build the constraint matrix.
        # Number of constraints:
        # - 2n for flow conservation (n in, n out)
        # - m_arcs for MTZ
        
        num_constraints = 2 * n + m_arcs
        
        # Build A matrix (sparse would be better but for n=24, m_arcs <= 276, so 552+276=828 constraints, 276+24=300 vars)
        # Dense matrix is fine.
        A = np.zeros((num_constraints, m_arcs + n))
        lb = np.zeros(num_constraints)
        ub = np.zeros(num_constraints)
        
        # Flow conservation constraints
        # For each vertex v: sum of in-arcs = 1
        for v in range(n):
            row = v
            for idx in in_arcs[v]:
                A[row, idx] = 1.0
            lb[row] = 1.0
            ub[row] = 1.0
            
            row = n + v
            for idx in out_arcs[v]:
                A[row, idx] = 1.0
            lb[row] = 1.0
            ub[row] = 1.0
        
        # MTZ constraints: x[u] - x[v] + n*y[idx] <= n - 1
        for idx, (u, v) in enumerate(arcs):
            row = 2 * n + idx
            # x[u] is at index m_arcs + u
            # x[v] is at index m_arcs + v
            # y[idx] is at index idx
            A[row, m_arcs + u] = 1.0
            A[row, m_arcs + v] = -1.0
            A[row, idx] = n
            lb[row] = -np.inf
            ub[row] = n - 1
        
        # Bounds on variables
        # y: binary [0,1]
        # x: integer [1, n]
        integrality = np.concatenate([np.ones(m_arcs), np.ones(n)])
        bounds = Bounds(
            lb=np.concatenate([np.zeros(m_arcs), np.ones(n)]),
            ub=np.concatenate([np.ones(m_arcs), np.full(n, n)])
        )
        
        constraint = LinearConstraint(A, lb, ub)
        
        try:
            res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
            if res.status == 0:
                return res.x[:m_arcs]
            else:
                return None
        except Exception:
            return None

    # Binary search for minimum K
    lo = n
    hi = n * (n - 1) // 2
    
    # Feasibility check at hi
    sol_hi = check_feasibility(hi)
    if sol_hi is None:
        return list(range(n))
        
    while lo < hi:
        mid = (lo + hi) // 2
        if check_feasibility(mid) is not None:
            hi = mid
        else:
            lo = mid + 1
            
    K = lo
    
    # Get the solution for the optimal K
    y = check_feasibility(K)
    
    if y is None:
        return list(range(n))
        
    # Reconstruct cycle from y
    # Build arcs list again
    arcs = []
    for (u, v) in all_edges:
        if edge_pos[(u, v)] <= K:
            arcs.append((u, v))
            arcs.append((v, u))
            
    # Find the next node for each node
    next_node = {}
    for idx, val in enumerate(y):
        if val > 0.5:
            u, v = arcs[idx]
            next_node[u] = v
            
    # Traverse
    cycle = [0]
    current = 0
    for _ in range(n - 1):
        current = next_node.get(current, None)
        if current is None:
            return list(range(n))
        cycle.append(current)
        
    return cycle
