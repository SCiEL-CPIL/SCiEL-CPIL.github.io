"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import lcm, gcd
    from fractions import Fraction
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds

    start_time = time.time()
    deadline = start_time + 85.0

    # Nodes are integers 2..256
    nodes = list(range(2, 257))
    N = len(nodes)

    # Build conflict graph: edge (i, j) exists if lcm(nodes[i], nodes[j]) <= 256
    # We want an independent set in this graph (i.e., no two selected nodes conflict)
    
    # Precompute conflicts using a more efficient method
    # For each node, we can determine which other nodes conflict with it
    # lcm(a, b) = a * b // gcd(a, b) <= 256
    
    # Conflict adjacency list
    conflict_adj = [set() for _ in range(N)]
    
    for i in range(N):
        ai = nodes[i]
        for j in range(i + 1, N):
            aj = nodes[j]
            if lcm(ai, aj) <= 256:
                conflict_adj[i].add(j)
                conflict_adj[j].add(i)
    
    # Collect conflict edges for MILP
    conflict_edges = []
    for i in range(N):
        for j in conflict_adj[i]:
            if i < j:
                conflict_edges.append((i, j))
    
    # Weights: 1/x for each node
    weights = [Fraction(1, x) for x in nodes]
    
    # Formulate MILP
    # Maximize sum(w_i * x_i)
    # Subject to: x_i + x_j <= 1 for all conflict edges (i, j)
    # x_i in {0, 1}
    
    # Use float coefficients for the solver
    c = np.array([-float(w) for w in weights])
    
    # Build constraint matrix
    E = len(conflict_edges)
    if E == 0:
        # No conflicts, take all nodes
        return nodes
    
    A = np.zeros((E, N))
    b = np.ones(E)
    
    for k, (i, j) in enumerate(conflict_edges):
        A[k, i] = 1.0
        A[k, j] = 1.0
    
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(N)
    
    # Solve MILP
    res = milp(
        c=c,
        constraints=LinearConstraint(A, ub=b),
        integrality=integrality,
        bounds=bounds
    )
    
    if not res.success:
        # Fallback: greedy by weight (smallest x first)
        selected = []
        selected_set = set()
        for x in sorted(nodes, key=lambda v: 1.0/v, reverse=True):
            idx = nodes.index(x)
            if not any(nodes[j] in selected_set for j in conflict_adj[idx]):
                selected.append(x)
                selected_set.add(x)
        return selected
    
    # Extract solution
    x = res.x
    selected_indices = [i for i in range(N) if x[i] > 0.5]
    
    if not selected_indices:
        return [2]
    
    selected_nodes = [nodes[i] for i in selected_indices]
    selected_set = set(selected_nodes)
    
    # Verify validity
    for i in range(len(selected_nodes)):
        for j in range(i + 1, len(selected_nodes)):
            if lcm(selected_nodes[i], selected_nodes[j]) <= 256:
                # Invalid, use greedy fallback
                selected = []
                selected_set = set()
                for x in sorted(nodes, key=lambda v: 1.0/v, reverse=True):
                    idx = nodes.index(x)
                    if not any(nodes[j] in selected_set for j in conflict_adj[idx]):
                        selected.append(x)
                        selected_set.add(x)
                return selected
    
    # Local search polish: try swaps to improve the reciprocal sum
    # Current sum
    current_sum = sum(Fraction(1, x) for x in selected_nodes)
    
    # Precompute for each node, the set of selected nodes it conflicts with
    # This helps quickly check if a swap is valid
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        
        best_gain = Fraction(0, 1)
        best_swap = None
        
        # Try removing a and adding b where 1/b > 1/a (i.e., b < a)
        # b must not conflict with any node in selected \ {a}
        
        for a in selected_nodes:
            a_idx = nodes.index(a)
            remaining = selected_set - {a}
            
            # Candidate b's: nodes not in selected_set, with b < a
            for b in nodes:
                if b in selected_set:
                    continue
                if b >= a:
                    continue
                b_idx = nodes.index(b)
                
                # Check if b conflicts with any node in remaining
                conflicts = False
                for y in remaining:
                    y_idx = nodes.index(y)
                    if y_idx in conflict_adj[b_idx]:
                        conflicts = True
                        break
                if conflicts:
                    continue
                
                gain = Fraction(1, b) - Fraction(1, a)
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (a, b)
        
        if best_swap is not None:
            a, b = best_swap
            selected_nodes.remove(a)
            selected_nodes.append(b)
            selected_set.discard(a)
            selected_set.add(b)
            current_sum += best_gain
            improved = True
    
    return selected_nodes
# EVOLVE-BLOCK-END
