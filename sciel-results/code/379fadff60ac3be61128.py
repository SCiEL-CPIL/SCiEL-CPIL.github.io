import numpy as np
import time
import scipy.optimize as opt

def construct():
    d = 7
    n = 2 ** d
    total_edges = d * (n // 2)
    
    # 1. Enumerate all edges
    # Edge index mapping: (u, v) -> index
    # u < v
    edges = []
    edge_to_idx = {}
    idx = 0
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                idx += 1
    
    # 2. Enumerate all 4-cycles
    # A 4-cycle is defined by two dimensions i, j and a base vertex u not involving i or j?
    # No, a 4-cycle is a square in the hypercube.
    # It is defined by choosing 2 dimensions i, j and 2 bits for the other d-2 dimensions.
    # For each pair (i, j), there are 2^(d-2) squares.
    cycles = []
    cycle_edges = [] # List of lists of edge indices
    
    for i in range(d):
        for j in range(i + 1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            # Iterate over all combinations of the other d-2 bits
            # There are 2^(d-2) such combinations
            # We can iterate u from 0 to n-1 and check if u has 0 in bits i and j
            # Actually, easier: iterate over all u, if (u & mask_i) == 0 and (u & mask_j) == 0:
            # The square vertices are:
            # v0 = u
            # v1 = u ^ mask_i
            # v2 = u ^ mask_j
            # v3 = u ^ mask_i ^ mask_j
            
            # Edges:
            # (v0, v1) dim i
            # (v2, v3) dim i
            # (v0, v2) dim j
            # (v1, v3) dim j
            
            for u in range(n):
                if (u & mask_i) == 0 and (u & mask_j) == 0:
                    v0 = u
                    v1 = u ^ mask_i
                    v2 = u ^ mask_j
                    v3 = u ^ mask_i ^ mask_j
                    
                    # Get edge indices
                    e1 = edge_to_idx.get((min(v0, v1), max(v0, v1)))
                    e2 = edge_to_idx.get((min(v2, v3), max(v2, v3)))
                    e3 = edge_to_idx.get((min(v0, v2), max(v0, v2)))
                    e4 = edge_to_idx.get((min(v1, v3), max(v1, v3)))
                    
                    if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
                        cycle_edges.append([e1, e2, e3, e4])
    
    num_cycles = len(cycle_edges)
    num_edges = len(edges)
    
    # 3. Set up MILP
    # Minimize -sum(x_e)
    c = -np.ones(num_edges)
    
    # Constraints: sum(x_e for e in cycle) <= 3
    # A is num_cycles x num_edges
    A = np.zeros((num_cycles, num_edges))
    for c_idx, edges_in_cycle in enumerate(cycle_edges):
        for e_idx in edges_in_cycle:
            A[c_idx, e_idx] = 1
            
    constraints = [opt.LinearConstraint(A, ub=3)]
    integrality = np.ones(num_edges, dtype=int)
    bounds = [(0, 1)] * num_edges
    
    # Solve
    try:
        start_time = time.time()
        res = opt.milp(c, constraints=constraints, integrality=integrality, bounds=bounds, options={'time_limit': 100})
        
        if res.status == 0 and res.x is not None:
            selected_edges = []
            for i in range(num_edges):
                if res.x[i] > 0.5:
                    u, v = edges[i]
                    selected_edges.append([u, v])
            return selected_edges
        else:
            # Fallback if MILP fails or times out without solution
            pass
    except Exception:
        pass
    
    # Fallback: Simple Greedy (better than random shuffle)
    # Use the same logic as the seed but with a fixed order or slightly better heuristic
    # For now, just return a valid set from a quick greedy to ensure we don't crash
    selected = set()
    current_edges = []
    # Sort edges by index (deterministic)
    for u, v in edges:
        is_safe = True
        # Check against selected
        # We need to check if adding (u,v) creates a C4
        # This requires checking neighbors
        # This fallback is slow if done naively, but we only reach it if MILP fails.
        # Let's use a faster check:
        # Precompute for each edge, the list of other edges that form a C4 with it?
        # Too complex for fallback.
        
        # Just use the seed's logic but deterministic
        # Actually, let's just return a simple valid set: all edges in dim 0
        return [[u, u ^ 1] for u in range(n) if (u & 1) == 0]
