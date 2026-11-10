import math
import numpy as np

def construct():
    N = 256
    max_sum = 2 * N
    squares = set()
    k = 1
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1
    
    # Identify invalid nodes (diagonal constraint: 2*i is a square)
    invalid = [False] * (N + 1)
    for i in range(1, N + 1):
        if 2 * i in squares:
            invalid[i] = True
            
    valid_nodes = [i for i in range(1, N + 1) if not invalid[i]]
    num_valid = len(valid_nodes)
    
    # Map valid node index to its position in the valid_nodes list for matrix construction
    node_to_idx = {node: idx for idx, node in enumerate(valid_nodes)}
    
    # Identify conflicts (i < j, i+j is square)
    # We only need to consider pairs where both are valid
    conflicts = []
    for i in valid_nodes:
        for j in valid_nodes:
            if i >= j:
                continue
            if i + j in squares:
                conflicts.append((i, j))
                
    # Try to solve using ILP if scipy.optimize.milp is available
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import time
        
        start_time = time.time()
        
        # c is the objective vector, we want to maximize sum(x), so minimize -sum(x)
        c = -np.ones(num_valid)
        
        # Constraints: x_i + x_j <= 1 for each conflict
        if conflicts:
            A = np.zeros((len(conflicts), num_valid))
            b = np.ones(len(conflicts))
            for k, (i, j) in enumerate(conflicts):
                idx_i = node_to_idx[i]
                idx_j = node_to_idx[j]
                A[k, idx_i] = 1
                A[k, idx_j] = 1
            
            # Linear constraints: A @ x <= b
            # milp expects LinearConstraint(A, lb, ub)
            # We want A @ x <= b, which is equivalent to A @ x >= -inf and A @ x <= b
            # Or simply define the constraint as A @ x <= b.
            # scipy.optimize.milp uses LinearConstraint(A, lb, ub)
            # So we can set lb = -np.inf, ub = b
            cons = LinearConstraint(A, -np.inf, b)
        else:
            cons = LinearConstraint(np.zeros((0, num_valid)), -np.inf, np.zeros(0))
            
        # Bounds: 0 <= x_i <= 1
        lb = np.zeros(num_valid)
        ub = np.ones(num_valid)
        bounds = Bounds(lb, ub)
        
        # Integrality: all variables are integer
        integrality = np.ones(num_valid)
        
        # Solve
        res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            # res.x contains the solution
            # Since we minimized -sum, res.x should be 0 or 1
            solution = []
            for idx, val in enumerate(res.x):
                if val > 0.5:
                    solution.append(valid_nodes[idx])
            return solution
        else:
            # If ILP fails, fallback to greedy
            pass
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: Greedy with degree heuristic (similar to original but cleaner)
    # Since we are in a "new approach" context, and ILP is the main thing,
    # the fallback is just to ensure we return *something* valid if ILP fails.
    
    # Build adjacency for valid nodes
    adj = {i: set() for i in valid_nodes}
    for i, j in conflicts:
        adj[i].add(j)
        adj[j].add(i)
        
    # Simple greedy: pick node with lowest degree, remove neighbors
    remaining = set(valid_nodes)
    result = []
    while remaining:
        # Find node with min degree in remaining
        min_deg_node = min(remaining, key=lambda x: len(adj[x] & remaining))
        result.append(min_deg_node)
        remaining.discard(min_deg_node)
        # Remove neighbors
        for neighbor in adj[min_deg_node]:
            if neighbor in remaining:
                remaining.discard(neighbor)
                
    return result
