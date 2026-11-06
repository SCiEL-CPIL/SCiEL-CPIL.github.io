import math
import random

def construct():
    n = 256
    max_sum = 2 * n
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    # Identify diagonal conflicts (a+a is square)
    diagonal_conflicts = set()
    for a in range(1, n + 1):
        if 2 * a in squares:
            diagonal_conflicts.add(a)
            
    # Identify off-diagonal conflicts
    # We only care about nodes that are NOT in diagonal_conflicts
    valid_nodes = [a for a in range(1, n + 1) if a not in diagonal_conflicts]
    
    # Map valid nodes to indices for ILP
    # Index 0..k-1 corresponds to valid_nodes[0]..valid_nodes[k-1]
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    k = len(valid_nodes)
    
    # Build conflict list for valid nodes
    # conflicts[i] is a list of indices j such that valid_nodes[i] + valid_nodes[j] is a square
    conflicts = [[] for _ in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            if (valid_nodes[i] + valid_nodes[j]) in squares:
                conflicts[i].append(j)
                conflicts[j].append(i)
                
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Objective: Maximize sum(x) -> Minimize -sum(x)
        c = -np.ones(k)
        
        # Constraints: x_i + x_j <= 1 for all edges
        # We need to build the A matrix and b vector
        # Number of constraints is the number of edges
        edges = []
        for i in range(k):
            for j in conflicts[i]:
                if i < j:
                    edges.append((i, j))
                    
        m = len(edges)
        if m == 0:
            # No conflicts, take all
            return sorted(valid_nodes)
            
        A = np.zeros((m, k))
        b_ub = np.ones(m)
        
        for idx, (i, j) in enumerate(edges):
            A[idx, i] = 1
            A[idx, j] = 1
            
        # Bounds: 0 <= x <= 1
        lb = np.zeros(k)
        ub = np.ones(k)
        bounds = Bounds(lb, ub)
        
        # Integrality: 0 means continuous, 1 means integer
        integrality = np.ones(k)
        
        res = milp(c=c, constraints=LinearConstraint(A, None, b_ub), bounds=bounds, integrality=integrality)
        
        if res.success and res.x is not None:
            # res.x is the solution vector
            solution = set()
            for idx, val in enumerate(res.x):
                if val > 0.5:
                    solution.add(valid_nodes[idx])
            return sorted(solution)
        else:
            raise RuntimeError("ILP failed")
            
    except Exception as e:
        # Fallback to Simulated Annealing if ILP is not available or fails
        pass
        
    # Simulated Annealing Fallback
    # State: set of selected nodes
    # We start with a greedy solution
    def greedy_init():
        s = set()
        # Sort by degree ascending?
        # Degree in the conflict graph restricted to valid_nodes
        degs = [len(conflicts[i]) for i in range(k)]
        order = sorted(range(k), key=lambda i: degs[i])
        for i in order:
            # Check if valid to add
            ok = True
            for j in conflicts[i]:
                if valid_nodes[j] in s:
                    ok = False
                    break
            if ok:
                s.add(valid_nodes[i])
        return s
        
    best_set = greedy_init()
    best_size = len(best_set)
    
    # Precompute neighbor sets for fast checks
    # neighbor_set[i] = set of indices j that conflict with i
    neighbor_sets = [set(conflicts[i]) for i in range(k)]
    
    def is_valid_add(s_idx_set, v_idx):
        return len(neighbor_sets[v_idx] & s_idx_set) == 0
        
    # SA Parameters
    T = 100.0
    cooling_rate = 0.99
    iterations = 5000
    random.seed(42)
    
    current_set = set(best_set)
    # Map value to index for quick access? 
    # We store indices in the set for speed
    current_indices = set(node_to_idx[v] for v in current_set)
    
    for _ in range(iterations):
        # Generate a move
        # 1. Add a random valid node
        # 2. Remove a random node
        # 3. Swap
        
        move_type = random.random()
        
        if move_type < 0.3:
            # Add
            candidates_to_add = [i for i in range(k) if i not in current_indices and is_valid_add(current_indices, i)]
            if candidates_to_add:
                v = random.choice(candidates_to_add)
                new_indices = current_indices | {v}
                delta = 1 # Size increase
                if True: # Always accept if valid and larger
                    current_indices = new_indices
        elif move_type < 0.6:
            # Remove
            if current_indices:
                u = random.choice(list(current_indices))
                new_indices = current_indices - {u}
                delta = -1 # Size decrease
                # Accept with probability exp(delta/T)
                if random.random() < math.exp(delta / T):
                    current_indices = new_indices
        else:
            # Swap
            if current_indices:
                u = random.choice(list(current_indices))
                # Find valid nodes to add if u is removed
                temp_indices = current_indices - {u}
                candidates_to_add = [i for i in range(k) if i not in temp_indices and is_valid_add(temp_indices, i)]
                if candidates_to_add:
                    v = random.choice(candidates_to_add)
                    new_indices = temp_indices | {v}
                    delta = 0 # Size same
                    # Accept with probability 1 if same size? Or use some heuristic?
                    # For now, accept if it reduces "potential conflicts" or just random
                    if random.random() < 0.5:
                        current_indices = new_indices
                        
        # Check if better
        if len(current_indices) > best_size:
            best_size = len(current_indices)
            best_set = set(valid_nodes[i] for i in current_indices)
            
        T *= cooling_rate
        if T < 0.01:
            break
            
    return sorted(best_set)
