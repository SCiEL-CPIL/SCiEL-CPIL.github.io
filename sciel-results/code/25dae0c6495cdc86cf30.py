import math
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s margin for 90s limit
    
    N = 256
    # Candidates are 2..256
    # Index i corresponds to number i+2
    n_nodes = 255
    numbers = list(range(2, 257))
    
    # Objective: Maximize sum(1/x_i)
    # milp minimizes c @ x, so we set c = -1/x_i
    c = np.array([-1.0 / x for x in numbers])
    
    # Constraints: x_i + x_j <= 1 for all conflicts (i, j)
    # We need to build the A matrix and b vector.
    # To avoid building a huge dense matrix, we can use sparse matrices if needed, 
    # but n_nodes=255, so a dense matrix of size E x 255 is fine if E is not huge.
    
    # First, count conflicts to estimate memory
    conflict_pairs = []
    for i in range(n_nodes):
        a = i + 2
        for j in range(i + 1, n_nodes):
            b = j + 2
            # lcm(a, b) = a * b // gcd(a, b)
            # Check if lcm <= 256
            g = math.gcd(a, b)
            if a * b <= 256 * g:
                conflict_pairs.append((i, j))
                
    E = len(conflict_pairs)
    
    # If no conflicts, take all
    if E == 0:
        return numbers
        
    # Build constraint matrix A
    # Each constraint is x_i + x_j <= 1
    # A is E x n_nodes
    # b is E x 1 with all 1s
    
    # Rows and cols for sparse matrix
    row_idx = []
    col_idx = []
    data = []
    
    for k, (i, j) in enumerate(conflict_pairs):
        row_idx.append(k)
        col_idx.append(i)
        data.append(1)
        
        row_idx.append(k)
        col_idx.append(j)
        data.append(1)
        
    A = sp.csr_matrix((data, (row_idx, col_idx)), shape=(E, n_nodes))
    b = np.ones(E)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(0, 1)
    
    # Integrality: 1 means integer
    integrality = np.ones(n_nodes)
    
    # Linear Constraint: A @ x <= b
    # scipy.optimize.milp uses LinearConstraint(A, lb, ub)
    # We want A @ x <= b, so lb = -inf, ub = b
    constraints = LinearConstraint(A, lb=-np.inf, ub=b)
    
    # Precompute conflict sets for fast lookup
    conflict_sets = [set() for _ in range(n_nodes)]
    for i, j in conflict_pairs:
        conflict_sets[i].add(j)
        conflict_sets[j].add(i)
    
    best_result = None
    best_score = 0.0
    
    # Try MIP with time limit
    try:
        # Use time_limit to ensure we don't exceed deadline
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:  # Need at least 5s for MIP
            res = milp(c=c, constraints=constraints, integrality=integrality, 
                      bounds=bounds, time_limit=remaining_time - 2.0)
            
            if res.status == 0 or res.status == 1:  # Optimal or best found
                # Extract solution
                x = res.x
                # x should be 0 or 1 (or very close)
                selected_indices = [i for i in range(n_nodes) if x[i] > 0.5]
                result = [idx + 2 for idx in selected_indices]
                
                if result:
                    # Validate
                    valid = True
                    for idx in range(len(result)):
                        for jdx in range(idx+1, len(result)):
                            a, b = result[idx], result[jdx]
                            if math.lcm(a, b) <= 256:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        score = sum(1.0/x for x in result)
                        best_result = result
                        best_score = score
                        return result
                    else:
                        # Invalid, try fallback
                        pass
    except Exception as e:
        # MIP failed or timed out
        pass
    
    # Fallback: Improved greedy with local search
    # Order by 1/x descending => x ascending
    # But we can try multiple orderings and pick best
    
    # Simple greedy first
    def greedy_select(order):
        selected = []
        selected_set = set()
        for i in order:
            if not conflict_sets[i].intersection(selected_set):
                selected.append(i)
                selected_set.add(i)
        return [idx + 2 for idx in selected]
    
    # Try different orderings
    candidates_orders = [
        list(range(n_nodes)),  # natural order (smallest first)
        sorted(range(n_nodes), key=lambda i: -1.0/(i+2)),  # by weight desc
        sorted(range(n_nodes), key=lambda i: (i+2, -len(conflict_sets[i])))  # by number, then degree
    ]
    
    for order in candidates_orders:
        result = greedy_select(order)
        if result:
            score = sum(1.0/x for x in result)
            if score > best_score:
                best_score = score
                best_result = result
    
    # Local search: try to improve by swapping
    if best_result is None:
        return [2]
    
    # Convert to indices
    best_indices = set(x - 2 for x in best_result)
    
    # Try local search: for each selected node, try removing it and adding a better non-selected node
    improved = True
    max_iterations = 100
    iteration = 0
    while improved and iteration < max_iterations and time.time() < deadline - 5.0:
        improved = False
        iteration += 1
        
        current_score = best_score
        current_indices = set(best_indices)
        
        # Try removing each selected node
        for remove_idx in list(current_indices):
            # Try adding each non-selected node
            for add_idx in range(n_nodes):
                if add_idx in current_indices:
                    continue
                if conflict_sets[add_idx].intersection(current_indices - {remove_idx}):
                    continue
                
                # Check if swap improves
                new_score = current_score - 1.0/(remove_idx + 2) + 1.0/(add_idx + 2)
                if new_score > current_score + 1e-10:
                    # Perform swap
                    current_indices.remove(remove_idx)
                    current_indices.add(add_idx)
                    current_score = new_score
                    improved = True
                    best_indices = current_indices
                    best_score = current_score
                    break
            if improved:
                break
        
        # Also try adding without removing
        for add_idx in range(n_nodes):
            if add_idx in current_indices:
                continue
            if conflict_sets[add_idx].intersection(current_indices):
                continue
            
            new_score = current_score + 1.0/(add_idx + 2)
            if new_score > current_score + 1e-10:
                current_indices.add(add_idx)
                current_score = new_score
                improved = True
                best_indices = current_indices
                best_score = current_score
                break
        
        # Try removing a low-value node if it doesn't help
        if not improved:
            for remove_idx in list(current_indices):
                if len(current_indices) > 1:
                    new_score = current_score - 1.0/(remove_idx + 2)
                    # Only remove if it allows adding something better
                    # For now, skip pure removals
    
    # Convert back
    result = [idx + 2 for idx in best_indices]
    if result:
        return result
    else:
        return [2]
