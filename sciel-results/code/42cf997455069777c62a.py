import time
from math import gcd
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 256
    nodes = list(range(2, N + 1))
    n = len(nodes)
    
    # Build conflict graph: 
    # x_i + x_j <= 1 if lcm(i, j) <= 256
    # If lcm(i, j) > 256, no constraint needed (they can both be 1)
    
    # Weights
    weights = np.array([1.0 / x for x in nodes])
    
    # Constraints for MILP
    # We need to formulate the constraints matrix A and bounds b
    # A @ x <= b
    
    # Initialize A and b
    # Number of constraints is roughly number of conflicting pairs
    # We can build it dynamically or pre-allocate
    
    A_rows = []
    b_vals = []
    
    # Iterate over all pairs to find conflicts
    # Conflict if lcm(a, b) <= 256
    # lcm(a, b) = a * b // gcd(a, b)
    # a * b // g <= 256  <=> a * b <= 256 * g
    
    for i in range(n):
        a = nodes[i]
        for j in range(i + 1, n):
            b = nodes[j]
            g = gcd(a, b)
            if a * b <= 256 * g:
                # Conflict: x_i + x_j <= 1
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_rows.append(row)
                b_vals.append(1.0)
                
    if not A_rows:
        # No conflicts? Then pick all? 
        # But we want to maximize sum of reciprocals. 
        # If no conflicts, we pick all nodes.
        # However, there are definitely conflicts for small numbers.
        # e.g., 2 and 4: lcm(2,4)=4 <= 256.
        # So A_rows should not be empty.
        
        # Fallback if somehow empty
        return nodes
        
    A = np.array(A_rows)
    b = np.array(b_vals)
    
    # Constraints:
    # 1. Conflict constraints: A @ x <= b
    # 2. Non-empty constraint: sum(x) >= 1  => -sum(x) <= -1
    #    Or just rely on the solver finding a non-trivial solution if we maximize positive weights.
    #    Actually, if we maximize positive weights, the solver will try to include as many as possible.
    #    If all x_i = 0 is a feasible solution (it is), but it gives 0 weight.
    #    Since weights are positive, the optimal solution will have at least one x_i = 1 unless all weights are 0 (which they aren't).
    #    So we don't strictly need the non-empty constraint if weights > 0.
    
    # Define LinearConstraint
    # A @ x <= b
    lin_cons = LinearConstraint(A, ub=b)
    
    # Variable bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are integers
    integrality = np.ones(n)
    
    # Objective: maximize sum(weights * x)
    # scipy.optimize.milp minimizes c @ x
    # So we minimize -weights @ x
    c = -weights
    
    # Solve
    # time_limit is in seconds
    # We use 50s to be safe
    try:
        res = milp(
            c=c,
            constraints=lin_cons,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 50.0, "mip_rel_gap": 1e-9}
        )
        
        if res.status == 0: # Optimal
            x = res.x
            # Extract selected nodes
            selected_indices = np.where(x > 0.5)[0]
            if len(selected_indices) == 0:
                # Fallback to single best node
                best_idx = np.argmax(weights)
                selected_indices = np.array([best_idx])
            
            result = [nodes[i] for i in selected_indices]
            
            # Verification
            if len(result) > 1:
                valid = True
                for i in range(len(result)):
                    for j in range(i + 1, len(result)):
                        a, b = result[i], result[j]
                        g = gcd(a, b)
                        if a * b <= 256 * g:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    # Solver returned invalid solution? Unlikely but possible with numerical issues
                    # Fallback to greedy
                    greedy_set = []
                    for idx in np.argsort(weights)[::-1]:
                        node = nodes[idx]
                        ok = True
                        for y in greedy_set:
                            g = gcd(node, y)
                            if node * y <= 256 * g:
                                ok = False
                                break
                        if ok:
                            greedy_set.append(node)
                    result = greedy_set
                    
            return result
            
        else:
            # Solver failed or timed out
            pass
            
    except Exception as e:
        print(f"MILP Error: {e}")
        
    # Fallback: Greedy Heuristic
    # Sort by weight descending (1/x descending => x ascending? No, 1/x is larger for smaller x)
    # So we prefer smaller numbers.
    # But smaller numbers have more conflicts.
    # Let's try a simple greedy: pick highest weight node, remove its conflicts, repeat.
    
    # Weights are 1/x. Highest weight is 1/2, then 1/3, etc.
    # Order indices by weight descending
    order = np.argsort(weights)[::-1]
    
    remaining = set(range(n))
    greedy_set = []
    
    for idx in order:
        if time.time() > deadline - 5:
            break
        if idx not in remaining:
            continue
            
        # Check if idx is compatible with all in greedy_set
        # Actually, the "remove conflicts" strategy is better.
        # If we pick idx, we must remove all nodes conflicting with idx from 'remaining'.
        
        # But we need to ensure idx doesn't conflict with already picked nodes.
        # Since we remove conflicts from 'remaining' when we pick a node, 
        # any node in 'remaining' is compatible with all nodes in 'greedy_set'.
        
        # So we just pick idx, add to greedy_set, and remove idx and its conflicts from remaining.
        
        # Wait, the standard greedy for Max Weight Clique is:
        # 1. Pick node with max weight.
        # 2. Add to solution.
        # 3. Remove node and all its non-neighbors (conflicts) from graph.
        # 4. Repeat.
        
        # Let's implement this properly.
        
        # We need a fast way to check conflicts.
        # Pre-compute conflict mask? 
        # n=255. 255*255 matrix is fine.
        
    # Re-implementing fallback properly
    conflict_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        a = nodes[i]
        for j in range(i + 1, n):
            b = nodes[j]
            g = gcd(a, b)
            if a * b <= 256 * g:
                conflict_mask[i, j] = True
                conflict_mask[j, i] = True
                
    remaining = set(range(n))
    greedy_set = []
    
    while remaining:
        # Find node in remaining with max weight
        max_idx = -1
        max_w = -1
        for idx in remaining:
            if weights[idx] > max_w:
                max_w = weights[idx]
                max_idx = idx
                
        if max_idx == -1:
            break
            
        greedy_set.append(max_idx)
        
        # Remove max_idx and all its conflicts from remaining
        to_remove = {max_idx}
        for j in range(n):
            if conflict_mask[max_idx, j]:
                to_remove.add(j)
                
        remaining -= to_remove
        
    if greedy_set:
        result = [nodes[i] for i in greedy_set]
        return result
    else:
        return [nodes[0]]

if __name__ == "__main__":
    construct()
