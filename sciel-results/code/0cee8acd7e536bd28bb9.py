"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm

def construct():
    t0 = time.time()
    deadline = t0 + 55.0  # safety margin before 60s
    
    N = 256
    # Candidates: 2..256
    # We want max sum of 1/x for subset S of [2,256]
    # such that for all distinct a,b in S, lcm(a,b) > 256
    
    # Step 1: Build conflict graph
    # Two numbers conflict if lcm(a,b) <= 256
    # We can use the fact that lcm(a,b) = a*b / gcd(a,b)
    # So lcm(a,b) <= 256 iff a*b <= 256 * gcd(a,b)
    
    # Precompute gcd table or just compute on the fly
    # For efficiency, let's precompute conflicts for all pairs
    
    candidates = list(range(2, 257))
    n = len(candidates)  # 255 elements
    
    # Precompute conflict matrix (n x n boolean)
    # conflict[i][j] = True if lcm(candidates[i], candidates[j]) <= 256
    conflict = [[False] * n for _ in range(n)]
    for i in range(n):
        a = candidates[i]
        for j in range(i + 1, n):
            b = candidates[j]
            if lcm(a, b) <= 256:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # Now we need to find a maximum weight independent set
    # Weight of node i is 1/candidates[i]
    # This is a small enough problem for branch and bound with good heuristics
    
    # Sort candidates by weight (1/x) descending = by x ascending
    # Actually, smaller x has larger weight, so sort by x ascending
    # Let's reorder: index 0 = smallest x (highest weight), etc.
    order = sorted(range(n), key=lambda i: candidates[i])
    # order[0] is the index of the smallest candidate, etc.
    
    # Let's create an array of (weight, candidate_index) sorted by weight desc
    sorted_indices = order[:]  # already sorted by x ascending = weight descending
    
    # For branch and bound, we need to be efficient
    # Use a greedy upper bound: for remaining candidates, sum of all weights
    # But this is loose. Better: use the fact that we can only add non-conflicting ones.
    
    # Alternative: use a simple greedy + local search approach
    # Or use ILP if available
    
    try:
        import scipy.optimize as so
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Formulate as ILP:
        # max sum_i w_i * x_i
        # s.t. x_i + x_j <= 1 for all conflicting pairs (i,j)
        #      x_i in {0, 1}
        
        w = [1.0 / candidates[i] for i in range(n)]
        
        # Constraint matrix: for each conflicting pair (i,j), row is 0 except 1 at i and j
        # This could be many constraints. Let's count them.
        n_conflicts = 0
        for i in range(n):
            for j in range(i + 1, n):
                if conflict[i][j]:
                    n_conflicts += 1
        
        # If too many constraints, use a different approach
        if n_conflicts < 5000:
            # Build constraint matrix
            A_rows = []
            for i in range(n):
                for j in range(i + 1, n):
                    if conflict[i][j]:
                        row = [0.0] * n
                        row[i] = 1.0
                        row[j] = 1.0
                        A_rows.append(row)
            
            if A_rows:
                import numpy as np
                A = np.array(A_rows)
                b = np.ones(len(A_rows))
                
                c = -np.array(w)  # minimize -sum w_i x_i
                integrality = np.ones(n)  # all binary
                lb = np.zeros(n)
                ub = np.ones(n)
                
                constraints = [LinearConstraint(A, lb=np.zeros(len(A_rows)), ub=b)]
                
                res = milp(c, constraints=constraints, integrality=integrality,
                           bounds=Bounds(lb, ub))
                
                if res.status == 0 and res.x is not None:
                    selected = [candidates[i] for i in range(n) if res.x[i] > 0.5]
                    if selected:
                        return selected
                
                # If ILP failed, fall through to heuristic
            else:
                # No conflicts, take all
                return candidates
        
        # Fall through to heuristic if ILP not used or failed
    except Exception:
        pass
    
    # Heuristic: Greedy + local search
    # Start with empty set, add candidates in order of decreasing weight (increasing x)
    # if they don't conflict with already selected
    
    def greedy_set():
        selected_indices = []
        selected_conflict = set()
        for idx in sorted_indices:
            # Check if idx conflicts with any selected
            can_add = True
            for sidx in selected_indices:
                if conflict[idx][sidx]:
                    can_add = False
                    break
            if can_add:
                selected_indices.append(idx)
                selected_conflict.add(idx)
        return selected_indices
    
    def compute_weight(indices):
        return sum(1.0 / candidates[i] for i in indices)
    
    best_indices = greedy_set()
    best_weight = compute_weight(best_indices)
    
    # Local search: try adding one, removing one, swapping
    # Since the problem is small, we can do a few iterations of 1-opt
    
    def is_valid(indices):
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if conflict[indices[i]][indices[j]]:
                    return False
        return True
    
    # Try 1-opt: add a single unselected node
    selected_set = set(best_indices)
    for _ in range(3):  # a few passes
        improved = False
        # Try adding
        for idx in sorted_indices:
            if idx in selected_set:
                continue
            # Check if adding idx is valid
            valid = True
            for sidx in selected_set:
                if conflict[idx][sidx]:
                    valid = False
                    break
            if valid:
                new_set = selected_set | {idx}
                new_w = compute_weight(list(new_set))
                if new_w > best_weight:
                    best_weight = new_w
                    selected_set = new_set
                    best_indices = list(new_set)
                    improved = True
        # Try removing and adding (swap)
        if time.time() > deadline:
            break
        for sidx in list(selected_set):
            if time.time() > deadline:
                break
            for idx in sorted_indices:
                if idx in selected_set:
                    continue
                # Try swapping sidx with idx
                new_set = (selected_set - {sidx}) | {idx}
                # Check validity
                valid = True
                new_set_list = list(new_set)
                for i in range(len(new_set_list)):
                    for j in range(i + 1, len(new_set_list)):
                        if conflict[new_set_list[i]][new_set_list[j]]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    new_w = compute_weight(new_set_list)
                    if new_w > best_weight:
                        best_weight = new_w
                        selected_set = new_set
                        best_indices = list(new_set)
                        improved = True
        
        if not improved:
            break
    
    result = [candidates[i] for i in best_indices]
    if not result:
        result = [2]
    return result

# EVOLVE-BLOCK-END
