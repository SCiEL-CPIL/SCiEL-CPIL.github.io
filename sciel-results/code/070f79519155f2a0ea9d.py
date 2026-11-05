import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute reciprocals
    recip = [0] * (n + 1)
    for i in range(1, n + 1):
        recip[i] = 1.0 / i

    # State
    in_set = [False] * (n + 1)
    
    def get_sum():
        return sum(recip[i] for i in range(1, n + 1) if in_set[i])

    def can_add(x):
        # Check if x can be added to current in_set
        for a in range(1, n + 1):
            if in_set[a]:
                # 1. x is middle: a, x, b -> 2x = a + b -> b = 2x - a
                b = 2 * x - a
                if b != x and 1 <= b <= n and in_set[b]:
                    return False
                # 2. x is endpoint, a is other endpoint: a, m, x -> 2m = a + x -> m = (a+x)/2
                if (a + x) % 2 == 0:
                    m = (a + x) // 2
                    if m != a and m != x and 1 <= m <= n and in_set[m]:
                        return False
                # 3. x is endpoint, a is middle: x, a, b -> 2a = x + b -> b = 2a - x
                b2 = 2 * a - x
                if b2 != x and 1 <= b2 <= n and in_set[b2]:
                    return False
        return True

    def greedy_fill():
        # Add all valid elements in increasing order
        # We iterate in increasing order because we want to prioritize small numbers (high reciprocal)
        # However, adding a small number might block a slightly larger one.
        # But since 1/x is convex, small numbers are so much more valuable that 
        # a greedy "small-first" approach is generally very strong for this specific objective.
        for x in range(1, n + 1):
            if not in_set[x]:
                if can_add(x):
                    in_set[x] = True

    def local_search():
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            current_sum = get_sum()
            
            # Try removing one element and refilling
            for rem in range(1, n + 1):
                if not in_set[rem]:
                    continue
                
                # Save state
                # Instead of copying the whole list, we can just toggle and restore if it fails
                # But greedy_fill modifies in_set. So we need a copy or careful restoration.
                # Given N=256, copying a list of bools is cheap.
                trial_set = in_set[:]
                
                trial_set[rem] = False
                
                # Greedy refill on trial_set
                # We need a version of greedy_fill that works on a specific array
                # Let's refactor greedy_fill to accept an array
                # For simplicity in this rewrite, I'll inline the logic or pass the array.
                
                # To avoid code duplication, I'll write a helper that takes an array.
                # But can_add depends on global in_set. 
                # Let's make can_add and greedy_fill take the array as an argument.
                
                pass # Placeholder for logic below

    # --- Refactored Helpers ---
    
    def can_add_to(x, s):
        for a in range(1, n + 1):
            if s[a]:
                b = 2 * x - a
                if b != x and 1 <= b <= n and s[b]:
                    return False
                if (a + x) % 2 == 0:
                    m = (a + x) // 2
                    if m != a and m != x and 1 <= m <= n and s[m]:
                        return False
                b2 = 2 * a - x
                if b2 != x and 1 <= b2 <= n and s[b2]:
                    return False
        return True

    def greedy_fill_array(s):
        for x in range(1, n + 1):
            if not s[x]:
                if can_add_to(x, s):
                    s[x] = True
        return s

    def get_sum_array(s):
        return sum(recip[i] for i in range(1, n + 1) if s[i])

    def local_search_array(s):
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            current_sum = get_sum_array(s)
            
            # Try removing one element
            for rem in range(1, n + 1):
                if not s[rem]:
                    continue
                
                trial = s[:]
                trial[rem] = False
                greedy_fill_array(trial)
                new_sum = get_sum_array(trial)
                
                if new_sum > current_sum + 1e-12:
                    s = trial
                    current_sum = new_sum
                    improved = True
                    break
            
            # If no improvement from removal, try swaps?
            # Removal + Refill is stronger than Swap because Refill can add multiple elements.
            # If removal + refill doesn't improve, a simple swap likely won't either 
            # (unless the refill is suboptimal, but greedy is usually good).
            
        return s

    # 1. Initial Greedy
    in_set = [False] * (n + 1)
    greedy_fill_array(in_set)
    best_set = in_set[:]
    best_sum = get_sum_array(best_set)
    
    # 2. Local Search on Initial
    improved_set = local_search_array(best_set[:])
    if get_sum_array(improved_set) > best_sum:
        best_set = improved_set
        best_sum = get_sum_array(best_set)
        
    # 3. Randomized Restart / Perturbation
    random.seed(42)
    
    while time.time() < deadline - 5.0:
        # Perturb best_set
        perturbed = best_set[:]
        
        # Remove 1 to 4 random elements
        num_removals = random.randint(1, 4)
        elements_in_set = [i for i in range(1, n + 1) if perturbed[i]]
        if not elements_in_set:
            break
            
        for _ in range(num_removals):
            if not elements_in_set:
                break
            idx = random.randint(0, len(elements_in_set) - 1)
            rem_val = elements_in_set[idx]
            perturbed[rem_val] = False
            elements_in_set.pop(idx)
        
        # Refill
        greedy_fill_array(perturbed)
        
        # Local Search
        improved_set = local_search_array(perturbed)
        
        new_sum = get_sum_array(improved_set)
        if new_sum > best_sum + 1e-12:
            best_sum = new_sum
            best_set = improved_set

    # --- MILP Attempt ---
    # Try to solve the problem using MILP with a time limit
    # Variables x_1 ... x_n are binary
    # Maximize sum(recip[i] * x_i)
    # Subject to x_a + x_b + x_c <= 2 for all 3-APs (a,b,c)
    # x_i in {0, 1}
    
    try:
        # Generate constraints
        # A 3-AP is defined by (a, a+d, a+2d) where 1<=a, a+2d<=n
        # Constraint: x_a + x_{a+d} + x_{a+2d} <= 2
        
        # We build the constraint matrix A and vector b
        # A is sparse for efficiency, but for n=256, dense is also fine (256x256 is small)
        # Number of 3-APs: sum_{d=1}^{127} (n - 2d) = 127*256 - 2*(127*128/2) = 32512 - 16128 = 16384 constraints
        # This is a small MILP.
        
        rows = []
        cols = []
        vals = []
        b_vec = []
        
        for a in range(1, n + 1):
            for d in range(1, n):
                c = a + 2 * d
                if c > n:
                    break
                b = a + d
                # Indices in MILP are 0-based for variables x_1...x_n -> 0...n-1
                rows.append(len(b_vec))
                cols.append(a - 1)
                vals.append(1.0)
                
                rows.append(len(b_vec))
                cols.append(b - 1)
                vals.append(1.0)
                
                rows.append(len(b_vec))
                cols.append(c - 1)
                vals.append(1.0)
                
                b_vec.append(2.0)
        
        from scipy.sparse import csr_matrix
        A = csr_matrix((vals, (rows, cols)), shape=(len(b_vec), n))
        
        # Objective: Minimize -sum(recip[i] * x_i)
        c_obj = [-recip[i] for i in range(1, n + 1)]
        
        # Bounds: x_i in [0, 1]
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: x_i are integers (binary)
        integrality = np.ones(n)
        
        # Constraints: A x <= b
        # LinearConstraint(A, lb=-np.inf, ub=b_vec)
        constraint = LinearConstraint(A, lb=-np.inf, ub=b_vec)
        
        # Solve
        # time_limit is in seconds
        milp_time_limit = max(1.0, deadline - time.time() - 5.0)
        
        res = milp(c=c_obj, constraints=constraint, integrality=integrality, bounds=bounds, options={"time_limit": milp_time_limit, "presolve": "default"})
        
        if res.status == 0 and res.x is not None:
            # res.x is the optimal solution (or best feasible if time limit reached)
            # Check if it's feasible
            milp_set = [False] * (n + 1)
            for i in range(n):
                if res.x[i] > 0.5:
                    milp_set[i + 1] = True
            
            # Validate MILP result
            s_set = set(i for i in range(1, n+1) if milp_set[i])
            valid = True
            for i, a in enumerate(sorted(s_set)):
                for b in sorted(s_set)[i+1:]:
                    if (a + b) % 2 == 0:
                        mid = (a + b) // 2
                        if mid != a and mid != b and mid in s_set:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                milp_sum = get_sum_array(milp_set)
                if milp_sum > best_sum:
                    best_set = milp_set
                    best_sum = milp_sum
                    
    except Exception:
        # If MILP fails or times out, we keep the heuristic result
        pass

    # Extract result
    result = [i for i in range(1, n + 1) if best_set[i]]
    
    # Validation
    s_set = set(result)
    for i, a in enumerate(result):
        for b in result[i+1:]:
            if (a + b) % 2 == 0:
                mid = (a + b) // 2
                if mid != a and mid != b and mid in s_set:
                    raise AssertionError("3-AP found")
                    
    return result
