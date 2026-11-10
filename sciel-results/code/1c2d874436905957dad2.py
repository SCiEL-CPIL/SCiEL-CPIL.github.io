"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

def construct():
    n = 128
    target_low = 128
    target_high = 256
    targets = list(range(target_low, target_high + 1))
    
    # Pre-compute pairs for each target sum
    # pairs[s] = list of (a, b) such that a <= b, a+b=s, 0<=a,b<=n
    pairs = {s: [] for s in targets}
    for a in range(n + 1):
        for b in range(a, n + 1):
            s = a + b
            if target_low <= s <= target_high:
                pairs[s].append((a, b))
                
    # Get all unique pairs involved
    all_pairs = []
    for s in targets:
        all_pairs.extend(pairs[s])
    # Remove duplicates? No, (a,b) is unique for a sum? 
    # Yes, for fixed s and a<=b, a is determined by b=s-a.
    # So each pair (a,b) belongs to exactly one sum s.
    # Thus, we can index pairs globally.
    
    unique_pairs = []
    pair_to_index = {}
    for s in targets:
        for p in pairs[s]:
            if p not in pair_to_index:
                pair_to_index[p] = len(unique_pairs)
                unique_pairs.append(p)
                
    num_x = n + 1 # x_0 ... x_128
    num_z = len(unique_pairs)
    num_vars = num_x + num_z
    
    # Indices
    x_idx = lambda i: i
    z_idx = lambda p: num_x + pair_to_index[p]
    
    def solve_for_K(K):
        # Objective: Minimize sum(x_i) to find a sparse set, or just 0
        # We just need feasibility. But minimizing size might help?
        # Actually, let's just minimize 0 (feasibility) or sum(x) to break ties.
        c = np.zeros(num_vars)
        c[:num_x] = 1.0 # Minimize size
        
        # Constraints:
        # 1. For each sum s, 1 <= sum(z_{a,b} for (a,b) in pairs[s]) <= K
        # 2. z_{a,b} <= x_a
        # 3. z_{a,b} <= x_b
        # 4. z_{a,b} >= x_a + x_b - 1  =>  -z_{a,b} + x_a + x_b >= 1
        
        A_ub = [] # for <= constraints
        b_ub = []
        A_eq = []
        b_eq = []
        
        # We use LinearConstraint(A, lb, ub)
        # Let's build A matrix for all constraints.
        # It's better to use sparse matrices if possible, but num_vars ~ 8000, constraints ~ 129*3 + 8000*3 ~ 25000.
        # Dense might be slow. Let's try lil_matrix.
        
        from scipy.sparse import lil_matrix
        A_lil = lil_matrix((num_vars * 2 + len(targets) * 2, num_vars)) # Rough estimate
        
        # Better: Build list of constraints and use LinearConstraint
        # LinearConstraint takes A, lb, ub.
        # We can stack them.
        
        # Let's just use the list of constraints for milp?
        # milp expects a single A matrix.
        
        # Let's construct A and bounds.
        # Rows:
        # 1. For each s: sum(z) >= 1
        # 2. For each s: sum(z) <= K
        # 3. For each pair (a,b): z - x_a <= 0
        # 4. For each pair (a,b): z - x_b <= 0
        # 5. For each pair (a,b): -z + x_a + x_b >= 1  => z - x_a - x_b <= -1
        
        num_constraints = 0
        # 1 & 2: 2 * len(targets)
        # 3, 4, 5: 3 * num_z
        num_constraints = 2 * len(targets) + 3 * num_z
        
        A = np.zeros((num_constraints, num_vars))
        lb = np.zeros(num_constraints)
        ub = np.zeros(num_constraints)
        
        row = 0
        for s in targets:
            # sum(z) >= 1
            for p in pairs[s]:
                idx = z_idx(p)
                A[row, idx] = 1.0
            lb[row] = 1.0
            ub[row] = np.inf
            row += 1
            
            # sum(z) <= K
            for p in pairs[s]:
                idx = z_idx(p)
                A[row, idx] = 1.0
            lb[row] = -np.inf
            ub[row] = K
            row += 1
            
        for p in unique_pairs:
            a, b = p
            idx_z = z_idx(p)
            idx_a = x_idx(a)
            idx_b = x_idx(b)
            
            # z - x_a <= 0
            A[row, idx_z] = 1.0
            A[row, idx_a] = -1.0
            lb[row] = -np.inf
            ub[row] = 0.0
            row += 1
            
            # z - x_b <= 0
            A[row, idx_z] = 1.0
            A[row, idx_b] = -1.0
            lb[row] = -np.inf
            ub[row] = 0.0
            row += 1
            
            # z - x_a - x_b <= -1
            A[row, idx_z] = 1.0
            A[row, idx_a] = -1.0
            A[row, idx_b] = -1.0
            lb[row] = -np.inf
            ub[row] = -1.0
            row += 1
            
        # Variables bounds
        # x_i in {0, 1}
        # z_j in {0, 1}
        integrality = np.ones(num_vars)
        integrality[:] = 1 # All binary
        
        bounds = Bounds(lb=np.zeros(num_vars), ub=np.ones(num_vars))
        
        res = milp(c=c, constraints=LinearConstraint(A, lb, ub), 
                   integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            x_sol = res.x[:num_x]
            A_set = set(int(i) for i in range(num_x) if x_sol[i] > 0.5)
            return A_set
        else:
            return None

    # Try increasing K
    for K in range(1, 20):
        A_set = solve_for_K(K)
        if A_set is not None:
            # Verify
            mults = {s: 0 for s in targets}
            A_list = sorted(A_set)
            for i, a in enumerate(A_list):
                for j in range(i, len(A_list)):
                    b = A_list[j]
                    s = a + b
                    if target_low <= s <= target_high:
                        mults[s] += 1
            max_m = max(mults.values())
            if max_m <= K:
                return sorted(A_set)
                
    # Fallback to seed if ILP fails or takes too long
    return list(range(64, 129))

# EVOLVE-BLOCK-END
