"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    n = 128
    # Variables: x_0, ..., x_128 (binary), t (continuous, peak multiplicity)
    # x_i = 1 if i is in A
    # For each target s in [128, 256], count pairs (a,b) with a<=b, a,b in A, a+b=s
    # We need count >= 1 for all s, and minimize t = max_s count_s
    
    # Precompute pair indices for each target
    # For target s, pairs are (a, s-a) where a <= s-a, a in [0,128], s-a in [0,128]
    # i.e., a in [max(0, s-128), min(128, s//2)]
    
    targets = list(range(128, 257))
    num_targets = len(targets)
    
    # For each target, list of (a, b) pairs with a <= b
    target_pairs = {}
    for s in targets:
        pairs = []
        a_min = max(0, s - 128)
        a_max = min(128, s // 2)
        for a in range(a_min, a_max + 1):
            b = s - a
            if b >= a and b <= 128:
                pairs.append((a, b))
        target_pairs[s] = pairs
    
    # ILP formulation:
    # Variables: x_0..x_128 (binary), t (continuous >= 1)
    # Minimize: t
    # Subject to:
    #   For each s in [128,256]: sum_{(a,b) in pairs(s)} y_{a,b} >= 1
    #   where y_{a,b} = x_a * x_b if a != b, and y_{a,a} = x_a
    #   But y_{a,b} for a!=b is product of two binaries -> need linearization
    #   
    # For a != b: y_{a,b} <= x_a, y_{a,b} <= x_b, y_{a,b} >= x_a + x_b - 1, y_{a,b} >= 0
    # For a == b: y_{a,a} = x_a
    #
    # Then: sum of y's for target s >= 1
    # And: sum of y's for target s <= t for all s
    
    # This is getting complex. Let me use a different approach:
    # Binary search on peak value K, check feasibility with ILP
    
    def check_feasible(K):
        """Check if there exists A with peak multiplicity <= K"""
        # Variables: x_0..x_128 (binary), y_{a,b} for all a<=b pairs (binary or continuous)
        # For a == b: y_{a,a} = x_a (just use x_a)
        # For a < b: y_{a,b} is binary, y_{a,b} <= x_a, y_{a,b} <= x_b, y_{a,b} >= x_a + x_b - 1
        
        # Count variables
        # x: 129 variables (indices 0..128)
        # y: for each pair (a,b) with a < b, we need a y variable
        # Let's collect all distinct pairs (a,b) with a < b where a+b is in [128,256]
        
        all_pairs = set()
        for s in targets:
            for (a, b) in target_pairs[s]:
                if a < b:
                    all_pairs.add((a, b))
        
        pair_list = sorted(all_pairs)
        num_pairs = len(pair_list)
        pair_index = {p: i for i, p in enumerate(pair_list)}
        
        # Total variables: 129 (x) + num_pairs (y)
        num_vars = 129 + num_pairs
        
        # Objective: minimize 0 (feasibility check)
        c = np.zeros(num_vars)
        
        # Constraints:
        # 1. For each target s: sum of relevant y's + sum of x_a for a==a pairs >= 1
        # 2. For each target s: sum of relevant y's + sum of x_a for a==a pairs <= K
        # 3. For each pair (a,b) with a<b: y_{a,b} <= x_a
        # 4. For each pair (a,b) with a<b: y_{a,b} <= x_b
        # 5. For each pair (a,b) with a<b: y_{a,b} >= x_a + x_b - 1
        
        constraints = []
        
        # Build constraint matrix rows
        A_ub = []
        b_ub = []
        A_eq = []
        b_eq = []
        
        # For each target s: lower bound constraint (sum >= 1) -> -sum <= -1
        for s in targets:
            row = np.zeros(num_vars)
            for (a, b) in target_pairs[s]:
                if a == b:
                    row[a] = -1  # -x_a
                else:
                    idx = pair_index[(a, b)]
                    row[129 + idx] = -1  # -y_{a,b}
            A_ub.append(row)
            b_ub.append(-1)
        
        # For each target s: upper bound constraint (sum <= K)
        for s in targets:
            row = np.zeros(num_vars)
            for (a, b) in target_pairs[s]:
                if a == b:
                    row[a] = 1  # x_a
                else:
                    idx = pair_index[(a, b)]
                    row[129 + idx] = 1  # y_{a,b}
            A_ub.append(row)
            b_ub.append(K)
        
        # For each pair (a,b) with a<b: y_{a,b} - x_a <= 0
        for (a, b) in pair_list:
            row = np.zeros(num_vars)
            row[a] = -1
            row[129 + pair_index[(a, b)]] = 1
            A_ub.append(row)
            b_ub.append(0)
        
        # For each pair (a,b) with a<b: y_{a,b} - x_b <= 0
        for (a, b) in pair_list:
            row = np.zeros(num_vars)
            row[b] = -1
            row[129 + pair_index[(a, b)]] = 1
            A_ub.append(row)
            b_ub.append(0)
        
        # For each pair (a,b) with a<b: -y_{a,b} + x_a + x_b >= -1 -> y_{a,b} - x_a - x_b <= 1
        for (a, b) in pair_list:
            row = np.zeros(num_vars)
            row[a] = -1
            row[b] = -1
            row[129 + pair_index[(a, b)]] = 1
            A_ub.append(row)
            b_ub.append(1)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        constraint = LinearConstraint(A_ub, -np.inf, b_ub)
        
        # Integrality: x variables are binary, y variables are binary
        integrality = np.zeros(num_vars)
        integrality[:129] = 1  # x_0..x_128 are binary
        integrality[129:] = 1  # y variables are binary
        
        bounds = Bounds(0, 1)
        
        result = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        return result.status == 0
    
    # Binary search on K
    # Lower bound: 1 (minimum possible)
    # Upper bound: current best is 5.5, so try up to 6
    lo = 1
    hi = 6  # we know 5.5 works, so 6 should work
    
    # First check if K=1 is feasible (unlikely)
    # Binary search for minimum feasible K
    
    best_K = None
    for K in [1, 2, 3, 4, 5, 6]:
        if time.time() > deadline:
            break
        feasible = check_feasible(K)
        if feasible:
            best_K = K
            break
    
    if best_K is None:
        # Fallback to known good solution
        return list(range(64, 129))
    
    # Now find the actual set A
    def find_set(K):
        """Find a feasible set with peak <= K"""
        all_pairs = set()
        for s in targets:
            for (a, b) in target_pairs[s]:
                if a < b:
                    all_pairs.add((a, b))
        
        pair_list = sorted(all_pairs)
        num_pairs = len(pair_list)
        pair_index = {p: i for i, p in enumerate(pair_list)}
        
        num_vars = 129 + num_pairs
        
        c = np.zeros(num_vars)
        
        A_ub = []
        b_ub = []
        
        for s in targets:
            row = np.zeros(num_vars)
            for (a, b) in target_pairs[s]:
                if a == b:
                    row[a] = -1
                else:
                    idx = pair_index[(a, b)]
                    row[129 + idx] = -1
            A_ub.append(row)
            b_ub.append(-1)
        
        for s in targets:
            row = np.zeros(num_vars)
            for (a, b) in target_pairs[s]:
                if a == b:
                    row[a] = 1
                else:
                    idx = pair_index[(a, b)]
                    row[129 + idx] = 1
            A_ub.append(row)
            b_ub.append(K)
        
        for (a, b) in pair_list:
            row = np.zeros(num_vars)
            row[a] = -1
            row[129 + pair_index[(a, b)]] = 1
            A_ub.append(row)
            b_ub.append(0)
        
        for (a, b) in pair_list:
            row = np.zeros(num_vars)
            row[b] = -1
            row[129 + pair_index[(a, b)]] = 1
            A_ub.append(row)
            b_ub.append(0)
        
        for (a, b) in pair_list:
            row = np.zeros(num_vars)
            row[a] = -1
            row[b] = -1
            row[129 + pair_index[(a, b)]] = 1
            A_ub.append(row)
            b_ub.append(1)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        constraint = LinearConstraint(A_ub, -np.inf, b_ub)
        
        integrality = np.zeros(num_vars)
        integrality[:129] = 1
        integrality[129:] = 1
        
        bounds = Bounds(0, 1)
        
        result = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if result.status == 0:
            x = result.x[:129]
            A = [i for i in range(129) if x[i] > 0.5]
            return A
        return None
    
    if time.time() < deadline:
        A = find_set(best_K)
        if A is not None and len(A) > 0:
            return A
    
    # Fallback
    return list(range(64, 129))
# EVOLVE-BLOCK-END
