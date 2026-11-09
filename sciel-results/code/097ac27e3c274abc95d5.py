import numpy as np
import time
import scipy.sparse
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    t0 = time.time()
    N = 12
    M = 40
    total = 1 << N  # 4096

    # Generate comparable pairs (A, B) where A is a strict subset of B
    pairs = []
    for B in range(total):
        if B != 0:
            pairs.append((0, B))
        s = B
        while s:
            if s != B:
                pairs.append((s, B))
            s = (s - 1) & B
            
    num_pairs = len(pairs)
    
    # Variables:
    # 0 ... total-1 : x_S (binary)
    # total ... total+num_pairs-1 : z_{A,B} (continuous, will be integral if x is)
    
    n_vars = total + num_pairs
    
    # Objective: Maximize sum z  =>  Minimize -sum z
    c = np.zeros(n_vars)
    c[total:total+num_pairs] = -1.0
    
    # Inequality Constraints:
    # 1. z_{A,B} - x_A <= 0
    # 2. z_{A,B} - x_B <= 0
    # 3. x_A + x_B - z_{A,B} <= 1
    
    r_list = []
    c_list = []
    v_list = []
    b_list = []
    
    for i, (A, B) in enumerate(pairs):
        z_idx = total + i
        
        # Row 3*i: z - x_A <= 0
        r_list.append(3*i)
        c_list.append(z_idx)
        v_list.append(1.0)
        c_list.append(A)
        v_list.append(-1.0)
        b_list.append(0.0)
        
        # Row 3*i+1: z - x_B <= 0
        r_list.append(3*i+1)
        c_list.append(z_idx)
        v_list.append(1.0)
        c_list.append(B)
        v_list.append(-1.0)
        b_list.append(0.0)
        
        # Row 3*i+2: x_A + x_B - z <= 1
        r_list.append(3*i+2)
        c_list.append(A)
        v_list.append(1.0)
        c_list.append(B)
        v_list.append(1.0)
        c_list.append(z_idx)
        v_list.append(-1.0)
        b_list.append(1.0)
        
    A_ub = scipy.sparse.csr_matrix((v_list, (r_list, c_list)), shape=(3*num_pairs, n_vars))
    b_ub = np.array(b_list)
    
    # Equality Constraint: sum x_S = 40
    A_eq = scipy.sparse.csr_matrix((np.ones(total), (np.zeros(total, dtype=np.int32), np.arange(total))), shape=(1, n_vars))
    b_eq = np.array([40])
    
    bounds = Bounds(lb=0, ub=1)
    
    integrality = np.zeros(n_vars)
    integrality[:total] = 1 # x variables are integer
    
    constraints = [
        LinearConstraint(A_ub, -np.inf, b_ub),
        LinearConstraint(A_eq, b_eq, b_eq)
    ]
    
    # Solve with time limit
    # If it times out, res.status might not be 0, but res.x might be a feasible point
    res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, options={'time_limit': 55})
    
    selected = []
    if res.x is not None:
        x_sol = res.x[:total]
        # Round to nearest integer, handle floating point errors
        x_bin = (x_sol > 0.5).astype(int)
        
        # Check if we have exactly 40
        count = np.sum(x_bin)
        if count == M:
            selected = [int(i) for i in np.where(x_bin == 1)[0]]
        else:
            # If not exactly 40 (e.g. due to infeasibility or rounding issues),
            # we need to fix it.
            # Get the indices with x > 0.5
            candidates = [int(i) for i in np.where(x_sol > 0.5)[0]]
            
            if len(candidates) > M:
                # Keep the ones with highest x values? Or just first M?
                # To maximize score, we should keep the ones that form the most pairs.
                # But we don't have the score easily here.
                # Just take first M or random?
                # Taking first M is deterministic.
                selected = candidates[:M]
            elif len(candidates) < M:
                # Add random missing ones
                missing = [i for i in range(total) if i not in set(candidates)]
                import random
                random.seed(42)
                random.shuffle(missing)
                selected = candidates + missing[:M - len(candidates)]
            else:
                selected = candidates
    else:
        # Fallback: Random selection
        import random
        random.seed(42)
        selected = random.sample(range(total), M)

    # Convert to lists
    result = []
    for mask in selected:
        result.append([b for b in range(N) if (mask >> b) & 1])

    return result
