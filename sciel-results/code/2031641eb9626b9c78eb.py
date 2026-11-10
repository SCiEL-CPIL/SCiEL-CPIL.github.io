import time
import random
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

def construct():
    n = 256
    deadline = time.time() + 50  # Safety margin for 90s limit (or 60s if strict)
    
    # 1. Enumerate APs
    # We focus on APs with length >= L to reduce LP size, 
    # but we must evaluate all APs for the final score.
    # For LP, we include APs with length >= 4.
    L_min_lp = 4
    
    progressions_all = []
    progressions_lp = []
    
    for step in range(1, n + 1):
        for r in range(step):
            if r >= n:
                continue
            indices = list(range(r, n, step))
            if indices:
                progressions_all.append(indices)
                if len(indices) >= L_min_lp:
                    progressions_lp.append(indices)
    
    num_progs_lp = len(progressions_lp)
    num_progs_all = len(progressions_all)
    
    # 2. Build Sparse Matrix for LP
    # Variables: x_0 ... x_{n-1} (signs), x_n (D, max discrepancy)
    # Objective: Minimize x_n
    # Constraints:
    #   -1 <= x_i <= 1  (LP relaxation of -1/+1)
    #   -D <= sum(x_j for j in AP_k) <= D for each AP_k in LP set
    
    # c = [0, ..., 0, 1]
    c = np.zeros(n + 1)
    c[-1] = 1.0
    
    # A_ub and b_ub for inequality constraints: A_ub @ x <= b_ub
    # We need:
    # 1. -x_i <= 1  => x_i >= -1
    # 2. x_i <= 1
    # 3. sum(x_j) <= D  => sum(x_j) - D <= 0
    # 4. -sum(x_j) <= D => -sum(x_j) - D <= 0
    
    # Number of constraints:
    # 2 * n (bounds on x_i)
    # 2 * num_progs_lp (bounds on AP sums)
    num_constraints = 2 * n + 2 * num_progs_lp
    
    A_ub = lil_matrix((num_constraints, n + 1), dtype=np.float64)
    b_ub = np.zeros(num_constraints)
    
    # Bounds on x_i
    for i in range(n):
        # -x_i <= 1  => -1 * x_i <= 1
        A_ub[2*i, i] = -1.0
        b_ub[2*i] = 1.0
        
        # x_i <= 1
        A_ub[2*i+1, i] = 1.0
        b_ub[2*i+1] = 1.0
        
    # Bounds on AP sums
    # Let D be index n in x.
    for k, indices in enumerate(progressions_lp):
        # sum(x_j) - D <= 0
        row_idx = 2 * n + 2 * k
        for j in indices:
            A_ub[row_idx, j] = 1.0
        A_ub[row_idx, n] = -1.0
        b_ub[row_idx] = 0.0
        
        # -sum(x_j) - D <= 0
        row_idx = 2 * n + 2 * k + 1
        for j in indices:
            A_ub[row_idx, j] = -1.0
        A_ub[row_idx, n] = -1.0
        b_ub[row_idx] = 0.0

    # Solve LP
    # bounds for x: x_i in [-1, 1], D in [0, None]
    bounds = [(-1, 1)] * n + [(0, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            lp_solution = res.x[:n]
            lp_D = res.x[n]
        else:
            lp_solution = None
            lp_D = None
    except Exception:
        lp_solution = None
        lp_D = None

    # 3. Initialize Signs
    rng = random.Random(42)
    
    if lp_solution is not None:
        # Threshold the LP solution to get initial signs
        # x_i > 0 => 1, x_i < 0 => -1
        # If close to 0, randomize or use sign of small noise? 
        # Standard rounding: sign(x_i)
        signs = np.sign(lp_solution).astype(np.int64)
        # Ensure no zeros (if any x_i == 0)
        zeros = np.where(signs == 0)[0]
        for z in zeros:
            signs[z] = rng.choice([-1, 1])
    else:
        signs = np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int64)
        
    # 4. Local Search on Discrete Signs
    # We need to evaluate max absolute sum over ALL APs (including short ones).
    # Pre-compute incidence structure for fast update?
    # For n=256, num_progs_all is ~33k.
    # Building a dense matrix M (33k x 256) is ~8.5M bytes. Fast enough.
    
    M = np.zeros((num_progs_all, n), dtype=np.int8)
    for k, indices in enumerate(progressions_all):
        M[k, indices] = 1
        
    # Current sums
    sums = M @ signs
    current_max = np.max(np.abs(sums))
    best_max = current_max
    best_signs = signs.copy()
    
    # Simulated Annealing / Local Search
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    start_time = time.time()
    iter_count = 0
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        # Pick random index to flip
        i = rng.randint(0, n - 1)
        
        old_val = signs[i]
        new_val = -old_val
        delta = new_val - old_val
        
        # Update sums
        M_col_i = M[:, i]
        sums += delta * M_col_i
        
        new_max = np.max(np.abs(sums))
        
        # Acceptance
        if new_max <= current_max:
            current_max = new_max
            signs[i] = new_val
            no_improve_count = 0
            if new_max < best_max:
                best_max = new_max
                best_signs = signs.copy()
        else:
            delta_obj = new_max - current_max
            # Accept worse move with probability exp(-delta_obj / T)
            if rng.random() < np.exp(-delta_obj / T):
                current_max = new_max
                signs[i] = new_val
                no_improve_count += 1
            else:
                # Revert
                sums -= delta * M_col_i
                no_improve_count += 1
                
        # Cooling
        T *= cooling
        if T < min_T:
            T = min_T
            
        # Restart mechanism if stuck
        if no_improve_count > max_no_improve:
            # Perturb: flip a random subset of signs to escape local minimum
            # Or just reset T
            T = 1.0
            no_improve_count = 0
            # Optional: small random perturbation
            for _ in range(5):
                j = rng.randint(0, n - 1)
                signs[j] *= -1
                delta = 2 * signs[j]
                sums += delta * M[:, j]
            current_max = np.max(np.abs(sums))
            
        iter_count += 1
        if iter_count % 1000 == 0:
             if time.time() > deadline - 2:
                 break

    return best_signs.tolist()
