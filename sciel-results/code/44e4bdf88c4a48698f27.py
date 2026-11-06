import time
import random
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct():
    start_time = time.time()
    limit = 80.0  # Safety margin for 90s hard kill
    
    n = 256
    weights = np.zeros(n + 1)
    for i in range(1, n + 1):
        weights[i] = 1.0 / i
    
    # Precompute all 3-APs: (a, b, c) where a < b < c and a+c = 2b
    ap_constraints = []
    for b in range(1, n + 1):
        for a in range(1, b):
            c = 2 * b - a
            if c <= n and c > b:
                ap_constraints.append((a, b, c))
    
    num_constraints = len(ap_constraints)
    
    # Build sparse constraint matrix A for MILP
    # Each row corresponds to a 3-AP constraint: x[a] + x[b] + x[c] <= 2
    rows = []
    cols = []
    vals = []
    
    for idx, (a, b, c) in enumerate(ap_constraints):
        rows.extend([idx, idx, idx])
        cols.extend([a - 1, b - 1, c - 1])
        vals.extend([1, 1, 1])
    
    A = csr_matrix((vals, (rows, cols)), shape=(num_constraints, n))
    
    # MILP: maximize sum(w_i * x_i) subject to A @ x <= 2, x in {0,1}
    # scipy milp minimizes, so we negate the objective
    c = -weights[1:]  # negate for minimization
    
    # Constraints: A @ x <= 2
    A_ub = A
    b_ub = np.ones(num_constraints) * 2
    
    # Variable bounds: 0 <= x_i <= 1
    lb = np.zeros(n)
    ub = np.ones(n)
    
    # Integrality: all binary
    integrality = np.ones(n)
    
    # Run MILP
    try:
        res = milp(
            c=c,
            constraints=[LinearConstraint(A_ub, -np.inf, b_ub)],
            integrality=integrality,
            bounds=Bounds(lb, ub),
            options={"time_limit": 40, "mip_rel_gap": 0.01}
        )
        
        if res.status == 0 and res.x is not None:
            milp_solution = set()
            for i in range(n):
                if res.x[i] > 0.5:
                    milp_solution.add(i + 1)
        else:
            # Fallback to greedy if MILP fails
            milp_solution = set()
            for i in range(1, n + 1):
                milp_solution.add(i)
                # This is just a placeholder; we'll refine below
            # Actually let's do a proper greedy fallback
            milp_solution = set()
            # Simple greedy: add smallest numbers first (highest weight)
            for i in range(1, n + 1):
                # Check if adding i creates a 3-AP with existing elements
                valid = True
                for j in milp_solution:
                    # Check if {i, j, k} forms a 3-AP for some k
                    # Case 1: i is smallest, j is middle, k = 2j - i
                    k = 2 * j - i
                    if k > j and k <= n and k in milp_solution:
                        valid = False
                        break
                    # Case 2: i is smallest, j is largest, middle = (i+j)/2
                    if (i + j) % 2 == 0:
                        mid = (i + j) // 2
                        if i < mid < j and mid in milp_solution:
                            valid = False
                            break
                    # Case 3: j is smallest, i is middle, k = 2i - j
                    k = 2 * i - j
                    if k > i and k <= n and k in milp_solution:
                        valid = False
                        break
                    # Case 4: j is smallest, i is largest, middle = (j+i)/2
                    if (j + i) % 2 == 0:
                        mid = (j + i) // 2
                        if j < mid < i and mid in milp_solution:
                            valid = False
                            break
                if valid:
                    milp_solution.add(i)
    except Exception:
        # Fallback greedy
        milp_solution = set()
        for i in range(1, n + 1):
            valid = True
            for j in milp_solution:
                k1 = 2 * j - i
                if k1 > j and k1 <= n and k1 in milp_solution:
                    valid = False
                    break
                if (i + j) % 2 == 0:
                    mid = (i + j) // 2
                    if i < mid < j and mid in milp_solution:
                        valid = False
                        break
                k2 = 2 * i - j
                if k2 > i and k2 <= n and k2 in milp_solution:
                    valid = False
                    break
                if (j + i) % 2 == 0:
                    mid = (j + i) // 2
                    if j < mid < i and mid in milp_solution:
                        valid = False
                        break
            if valid:
                milp_solution.add(i)
    
    # Local search refinement
    S = set(milp_solution)
    
    # Precompute conflict masks for fast local search
    conflict_mask = [0] * (n + 1)
    for x in range(1, n + 1):
        mask = 0
        for y in range(1, n + 1):
            if x == y:
                continue
            # Check if x and y are in a 3-AP together
            # They conflict if there exists z such that {x,y,z} is a 3-AP
            # Case 1: x, y are ends: z = (x+y)/2
            if (x + y) % 2 == 0:
                z = (x + y) // 2
                if 1 <= z <= n and z != x and z != y:
                    mask |= (1 << y)
                    continue
            # Case 2: x is middle: z = 2x - y
            z = 2 * x - y
            if 1 <= z <= n and z != x and z != y:
                mask |= (1 << y)
                continue
            # Case 3: y is middle: z = 2y - x
            z = 2 * y - x
            if 1 <= z <= n and z != x and z != y:
                mask |= (1 << y)
                continue
        conflict_mask[x] = mask
    
    selected_mask = 0
    for x in S:
        selected_mask |= (1 << x)
    
    current_score = sum(1.0 / x for x in S)
    best_score = current_score
    best_S = set(S)
    
    all_nodes = list(range(1, n + 1))
    
    def can_add(v):
        return (conflict_mask[v] & selected_mask) == 0
    
    # Local search: try adds, swaps, and 2-swaps
    T = 0.02
    cooling_rate = 0.9995
    min_T = 0.0005
    
    while time.time() - start_time < limit:
        improved = False
        
        # Try to add nodes (always beneficial if possible)
        for _ in range(10):
            v = random.choice(all_nodes)
            if not (selected_mask & (1 << v)) and can_add(v):
                S.add(v)
                selected_mask |= (1 << v)
                current_score += 1.0 / v
                improved = True
                break
        
        if improved:
            if current_score > best_score:
                best_score = current_score
                best_S = set(S)
            continue
        
        # Try swaps
        if len(S) == 0:
            break
            
        u = random.choice(list(S))
        v = random.choice(all_nodes)
        
        if v == u:
            continue
            
        temp_mask = selected_mask ^ (1 << u)
        
        if (conflict_mask[v] & temp_mask) == 0:
            delta = 1.0 / v - 1.0 / u
            if delta >= 0 or random.random() < math.exp(delta / T):
                S.remove(u)
                S.add(v)
                selected_mask = temp_mask | (1 << v)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(S)
                T *= 0.9995
            else:
                T *= cooling_rate
        else:
            T *= cooling_rate
        
        # Occasionally try 2-swap: remove two, add two
        if random.random() < 0.05 and len(S) >= 2:
            u1, u2 = random.sample(list(S), 2)
            v1 = random.choice(all_nodes)
            v2 = random.choice(all_nodes)
            
            if v1 in S or v2 in S or v1 == v2:
                continue
                
            temp_mask = selected_mask & ~(1 << u1) & ~(1 << u2)
            
            if ((conflict_mask[v1] & temp_mask) == 0 and 
                (conflict_mask[v2] & temp_mask) == 0 and
                (conflict_mask[v1] & (1 << v2)) == 0):
                delta = (1.0 / v1 + 1.0 / v2) - (1.0 / u1 + 1.0 / u2)
                if delta >= 0 or random.random() < math.exp(delta / T):
                    S.discard(u1)
                    S.discard(u2)
                    S.add(v1)
                    S.add(v2)
                    selected_mask = temp_mask | (1 << v1) | (1 << v2)
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(S)
                    T *= 0.9995
                else:
                    T *= cooling_rate
            else:
                T *= cooling_rate
    
    if not best_S:
        best_S = {1}
    
    return list(best_S)
