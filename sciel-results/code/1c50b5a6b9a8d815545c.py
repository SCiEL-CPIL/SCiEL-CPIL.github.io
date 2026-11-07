import math
import time
import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = 10
    max_val = 1000000
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Try MILP if available
    if SCIPY_AVAILABLE:
        try:
            num_k = n
            idx_m = num_k
            idx_M = num_k + 1
            idx_z_start = num_k + 2
            
            num_pairs = n * (n - 1) // 2
            total_vars = idx_z_start + num_pairs
            
            c = [0.0] * total_vars
            c[idx_m] = -1.0
            c[idx_M] = 1.0
            
            lb = [0.0] * total_vars
            ub = [0.0] * total_vars
            
            for i in range(n):
                lb[i] = 1.0
                ub[i] = max_val / divisors[i]
            
            lb[idx_m] = 1.0
            ub[idx_m] = max_val
            lb[idx_M] = 1.0
            ub[idx_M] = max_val
            
            for j in range(num_pairs):
                lb[idx_z_start + j] = 0.0
                ub[idx_z_start + j] = 1.0
            
            integrality = [1] * total_vars
            
            M_big = float(max_val)
            
            rows = []
            rhs = []
            
            for i in range(n):
                # m <= k_i d_i
                row = [0.0] * total_vars
                row[i] = -divisors[i]
                row[idx_m] = 1.0
                rows.append(row)
                rhs.append(0.0)
                
                # k_i d_i <= M
                row = [0.0] * total_vars
                row[i] = divisors[i]
                row[idx_M] = -1.0
                rows.append(row)
                rhs.append(0.0)
            
            pair_idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    z_idx = idx_z_start + pair_idx
                    pair_idx += 1
                    
                    # x_i - x_j <= -1 + M_big (1 - z)
                    row = [0.0] * total_vars
                    row[i] = divisors[i]
                    row[j] = -divisors[j]
                    row[z_idx] = M_big
                    rows.append(row)
                    rhs.append(-1.0 + M_big)
                    
                    # x_j - x_i <= -1 + M_big z
                    row = [0.0] * total_vars
                    row[i] = -divisors[i]
                    row[j] = divisors[j]
                    row[z_idx] = M_big
                    rows.append(row)
                    rhs.append(-1.0 + M_big)
                    
            A = np.array(rows)
            b = np.array(rhs)
            
            constraints = LinearConstraint(A, -np.inf, b)
            bounds = Bounds(lb, ub)
            
            # Set time limit for MILP to be safe
            options = {"time_limit": 40}
            res = milp(c, constraints=constraints, bounds=bounds, integrality=integrality, options=options)
            
            if res.status == 0:
                k_vals = res.x[:n]
                result = [int(k * d) for k, d in zip(k_vals, divisors)]
                
                # Verify
                if (len(set(result)) == n and 
                    all(1 <= x <= max_val for x in result) and 
                    all(result[i] % divisors[i] == 0 for i in range(n))):
                    return result
        except Exception:
            pass
            
    # Fallback: Combinatorial Search
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in divisors:
        L = lcm(L, d)
        
    def find_distinct_assignment(candidates):
        indices = sorted(range(n), key=lambda i: len(candidates[i]))
        chosen = [None] * n
        used = set()
        
        def backtrack(idx):
            if idx == n:
                return True
            i = indices[idx]
            for val in candidates[i]:
                if val not in used:
                    used.add(val)
                    chosen[i] = val
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
                    chosen[i] = None
            return False
            
        if backtrack(0):
            return chosen
        return None

    for S in range(9, 500):
        if time.time() > deadline:
            break
            
        for m in range(L):
            if time.time() > deadline:
                break
                
            candidates = []
            valid = True
            for d in divisors:
                min_k = (m + d - 1) // d
                max_k = (m + S) // d
                if min_k > max_k:
                    valid = False
                    break
                cands = [k * d for k in range(min_k, max_k + 1)]
                candidates.append(cands)
                
            if not valid:
                continue
                
            assignment = find_distinct_assignment(candidates)
            if assignment is None:
                continue
                
            min_x = min(assignment)
            max_x = max(assignment)
            
            k_min = math.ceil((1 - min_x) / L)
            k_max = math.floor((max_val - max_x) / L)
            
            if k_min <= k_max:
                k = k_min
                result = [x + k * L for x in assignment]
                return result
                
    return divisors[:]
