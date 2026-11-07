"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = 10
MAX_VAL = 1000000
LCM_VAL = 30030

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    divisors = DIVISORS[:]
    n = len(divisors)
    
    # Trivial solution: the divisors themselves
    trivial = divisors[:]
    best_R = max(trivial) - min(trivial)
    best_result = trivial[:]
    
    # Search for smaller span R
    # For a given R, we want to find distinct offsets o_0..o_9 in [0, R]
    # and a base S in [0, LCM_VAL) such that (S + o_i) % divisors[i] == 0.
    # This is equivalent to S ≡ (-o_i) (mod divisors[i]) for all i.
    # Such an S exists iff (-o_i) ≡ (-o_j) (mod gcd(divisors[i], divisors[j])) for all i != j,
    # i.e., o_i ≡ o_j (mod gcd(divisors[i], divisors[j])).
    
    # Precompute gcd matrix
    g = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            g[i][j] = math.gcd(divisors[i], divisors[j])
    
    # For each R from 9 upwards, check feasibility
    # We use backtracking with forward checking on the offsets.
    
    def solve_offsets(R):
        # domains[i] = set of possible offsets for variable i
        # Initially [0, R]
        # Constraint: o_i != o_j and o_i ≡ o_j (mod g[i][j])
        
        domains = [set(range(R + 1)) for _ in range(n)]
        assignment = [None] * n
        
        def backtrack():
            # Find unassigned variable with smallest domain (MRV)
            min_size = float('inf')
            min_var = -1
            for k in range(n):
                if assignment[k] is None:
                    size = len(domains[k])
                    if size < min_size:
                        min_size = size
                        min_var = k
                        if size == 0:
                            return None
            
            if min_var == -1:
                return assignment[:]
            
            # Try values in domain
            for v in sorted(domains[min_var]):
                # Check distinctness with assigned
                if any(assignment[k] == v for k in range(n) if assignment[k] is not None):
                    continue
                
                # Check compatibility with assigned
                valid = True
                for k in range(n):
                    if assignment[k] is not None and k != min_var:
                        if (v - assignment[k]) % g[min_var][k] != 0:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Assign and propagate
                assignment[min_var] = v
                undo_list = []
                feasible = True
                for k in range(n):
                    if assignment[k] is None and k != min_var:
                        new_domain = set()
                        for u in domains[k]:
                            if u == v:
                                continue
                            if (v - u) % g[min_var][k] == 0:
                                new_domain.add(u)
                        
                        if len(new_domain) < len(domains[k]):
                            undo_list.append((k, domains[k].copy()))
                        domains[k] = new_domain
                        
                        if len(new_domain) == 0:
                            feasible = False
                            break
                
                if feasible:
                    res = backtrack()
                    if res is not None:
                        return res
                
                # Undo
                assignment[min_var] = None
                for k, old_domain in undo_list:
                    domains[k] = old_domain
                    
            return None
        
        return backtrack()
    
    # Search R from 9 (minimum possible for 10 distinct integers) upwards
    for R in range(9, 200):
        if time.time() > deadline - 5.0:
            break
        
        offsets = solve_offsets(R)
        
        if offsets is not None:
            # Find S in [0, LCM_VAL) such that (S + offsets[i]) % divisors[i] == 0
            found_S = False
            for S in range(LCM_VAL):
                valid = True
                for i in range(n):
                    if (S + offsets[i]) % divisors[i] != 0:
                        valid = False
                        break
                if valid:
                    found_S = True
                    break
            
            if found_S:
                result = [S + offsets[i] for i in range(n)]
                
                # Ensure all values are >= 1
                min_val = min(result)
                if min_val < 1:
                    k = math.ceil((1 - min_val) / LCM_VAL)
                    if k < 0:
                        k = 0
                    for i in range(n):
                        result[i] += k * LCM_VAL
                
                # Check upper bound
                if all(0 < val <= MAX_VAL for val in result):
                    actual_R = max(result) - min(result)
                    if actual_R < best_R:
                        best_R = actual_R
                        best_result = result[:]
                    break  # First valid R is optimal
    
    if best_result is None:
        best_result = divisors[:]
    
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
