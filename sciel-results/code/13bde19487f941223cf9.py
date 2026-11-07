"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = 10
MAX_VAL = 1000000
LCM_VAL = 30030

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    divisors = DIVISORS[:]
    n = len(divisors)
    
    # Precompute gcd matrix for compatibility checks
    g = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            g[i][j] = math.gcd(divisors[i], divisors[j])

    # Trivial solution
    trivial = divisors[:]
    best_R = max(trivial) - min(trivial)
    best_result = trivial[:]

    # Search for smaller R
    # We iterate R from 9 (min possible for 10 distinct ints) upwards.
    # For each R, we check if there exists a window [S, S+R] containing distinct multiples.
    
    # Precompute valid_offsets for the maximum R we might need? 
    # No, R changes. But we can compute on the fly or precompute up to a limit.
    # Since R is likely small (< 100), we can compute per R.
    
    # Optimization: Precompute for each divisor d and residue r in [0, LCM-1], 
    # the list of offsets k in [0, R_max] such that (r + k) % d == 0.
    # But R varies. Let's just compute valid_offsets for the current R in the loop.
    # Actually, we can precompute a large table for k in [0, 200] for all d, r.
    # Then for a given R, we just filter.
    
    MAX_R_PRECOMP = 200
    # valid_offsets_pre[d][r] = list of all k in [0, MAX_R_PRECOMP] such that (r+k)%d==0
    valid_offsets_pre = {}
    for d in divisors:
        valid_offsets_pre[d] = [[] for _ in range(d)]
        for r in range(d):
            target = (-r) % d
            k = target
            while k <= MAX_R_PRECOMP:
                valid_offsets_pre[d][r].append(k)
                k += d

    # CSP Solver for a fixed R
    # We want to find distinct offsets o_0..o_9 in [0, R] and a base S in [0, LCM-1]
    # such that for all i, (S + o_i) % divisors[i] == 0.
    # This is equivalent to:
    # 1. Find distinct o_i in [0, R] such that for all i, j:
    #    (S + o_i) % d_i == 0 and (S + o_j) % d_j == 0
    #    => S % d_i = (-o_i) % d_i
    #    => S % d_j = (-o_j) % d_j
    #    => S is a solution to the system of congruences.
    #    This system has a solution iff for all i, j:
    #    (-o_i) % d_i ≡ (-o_j) % d_j (mod gcd(d_i, d_j))
    #    => o_i ≡ o_j (mod gcd(d_i, d_j))
    
    # So the problem reduces to:
    # Find distinct o_0..o_9 in [0, R] such that o_i ≡ o_j (mod g[i][j]) for all i != j.
    # If such offsets exist, we can find S via CRT (or brute force S in [0, LCM-1]).
    # Since LCM is small, we can just check S.
    # But checking offsets first is a CSP on the offsets only, which is faster.
    
    def solve_offsets(R):
        # domains[i] is a set of possible offsets for variable i
        # Initially [0, R]
        # Constraints: o_i != o_j and o_i ≡ o_j (mod g[i][j])
        
        # We use backtracking with MRV and forward checking.
        
        # Precompute compatibility:
        # compat[i][v] = set of (j, u) such that u in [0, R] and (v - u) % g[i][j] == 0 and u != v?
        # No, u != v is a separate constraint.
        # Let's just use the constraint check in the loop.
        
        domains = [set(range(R + 1)) for _ in range(n)]
        assignment = [None] * n
        
        # To speed up, precompute for each i, j, and value v in [0, R],
        # the set of u in [0, R] such that (v - u) % g[i][j] == 0.
        # Since R is small, we can do this on the fly or precompute a mask.
        
        # Let's use a bitset for domains to speed up propagation?
        # Python sets are okay for R < 200.
        
        # Order variables: maybe by divisor size? Or just 0..n-1?
        # MRV will handle it.
        
        def is_compatible(i, v, j, u):
            return (v - u) % g[i][j] == 0

        def backtrack():
            # Find unassigned variable with smallest domain
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
            
            # Try values in domain of min_var
            # Order: maybe random to avoid bad local minima? Or sorted.
            # For finding ANY solution, sorted is fine.
            # But if we want to find a "good" S later, does it matter?
            # S is determined by the offsets. Any valid offsets yield a valid S.
            
            # Heuristic: Try values that are "central"? Or just sorted.
            for v in sorted(domains[min_var]):
                # Check distinctness with assigned
                if any(assignment[k] == v for k in range(n) if assignment[k] is not None):
                    continue
                
                # Check compatibility with assigned
                valid = True
                for k in range(n):
                    if assignment[k] is not None and k != min_var:
                        if not is_compatible(min_var, v, k, assignment[k]):
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Assign
                assignment[min_var] = v
                
                # Propagate
                undo_list = []
                feasible = True
                for k in range(n):
                    if assignment[k] is None and k != min_var:
                        # Remove v (distinctness)
                        # Remove incompatible values
                        new_domain = set()
                        for u in domains[k]:
                            if u == v:
                                continue
                            if is_compatible(min_var, v, k, u):
                                new_domain.add(u)
                        
                        if len(new_domain) < len(domains[k]):
                            undo_list.append((k, domains[k].copy()))
                        domains[k] = new_domain
                        
                        if len(domains[k]) == 0:
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

    # Search R from 9 upwards
    for R in range(9, 200):
        if time.time() > deadline - 5.0:
            break
        
        # Check if R is feasible using CSP on offsets
        # We need to find distinct o_i in [0, R] satisfying congruences.
        offsets = solve_offsets(R)
        
        if offsets is not None:
            # We have valid offsets. Now find S.
            # We need S such that (S + offsets[i]) % divisors[i] == 0 for all i.
            # This means S % divisors[i] == (-offsets[i]) % divisors[i].
            # We can find S by iterating through S in [0, LCM-1] and checking.
            # Or use CRT. Since LCM is small, brute force is fine.
            
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
                # Construct result
                # x_i = S + offsets[i]
                # But we need x_i >= 1.
                # If S + min(offsets) < 1, we can add LCM to S.
                min_val = S + min(offsets)
                if min_val < 1:
                    # Add k * LCM to S such that S + min(offsets) >= 1
                    # S_new = S + k * LCM
                    # S + k * LCM + min(offsets) >= 1
                    # k * LCM >= 1 - S - min(offsets)
                    k = math.ceil((1 - S - min(offsets)) / LCM_VAL)
                    if k < 0:
                        k = 0
                    S += k * LCM_VAL
                
                result = [S + offsets[i] for i in range(n)]
                
                # Check bounds
                if all(0 < val <= MAX_VAL for val in result):
                    actual_R = max(result) - min(result)
                    if actual_R < best_R:
                        best_R = actual_R
                        best_result = result[:]
                    # Since we iterate R upwards, the first R that works gives the optimal span
                    # (as argued before, if span < R, it would have been found at smaller R)
                    # So we can break
                    break
        else:
            # No solution for this R
            continue
    
    if best_result is None:
        best_result = divisors[:]
    
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
