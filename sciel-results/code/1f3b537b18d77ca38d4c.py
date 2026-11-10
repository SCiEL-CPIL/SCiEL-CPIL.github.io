import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 256
    all_nums = list(range(1, N + 1))
    recip = [0.0] + [1.0 / i for i in range(1, N + 1)]
    
    # Precompute conflicts for fast can_add check
    # For each x, store a list of (y, z) pairs such that {x, y, z} is a 3-AP
    # We need to check if y and z are both in S.
    # We will use bitmasks for S.
    
    # conflict_pairs[x] is a list of (y, z)
    conflict_pairs = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            # b is the middle term
            c = 2 * b - a
            if c > N or c <= b:
                continue
            # We have a valid 3-AP (a, b, c) with a < b < c
            # For a: if b and c are in S, we cannot add a
            conflict_pairs[a].append((b, c))
            # For b: if a and c are in S, we cannot add b
            conflict_pairs[b].append((a, c))
            # For c: if a and b are in S, we cannot add c
            conflict_pairs[c].append((a, b))
    
    def can_add_bitset(x, S_mask):
        """Check if x can be added to set represented by S_mask."""
        for y, z in conflict_pairs[x]:
            # Check if y and z are both in S_mask
            if (S_mask & (1 << y)) and (S_mask & (1 << z)):
                return False
        return True
    
    def get_float_sum_fast(S_mask):
        """Compute floating-point reciprocal sum from bitmask using numpy for speed."""
        mask_arr = np.array([(S_mask >> i) & 1 for i in range(1, N + 1)], dtype=np.float64)
        return float(np.dot(mask_arr, recip[1:]))

    def get_float_sum(S_mask):
        """Compute floating-point reciprocal sum from bitmask."""
        total = 0.0
        for x in range(1, N + 1):
            if S_mask & (1 << x):
                total += recip[x]
        return total

    def greedy_fill(forbidden_mask, S_mask):
        """Fill S by adding elements in ascending order."""
        S = S_mask
        for x in all_nums:
            if forbidden_mask & (1 << x):
                continue
            if S & (1 << x):
                continue
            if can_add_bitset(x, S):
                S |= (1 << x)
        return S

    def greedy_fill_descending(forbidden_mask, S_mask):
        """Fill S by adding elements in descending order."""
        S = S_mask
        for x in reversed(all_nums):
            if forbidden_mask & (1 << x):
                continue
            if S & (1 << x):
                continue
            if can_add_bitset(x, S):
                S |= (1 << x)
        return S

    def local_search(S_mask, max_time):
        """Run local search starting from S_mask, return best mask found."""
        S = S_mask
        sum_float = get_float_sum(S)
        
        # Precompute list of elements in S for faster iteration
        S_elements = [x for x in range(1, N + 1) if S & (1 << x)]
        
        improved = True
        while improved and (time.time() - start_time) < max_time:
            improved = False
            
            # Strategy 1: 1-Swap - try replacing x in S with y not in S
            best_gain = 0
            best_x = -1
            best_y = -1
            
            for y in range(1, N + 1):
                if (time.time() - start_time) > max_time:
                    break
                if S & (1 << y):
                    continue
                
                for x in S_elements:
                    gain = recip[y] - recip[x]
                    if gain <= 1e-15:
                        continue
                    
                    S_new = S & ~(1 << x)
                    if can_add_bitset(y, S_new):
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                            best_y = y
            
            if best_y != -1:
                S = S & ~(1 << best_x)
                S |= (1 << best_y)
                sum_float += best_gain
                S_elements.remove(best_x)
                S_elements.append(best_y)
                improved = True
                continue
            
            # Strategy 2: 1-Remove and re-greedy
            best_remove_gain = -1
            best_remove_S = None
            
            for x in S_elements[:]:  # Iterate over copy since we might modify
                if (time.time() - start_time) > max_time:
                    break
                
                S_no_x = S & ~(1 << x)
                S_new = greedy_fill(1 << x, S_no_x)
                sum_new = get_float_sum(S_new)
                gain = sum_new - sum_float
                
                if gain > 1e-15 and gain > best_remove_gain:
                    best_remove_gain = gain
                    best_remove_S = S_new
            
            if best_remove_S is not None:
                S = best_remove_S
                sum_float += best_remove_gain
                S_elements = [x for x in range(1, N + 1) if S & (1 << x)]
                improved = True
                continue
            
            # Strategy 3: Try adding a small element
            for y in range(1, N + 1):
                if (time.time() - start_time) > max_time:
                    break
                if S & (1 << y):
                    continue
                if can_add_bitset(y, S):
                    S |= (1 << y)
                    sum_float += recip[y]
                    S_elements.append(y)
                    improved = True
                    break
            
            if not improved:
                # Random perturbation
                if (time.time() - start_time) > max_time:
                    break
                if not S_elements:
                    break
                x = random.choice(S_elements)
                S_no_x = S & ~(1 << x)
                S_new = greedy_fill(1 << x, S_no_x)
                sum_new = get_float_sum(S_new)
                if sum_new > sum_float:
                    S = S_new
                    sum_float = sum_new
                    S_elements = [x for x in range(1, N + 1) if S & (1 << x)]
                    improved = True
        
        return S, sum_float

    # Step 1: Initializations
    initializations = []
    
    # Greedy ascending
    S_asc = greedy_fill(0, 0)
    initializations.append(S_asc)
    
    # Greedy descending
    S_desc = greedy_fill_descending(0, 0)
    initializations.append(S_desc)
    
    # Randomized greedy
    for trial in range(15):
        if (time.time() - start_time) > deadline - 25:
            break
        random.seed(trial * 12345 + 67890)
        S_rand = 0
        for x in all_nums:
            if random.random() < 0.15 and x > 5:
                continue
            if can_add_bitset(x, S_rand):
                S_rand |= (1 << x)
        initializations.append(S_rand)
    
    # Step 2: Try ILP solver if time permits
    ilp_result = None
    if (time.time() - start_time) < deadline - 30:
        try:
            ilp_result = solve_ilp(start_time, deadline)
        except Exception:
            ilp_result = None
    
    # Step 3: Local Search on all initializations
    best_S = 0
    best_sum = 0.0
    
    # Add ILP result to initializations if we got one
    if ilp_result is not None:
        initializations.append(ilp_result)
    
    time_per_ls = max(3.0, (deadline - 10 - (time.time() - start_time)) / max(1, len(initializations)))
    
    for i, S_init in enumerate(initializations):
        if (time.time() - start_time) > deadline - 5:
            break
        
        remaining_time = deadline - 5 - (time.time() - start_time)
        if remaining_time < 3:
            break
        
        S_ls, sum_ls = local_search(S_init, min(time_per_ls, remaining_time))
        
        if sum_ls > best_sum:
            best_sum = sum_ls
            best_S = S_ls
    
    # Step 4: Final Polish
    if best_S != 0:
        remaining = deadline - 5 - (time.time() - start_time)
        if remaining > 5:
            S_final, sum_final = local_search(best_S, remaining)
            if sum_final > best_sum:
                best_S = S_final
                best_sum = sum_final
    
    # Convert bitmask to list
    result = [x for x in range(1, N + 1) if best_S & (1 << x)]
    
    # Validate
    S_set = set(result)
    valid = True
    for a in S_set:
        for b in S_set:
            if a >= b:
                continue
            s = a + b
            if s % 2 == 0:
                mid = s // 2
                if mid in S_set and mid != a and mid != b:
                    valid = False
                    break
        if not valid:
            break
            
    if not valid:
        result = [x for x in range(1, N + 1) if S_asc & (1 << x)]
        
    return result

def solve_ilp(start_time, deadline):
    """Solve the ILP formulation using scipy.optimize.milp."""
    N = 256
    recip = [0.0] + [1.0 / i for i in range(1, N + 1)]
    
    # Decision variables: x_i = 1 if i is in the set, for i = 1..N
    # Objective: maximize sum(i=1 to N) recip[i] * x_i
    # Constraints: For each 3-AP (a, b, c), x_a + x_b + x_c <= 2
    
    # Generate all 3-APs
    aps = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            c = 2 * b - a
            if c > N:
                break
            if c > b:
                aps.append((a, b, c))
    
    m = len(aps)
    
    # c: objective coefficients (negate for minimization)
    c = np.array([-recip[i] for i in range(1, N + 1)])
    
    # A: constraint matrix
    A = np.zeros((m, N))
    for idx, (a, b, c_val) in enumerate(aps):
        A[idx, a - 1] = 1
        A[idx, b - 1] = 1
        A[idx, c_val - 1] = 1
    
    # Constraints: A @ x <= 2 for each row
    constraint = LinearConstraint(A, ub=2)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are integer
    integrality = np.ones(N)
    
    # Time limit in seconds
    time_limit = max(1.0, min(20.0, deadline - 5 - (time.time() - start_time)))
    
    try:
        res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds,
                   options={'time_limit': time_limit, 'presolve': True})
        
        if res.status == 0 and res.x is not None:
            # Convert to bitmask
            mask = 0
            for i in range(N):
                if res.x[i] > 0.5:
                    mask |= (1 << (i + 1))
            return mask
    except Exception:
        pass
    
    return None
