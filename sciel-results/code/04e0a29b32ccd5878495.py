"""Reciprocal weight of three-progression-free sets."""
import time
import random
from fractions import Fraction
import numpy as np

N = 256

def _precompute_conflicts(N):
    """Precompute for each x the list of (a, c) pairs that would form a 3-AP with x."""
    # For x to be added to S, we need:
    # 1. No a, c in S with a + c = 2x (x is middle)
    # 2. No b, c in S with x + c = 2b (x is first, c = 2b - x)
    # 3. No a, b in S with a + x = 2b (x is last, a = 2b - x)
    
    # We'll store for each x a list of pairs (u, v) such that if both u and v are in S,
    # then x cannot be added.
    
    conflicts = [[] for _ in range(N + 1)]
    
    for x in range(1, N + 1):
        # Case 1: x is middle. a + c = 2x, a < x < c
        for a in range(1, x):
            c = 2 * x - a
            if c <= N and c > x:
                conflicts[x].append((a, c))
        
        # Case 2: x is first. x + c = 2b, x < b < c => c = 2b - x
        for b in range(x + 1, N + 1):
            c = 2 * b - x
            if c <= N and c > b:
                conflicts[x].append((b, c))
        
        # Case 3: x is last. a + x = 2b, a < b < x => a = 2b - x
        for b in range(1, x):
            a = 2 * b - x
            if a >= 1 and a < b:
                conflicts[x].append((a, b))
    
    return conflicts

def _is_valid_addition(bitmask, x, conflicts):
    """Check if x can be added to set represented by bitmask."""
    if (bitmask >> (x - 1)) & 1:
        return False
    
    for a, c in conflicts[x]:
        if ((bitmask >> (a - 1)) & 1) and ((bitmask >> (c - 1)) & 1):
            return False
    
    return True

def _reciprocal_sum_float(bitmask):
    """Compute float reciprocal sum for fast comparison."""
    total = 0.0
    for x in range(1, N + 1):
        if (bitmask >> (x - 1)) & 1:
            total += 1.0 / x
    return total

def _reciprocal_sum_exact(bitmask):
    """Compute exact reciprocal sum as Fraction."""
    total = Fraction(0, 1)
    for x in range(1, N + 1):
        if (bitmask >> (x - 1)) & 1:
            total += Fraction(1, x)
    return total

def _greedy_initial(conflicts, order=None):
    """Greedy: add numbers in given order (default increasing) if they don't create a 3-AP."""
    if order is None:
        order = list(range(1, N + 1))
    
    bitmask = 0
    for x in order:
        if _is_valid_addition(bitmask, x, conflicts):
            bitmask |= (1 << (x - 1))
    return bitmask

def _local_search(bitmask, conflicts, max_iter=5000, deadline=None):
    """Local search: try adding, removing, or swapping elements to improve reciprocal sum."""
    current = bitmask
    current_float = _reciprocal_sum_float(current)
    best = bitmask
    best_float = current_float
    
    recip = [0.0] + [1.0 / i for i in range(1, N + 1)]
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Try adding a single element (prioritize small x)
        for x in range(1, N + 1):
            if (current >> (x - 1)) & 1:
                continue
            if _is_valid_addition(current, x, conflicts):
                new_bitmask = current | (1 << (x - 1))
                new_float = current_float + recip[x]
                if new_float > best_float + 1e-15:
                    best_float = new_float
                    best = new_bitmask
                    current = new_bitmask
                    current_float = new_float
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try swaps: remove one element, add another
        improved = False
        for rem in range(1, N + 1):
            if not ((current >> (rem - 1)) & 1):
                continue
            
            temp = current & ~(1 << (rem - 1))
            
            # Try adding smaller candidates first (higher reciprocal)
            for cand in range(1, N + 1):
                if cand == rem:
                    continue
                if (current >> (cand - 1)) & 1:
                    continue
                if _is_valid_addition(temp, cand, conflicts):
                    new_float = current_float - recip[rem] + recip[cand]
                    if new_float > best_float + 1e-15:
                        new_bitmask = temp | (1 << (cand - 1))
                        best_float = new_float
                        best = new_bitmask
                        current = new_bitmask
                        current_float = new_float
                        improved = True
                        break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best

def _multi_start(conflicts, deadline=None):
    """Try different greedy orderings and local search from each."""
    best_bitmask = None
    best_float = 0.0
    
    # Standard greedy (increasing order)
    bitmask = _greedy_initial(conflicts)
    bitmask = _local_search(bitmask, conflicts, max_iter=3000, deadline=deadline)
    bf = _reciprocal_sum_float(bitmask)
    if bf > best_float:
        best_float = bf
        best_bitmask = bitmask
    
    # Try greedy with some randomization
    random.seed(42)
    
    for trial in range(20):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb the greedy: skip some small numbers occasionally
        order = list(range(1, N + 1))
        for i in range(len(order)):
            if random.random() < 0.05 and order[i] > 1:
                # Move this element to the end
                val = order.pop(i)
                order.append(val)
                i -= 1
        
        bitmask = _greedy_initial(conflicts, order)
        bitmask = _local_search(bitmask, conflicts, max_iter=2000, deadline=deadline)
        bf = _reciprocal_sum_float(bitmask)
        if bf > best_float:
            best_float = bf
            best_bitmask = bitmask
    
    return best_bitmask

def _try_milp(conflicts, deadline=None):
    """Try to solve using scipy's MILP solver if available."""
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix
    except ImportError:
        return None
    
    # Number of 3-AP constraints
    ap_list = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            c = 2 * b - a
            if c <= N and c > b:
                ap_list.append((a, b, c))
    
    m = len(ap_list)
    
    # Build constraint matrix: each row corresponds to an AP (a, b, c)
    # Constraint: x_a + x_b + x_c <= 2
    rows = []
    cols = []
    data = []
    
    for i, (a, b, c) in enumerate(ap_list):
        rows.extend([i, i, i])
        cols.extend([a - 1, b - 1, c - 1])
        data.extend([1, 1, 1])
    
    A = csr_matrix((data, (rows, cols)), shape=(m, N))
    
    # Objective: maximize sum(1/x * x_i) => minimize -sum(1/x * x_i)
    c_obj = -np.array([1.0 / i for i in range(1, N + 1)])
    
    # Constraints: A @ x <= 2
    constraint = LinearConstraint(A, ub=2)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(0, 1)
    
    # Integer constraints
    integrality = np.ones(N)
    
    # Add a small timeout
    time_left = 10
    if deadline is not None:
        time_left = max(1, int(deadline - time.time()))
    
    try:
        result = milp(
            c=c_obj,
            constraints=constraint,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": time_left, "mip_rel_gap": 0.01}
        )
        
        if result.success and result.x is not None:
            # Round to nearest integer
            x_sol = np.round(result.x).astype(int)
            # Ensure validity
            bitmask = 0
            for i, val in enumerate(x_sol):
                if val > 0:
                    bitmask |= (1 << i)
            
            # Verify
            if _is_valid_3ap_free(bitmask):
                return bitmask
    except Exception:
        pass
    
    return None

def _is_valid_3ap_free(bitmask):
    """Verify the entire set has no 3-term AP."""
    s = set()
    for x in range(1, N + 1):
        if (bitmask >> (x - 1)) & 1:
            s.add(x)
    
    for a in s:
        for b in s:
            if a == b:
                continue
            c = 2 * b - a
            if c in s and c != a and c != b:
                return False
    return True

def _fix_bitmask(bitmask):
    """Fix by removing conflicting elements."""
    changed = True
    while changed:
        changed = False
        s = set()
        for x in range(1, N + 1):
            if (bitmask >> (x - 1)) & 1:
                s.add(x)
        
        for a in list(s):
            for b in list(s):
                if a == b:
                    continue
                c = 2 * b - a
                if c in s and c != a and c != b:
                    # Remove the largest to preserve small numbers
                    to_remove = max(a, b, c)
                    s.discard(to_remove)
                    bitmask &= ~(1 << (to_remove - 1))
                    changed = True
                    break
            if changed:
                break
    return bitmask

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s limit
    
    # Precompute conflicts
    conflicts = _precompute_conflicts(N)
    
    # Get initial solution via multi-start greedy + local search
    result_bitmask = _multi_start(conflicts, deadline)
    
    if result_bitmask is None or result_bitmask == 0:
        # Fallback: simple greedy
        result_bitmask = _greedy_initial(conflicts)
    
    # Try MILP if we have time
    time_left = deadline - time.time()
    if time_left > 5:
        milp_result = _try_milp(conflicts, deadline)
        if milp_result is not None and milp_result != 0:
            # Compare with current best
            if _reciprocal_sum_float(milp_result) > _reciprocal_sum_float(result_bitmask):
                result_bitmask = milp_result
    
    # Final verification and fix
    if not _is_valid_3ap_free(result_bitmask):
        result_bitmask = _fix_bitmask(result_bitmask)
    
    # Convert bitmask to list
    result = []
    for x in range(1, N + 1):
        if (result_bitmask >> (x - 1)) & 1:
            result.append(x)
    
    if len(result) == 0:
        # Fallback: just return [1]
        result = [1]
    
    return result
