"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    """
    Find the maximum sum of pairwise-coprime integers from 1..256.
    Uses ILP via scipy.optimize.milp for exact solution with fallback to
    branch-and-bound if ILP is unavailable or times out.
    """
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    N = 256
    # Vertices: integers 2..N. We'll decide which to include.
    # Integer 1 is always included (it's coprime with everything and adds 1 to sum).
    # So we solve for 2..N and add 1 at the end.
    
    # Build conflict graph: edge between i and j if gcd(i,j) > 1
    # Variables: x_i for i in 2..N, binary
    # Maximize sum(i * x_i)
    # Subject to: x_i + x_j <= 1 for all pairs (i,j) with gcd(i,j) > 1
    
    # Precompute gcd conflicts
    conflicts = []
    for i in range(2, N + 1):
        for j in range(i + 1, N + 1):
            if math.gcd(i, j) > 1:
                conflicts.append((i, j))
    
    n_vars = N - 1  # variables for 2..N
    # Map integer value to variable index
    val_to_idx = {v: v - 2 for v in range(2, N + 1)}
    
    # Objective: maximize sum(v * x_v)
    c = [-(v) for v in range(2, N + 1)]  # scipy minimizes, so negate
    
    # Constraints: x_i + x_j <= 1 for each conflict
    # A @ x <= b
    A_rows = []
    b_vals = []
    for (i, j) in conflicts:
        row = [0] * n_vars
        row[val_to_idx[i]] = 1
        row[val_to_idx[j]] = 1
        A_rows.append(row)
        b_vals.append(1)
    
    # Bounds: each variable in [0, 1]
    integrality = [1] * n_vars  # all integer (binary)
    lb = [0] * n_vars
    ub = [1] * n_vars
    
    # Try scipy.optimize.milp
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        A_arr = np.array(A_rows, dtype=float)
        b_arr = np.array(b_vals, dtype=float)
        c_arr = np.array(c, dtype=float)
        
        constraint = LinearConstraint(A_arr, -np.inf, b_arr)
        bounds = Bounds(lb, ub)
        
        result = milp(c=c_arr, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if result.success and result.x is not None:
            selected = []
            for idx, x_val in enumerate(result.x):
                if abs(x_val) > 0.5:  # x_v = 1
                    selected.append(idx + 2)
            # Add 1
            selected.append(1)
            result_list = list(selected)
            # Verify types
            result_list = [int(x) for x in result_list]
            # Verify validity
            if _verify(result_list, N):
                return result_list
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > DEADLINE:
        # Fallback: just return primes + 1 (the seed solution)
        return _primes_plus_one(N)
    
    # Fallback: branch and bound
    return _branch_and_bound(N, start_time, DEADLINE)


def _verify(lst, N):
    """Verify the list is valid: distinct ints from 1..N, pairwise coprime."""
    if not lst or len(lst) == 0:
        return False
    s = set(lst)
    if len(s) != len(lst):
        return False
    for x in lst:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x < 1 or x > N:
            return False
    # Check pairwise coprime
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if math.gcd(lst[i], lst[j]) != 1:
                return False
    return True


def _primes_plus_one(N):
    """Return primes up to N plus 1."""
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for d in range(3, int(math.isqrt(n)) + 1, 2):
            if n % d == 0:
                return False
        return True
    return [1] + [a for a in range(2, N + 1) if is_prime(a)]


def _branch_and_bound(N, start_time, deadline):
    """
    Branch and bound for maximum weight independent set.
    Vertices: 2..N, weight = value.
    Edges: non-coprime pairs.
    """
    # Build adjacency: for each vertex, set of conflicting vertices
    adj = {v: set() for v in range(2, N + 1)}
    for i in range(2, N + 1):
        for j in range(i + 1, N + 1):
            if math.gcd(i, j) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Sort vertices by weight descending for better pruning
    vertices = list(range(2, N + 1))
    vertices.sort(key=lambda v: -v)
    
    best_sum = 0
    best_set = set()
    
    # Greedy upper bound: sum of weights of remaining vertices that don't conflict
    # with the current partial solution and with each other (relaxed)
    
    def _bb(current_set, remaining, current_sum):
        nonlocal best_sum, best_set
        
        if time.time() - start_time > deadline - 1.0:
            return
        
        if not remaining:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(current_set)
            return
        
        # Upper bound: current_sum + sum of all remaining weights (relaxed, no conflict check)
        ub = current_sum + sum(remaining)
        if ub <= best_sum:
            return
        
        # Pick a vertex to branch on: the one with highest weight
        v = max(remaining, key=lambda x: x)
        rem_list = list(remaining)
        
        # Branch 1: include v
        new_remaining = [x for x in rem_list if x != v and x not in adj[v]]
        current_set.add(v)
        _bb(current_set, new_remaining, current_sum + v)
        current_set.discard(v)
        
        # Branch 2: exclude v
        _bb(current_set, [x for x in rem_list if x != v], current_sum)
    
    _bb(set(), vertices, 0)
    
    if best_sum > 0:
        result = list(best_set) + [1]
        result = [int(x) for x in result]
        if _verify(result, N):
            return result
    
    # Fallback
    return _primes_plus_one(N)
# EVOLVE-BLOCK-END
