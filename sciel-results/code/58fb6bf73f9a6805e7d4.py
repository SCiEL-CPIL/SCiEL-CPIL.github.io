"""Maximum independent set in the bad-triple hypergraph on [1,128]."""
import time
from math import gcd

N = 128

# Precompute LCM table
_lcm_table = [[1] * (N + 1) for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(i, N + 1):
        _lcm_table[i][j] = i * j // gcd(i, j)

# Precompute pair_bad_partners: for each pair (i,j), the set of k that form a bad triple
pair_bad_partners = {}
for i in range(1, N + 1):
    for j in range(i + 1, N + 1):
        m = _lcm_table[i][j]
        partners = []
        for k in range(1, N + 1):
            if k == i or k == j:
                continue
            if _lcm_table[i][k] == m and _lcm_table[j][k] == m:
                partners.append(k)
        pair_bad_partners[(i, j)] = frozenset(partners)

def _compute_forbidden_addition(v, solution_mask):
    """Compute additional forbidden vertices when adding v to solution."""
    extra = 0
    for u in range(1, N + 1):
        if u == v:
            continue
        if (solution_mask >> (u - 1)) & 1:
            a, b = (u, v) if u < v else (v, u)
            partners = pair_bad_partners.get((a, b), frozenset())
            for k in partners:
                extra |= (1 << (k - 1))
    return extra

def _greedy_initial():
    """Get a good initial solution via greedy."""
    solution = 0
    forbidden = 0
    avail = (1 << N) - 1
    while True:
        candidates = []
        m = avail & ~forbidden
        while m:
            low = m & (-m)
            v = low.bit_length()
            candidates.append(v)
            m ^= low
        if not candidates:
            break
        best_v = None
        best_block = float('inf')
        for v in candidates:
            extra = _compute_forbidden_addition(v, solution)
            block_count = bin(extra & avail).count('1')
            if block_count < best_block:
                best_block = block_count
                best_v = v
        solution |= (1 << (best_v - 1))
        forbidden |= _compute_forbidden_addition(best_v, solution)
        avail &= ~(1 << (best_v - 1))
    return solution

def _bnds(available, solution_mask, forbidden, current_size, best_size, best_sol, deadline):
    """Branch and bound. Returns True if optimal found."""
    if time.time() > deadline:
        return False
    
    ub = current_size + bin(available & ~forbidden).count('1')
    if ub <= best_size[0]:
        return False
    
    if available == 0:
        return False
    
    candidates = []
    m = available & ~forbidden
    while m:
        low = m & (-m)
        v = low.bit_length()
        candidates.append(v)
        m ^= low
    
    if not candidates:
        return False
    
    # Try each candidate
    for v in candidates:
        if time.time() > deadline:
            return False
        extra = _compute_forbidden_addition(v, solution_mask)
        new_forbidden = forbidden | extra
        new_solution = solution_mask | (1 << (v - 1))
        new_available = available & ~(1 << (v - 1))
        
        if _bnds(new_available, new_solution, new_forbidden, current_size + 1, best_size, best_sol, deadline):
            return True
        
        # Update best if this leaf is better
        if current_size + 1 > best_size[0]:
            best_size[0] = current_size + 1
            best_sol[0] = new_solution
    
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin
    
    # Get initial greedy solution
    greedy_sol = _greedy_initial()
    greedy_size = bin(greedy_sol).count('1')
    
    best_size = [greedy_size]
    best_sol = [greedy_sol]
    
    # Run B&B to try to improve
    initial_available = (1 << N) - 1
    _bnds(initial_available, 0, 0, 0, best_size, best_sol, deadline)
    
    # Convert best solution to list
    result = []
    for i in range(1, N + 1):
        if (best_sol[0] >> (i - 1)) & 1:
            result.append(i)
    
    return result
