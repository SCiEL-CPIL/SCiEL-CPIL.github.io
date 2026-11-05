"""
Strong algorithm for bounded-lcm set maximization on [1,512].
Uses Integer Linear Programming (ILP) via SciPy for exact solution,
with a heuristic fallback.
"""
import time
import numpy as np
from math import gcd, isqrt
from scipy.optimize import milp, LinearConstraint
from scipy.sparse import csr_matrix

N = 512
DEADLINE = 85.0

def _lcm(a, b):
    return a * b // gcd(a, b)

def _validate(s):
    s = set(s)
    if not s:
        return False
    for x in s:
        if x < 1 or x > N:
            return False
    lst = sorted(s)
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if _lcm(lst[i], lst[j]) > N:
                return False
    return True

def _divisors_of(n, limit=N):
    res = set()
    for i in range(1, isqrt(n) + 1):
        if n % i == 0:
            if i <= limit:
                res.add(i)
            j = n // i
            if j != i and j <= limit:
                res.add(j)
    return res

def _heuristic_fallback():
    """
    Fallback heuristic based on the prompt's seed logic.
    """
    start = time.time()
    
    # Precompute LCM compatibility
    compat_mask = [0] * (N + 1)
    for v in range(1, N + 1):
        mask = 0
        for u in range(1, N + 1):
            if _lcm(v, u) <= N:
                mask |= (1 << (u - 1))
        compat_mask[v] = mask

    def set_to_mask(s):
        m = 0
        for v in s:
            m |= (1 << (v - 1))
        return m

    def is_compatible(v, S_mask):
        return (S_mask & ~compat_mask[v]) == 0

    def greedy_by_degree(seed_set):
        S = set(seed_set)
        S_mask = set_to_mask(S)
        remaining_mask = 0
        for v in range(1, N + 1):
            if v not in S and is_compatible(v, S_mask):
                remaining_mask |= (1 << (v - 1))
        
        while remaining_mask:
            best_v = None
            best_deg = -1
            r = remaining_mask
            while r:
                lb = r & (-r)
                v = lb.bit_length()
                deg = (compat_mask[v] & remaining_mask).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                r &= r - 1
            
            if best_v is None:
                break
            S.add(best_v)
            S_mask |= (1 << (best_v - 1))
            remaining_mask &= compat_mask[best_v]
            remaining_mask &= ~(1 << (best_v - 1))
        
        return S

    # Run greedy from empty
    S = greedy_by_degree(set())
    
    # Try to augment with divisors of highly composite numbers
    M_sorted = sorted(range(1, N + 1), key=lambda m: len(_divisors_of(m, N)), reverse=True)
    for M in M_sorted[:20]:
        d = _divisors_of(M, N)
        S_test = greedy_by_degree(d)
        if len(S_test) > len(S):
            S = S_test
            
    return list(S)

def construct():
    start = time.time()
    
    # 1. Precompute incompatible pairs
    incompat_pairs = []
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if _lcm(i, j) > N:
                incompat_pairs.append((i, j))
                
    # 2. Precompute down-set constraints
    downset_pairs = []
    for v in range(1, N + 1):
        for u in _divisors_of(v):
            if u != v:
                downset_pairs.append((u, v))
                
    n_vars = N
    c = np.ones(n_vars)
    
    n_down = len(downset_pairs)
    n_inc = len(incompat_pairs)
    
    # Build sparse matrix A
    row_idx = []
    col_idx = []
    val = []
    
    # Downset constraints: x_u - x_v >= 0
    for k, (u, v) in enumerate(downset_pairs):
        row_idx.append(k)
        col_idx.append(u - 1)
        val.append(1)
        
        row_idx.append(k)
        col_idx.append(v - 1)
        val.append(-1)
        
    # Incompatibility constraints: x_i + x_j <= 1
    offset = n_down
    for k, (i, j) in enumerate(incompat_pairs):
        row_idx.append(offset + k)
        col_idx.append(i - 1)
        val.append(1)
        
        row_idx.append(offset + k)
        col_idx.append(j - 1)
        val.append(1)
        
    A = csr_matrix((val, (row_idx, col_idx)), shape=(n_down + n_inc, n_vars))
    
    # Bounds for constraints
    lb = np.full(A.shape[0], -np.inf)
    ub = np.full(A.shape[0], np.inf)
    
    # Downset: x_u - x_v >= 0
    lb[:n_down] = 0
    ub[:n_down] = np.inf
    
    # Incompat: x_i + x_j <= 1
    lb[n_down:] = -np.inf
    ub[n_down:] = 1
    
    cons = LinearConstraint(A, lb, ub)
    
    # Integrality and variable bounds
    integrality = np.ones(n_vars)
    bounds = [(0, 1)] * n_vars
    
    try:
        # Check time
        if time.time() - start > DEADLINE - 10:
            return _heuristic_fallback()
            
        # Solve ILP
        # maxsum is not a parameter, we minimize c^T x. 
        # We want to maximize sum x_i, so minimize -sum x_i.
        res = milp(c=-c, constraints=cons, integrality=integrality, bounds=bounds)
        
        if res.status == 0:  # Optimal
            x = res.x
            S = set()
            for i in range(n_vars):
                if x[i] > 0.5:
                    S.add(i + 1)
            
            if _validate(S) and len(S) > 0:
                return list(S)
            else:
                return _heuristic_fallback()
        else:
            return _heuristic_fallback()
            
    except Exception:
        return _heuristic_fallback()

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Found set of size {len(res)}")
    if _validate(res):
        print("Valid")
    else:
        print("Invalid")
