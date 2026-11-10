import time
import random
from fractions import Fraction
import numpy as np
import scipy.optimize as opt

_N = 256
_KILL = 90.0
_MARGIN = 8.0

_RECIP = [0.0] + [1.0 / i for i in range(1, _N + 1)]

def _generate_aps():
    """Generate all 3-term arithmetic progressions in [1, N]."""
    aps = []
    for b in range(1, _N + 1):
        max_d = min(b - 1, _N - b)
        for d in range(1, max_d + 1):
            a = b - d
            c = b + d
            aps.append((a, b, c))
    return aps

def _solve_ilp():
    """Solve the problem as an Integer Linear Program."""
    try:
        # Check if milp is available
        if not hasattr(opt, 'milp'):
            return None
            
        aps = _generate_aps()
        if not aps:
            return list(range(1, _N + 1)) # No constraints, take all? No, weights decrease, but no APs? 
            # Actually if N=1, no APs. If N=2, no APs.
            
        # Variables x_1 ... x_N
        # Objective: maximize sum(1/i * x_i)
        # milp minimizes, so we minimize -sum(1/i * x_i)
        c = np.array([-_RECIP[i] for i in range(1, _N + 1)])
        
        # Constraints: x_a + x_b + x_c <= 2 for each AP (a,b,c)
        # A is sparse matrix? scipy.optimize.milp accepts LinearConstraint with sparse A.
        # Let's build a sparse matrix.
        rows = []
        cols = []
        data = []
        for (a, b, c) in aps:
            # indices are 0-based for variables x_1...x_N
            rows.extend([len(rows), len(rows), len(rows)])
            cols.extend([a-1, b-1, c-1])
            data.extend([1, 1, 1])
            
        from scipy.sparse import csr_matrix
        A = csr_matrix((data, (rows, cols)), shape=(len(aps), _N))
        
        # Bounds: 0 <= x_i <= 1
        bounds = [(0, 1) for _ in range(_N)]
        
        # Integrality: 1 for all variables (binary)
        integrality = np.ones(_N)
        
        # Solve
        res = opt.milp(
            c=c,
            constraints=opt.LinearConstraint(A, ub=2),
            integrality=integrality,
            bounds=bounds
        )
        
        if res.status == 0: # Optimal
            x = res.x
            # x is float, but should be 0 or 1
            # Threshold at 0.5
            sol = [i + 1 for i in range(_N) if x[i] > 0.5]
            return sol
        else:
            return None
            
    except Exception as e:
        return None

def _verify(sol):
    """Check if solution is 3-AP free."""
    if not sol:
        return False
    S = set(sol)
    for i, a in enumerate(sol):
        for b in sol[i+1:]:
            c = 2 * b - a
            if c in S and c > b:
                return False
            d = 2 * a - b
            if d in S and d < a:
                return False
            s = a + b
            if s % 2 == 0:
                m = s // 2
                if m in S and m != a and m != b:
                    return False
    return True

def _fallback_search():
    """Fallback to a fast greedy + local search if ILP fails."""
    # Simple greedy: add elements in order of decreasing weight if valid
    # Then hill climb
    S = set()
    # Order by weight (decreasing 1/i means increasing i? No, 1/1 > 1/2. So decreasing weight means increasing i? No.
    # Weight is 1/i. Max weight is 1/1. So we should try 1, then 2, etc?
    # Greedy for max weight usually picks highest weight first.
    candidates = list(range(1, _N + 1))
    
    # Check validity of adding x
    def can_add(x, S):
        for y in S:
            # Check if x, y, z form AP
            # 1. x, y, z => z = 2y - x
            z = 2 * y - x
            if 1 <= z <= _N and z in S:
                return False
            # 2. y, x, z => z = 2x - y
            z = 2 * x - y
            if 1 <= z <= _N and z in S:
                return False
            # 3. x, z, y => z = (x+y)/2
            s = x + y
            if s % 2 == 0:
                z = s // 2
                if 1 <= z <= _N and z in S:
                    return False
        return True

    # Greedy: Sort by weight descending (1/i descending => i ascending)
    # Wait, 1/1 is largest. So i=1 is highest weight.
    for x in range(1, _N + 1):
        if can_add(x, S):
            S.add(x)
            
    # Hill Climb: Try to swap
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_swap = None
        S_list = list(S)
        for x in S_list:
            for y in range(1, _N + 1):
                if y in S: continue
                # Try swap x for y
                # Gain = 1/y - 1/x
                gain = _RECIP[y] - _RECIP[x]
                if gain <= 0: continue
                # Check if S - {x} + {y} is valid
                S_new = S - {x}
                if can_add(y, S_new):
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (x, y)
        if best_swap:
            x, y = best_swap
            S.remove(x)
            S.add(y)
            improved = True
            
    return list(S)

def construct():
    start = time.time()
    deadline = start + _KILL - _MARGIN
    
    # Try ILP first
    sol = _solve_ilp()
    
    if sol and _verify(sol):
        # Ensure non-empty
        if not sol:
            sol = [1]
        return sol
        
    # Fallback
    if time.time() < deadline:
        sol = _fallback_search()
        if sol and _verify(sol):
            return sol
            
    # Last resort
    return [1]
