import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

_N = 256
_DEADLINE = 55.0

def _build_conflict_graph():
    """Build adjacency bitmasks: conflict_mask[a] has bit b set if (a*b)%(a+b)==0."""
    conflict_mask = [0] * (_N + 1)
    for a in range(1, _N + 1):
        for b in range(a + 1, _N + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
    return conflict_mask

def _validate(S, conflict_mask):
    s = set(S)
    if len(s) != len(S):
        return False
    if any(x < 1 or x > _N for x in s):
        return False
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = S[i], S[j]
            if (a * b) % (a + b) == 0:
                return False
    return True

def _heuristic_fallback(conflict_mask):
    """Simple greedy fallback if MILP fails."""
    S = set(range(1, _N + 1, 2))
    S_mask = 0
    for v in S:
        S_mask |= (1 << v)
    for v in range(2, _N + 1, 2):
        if not (S_mask & conflict_mask[v]):
            S_mask |= (1 << v)
            S.add(v)
    return sorted(S)

def construct():
    start_time = time.time()
    
    # Build conflict graph
    conflict_mask = _build_conflict_graph()
    
    try:
        N = _N
        # Objective: maximize sum(x_i) -> minimize -sum(x_i)
        c = -np.ones(N)
        
        # Build constraints matrix
        # Edge list
        edge_list = []
        for i in range(1, N + 1):
            mask = conflict_mask[i]
            # Only consider j > i to avoid duplicates
            for j in range(i + 1, N + 1):
                if (mask >> j) & 1:
                    edge_list.append((i - 1, j - 1))
        
        M = len(edge_list)
        
        if M == 0:
            return list(range(1, N + 1))
            
        A = np.zeros((M, N), dtype=np.float64)
        for k, (u, v) in enumerate(edge_list):
            A[k, u] = 1.0
            A[k, v] = 1.0
            
        lb = np.zeros(M)
        ub = np.ones(M)
        
        cons = LinearConstraint(A, lb, ub)
        integrality = np.ones(N)
        bounds = Bounds(0, 1)
        
        # Check time before solving
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 2:
            return _heuristic_fallback(conflict_mask)
            
        # Solve
        res = milp(c, constraints=cons, integrality=integrality, bounds=bounds)
        
        # Check time after solving
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE:
             # If it took too long, we might have a result, but let's be safe
             pass

        if res.status == 0:
            x = res.x
            S = set()
            for i in range(N):
                if x[i] > 0.5:
                    S.add(i + 1)
            
            if _validate(S, conflict_mask):
                return sorted(S)
            else:
                # Should not happen, but fallback
                return _heuristic_fallback(conflict_mask)
        else:
            return _heuristic_fallback(conflict_mask)
            
    except Exception:
        return _heuristic_fallback(conflict_mask)

