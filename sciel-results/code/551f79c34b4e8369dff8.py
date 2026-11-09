import time
import sys
from math import gcd, isqrt

def construct():
    start = time.time()
    DEADLINE = 75  # seconds, safety margin under 90s
    N = 256
    
    def time_left():
        return DEADLINE - (time.time() - start)
    
    # Always include 1
    # Nodes 2..N
    nodes = list(range(2, N+1))
    n_nodes = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    
    # Build conflict pairs
    conflicts = []
    for i in range(n_nodes):
        a = nodes[i]
        for j in range(i+1, n_nodes):
            b = nodes[j]
            if gcd(a, b) > 1:
                conflicts.append((i, j))
    
    # Greedy initial
    selected = [1]
    used_primes = set()
    # factorize
    def prime_factors(x):
        factors = set()
        d = 2
        while d*d <= x:
            while x % d == 0:
                factors.add(d)
                x //= d
            d += 1
        if x > 1:
            factors.add(x)
        return factors
    
    pf = {v: prime_factors(v) for v in nodes}
    
    # Greedy: sort by value desc, add if no prime conflict
    for v in sorted(nodes, reverse=True):
        if time_left() < 5:
            break
        if not (pf[v] & used_primes):
            selected.append(v)
            used_primes |= pf[v]
    
    best = list(selected)
    best_sum = sum(best)
    
    # Local search: try to improve
    selected_set = set(best)
    used = set()
    for v in best:
        if v != 1:
            used |= pf[v]
    
    unselected = [v for v in nodes if v not in selected_set]
    
    improved = True
    while improved and time_left() > 10:
        improved = False
        # Try single replacement
        for s in list(selected_set):
            if s == 1:
                continue
            if time_left() < 10:
                break
            # remove s
            new_used = used - pf[s]
            best_gain = 0
            best_add = None
            for v in unselected:
                if v in selected_set:
                    continue
                if not (pf[v] & new_used):
                    gain = v - s
                    if gain > best_gain:
                        best_gain = gain
                        best_add = v
            if best_add is not None:
                # apply
                selected_set.discard(s)
                selected_set.add(best_add)
                used = used - pf[s]
                used |= pf[best_add]
                best_sum += best_gain
                improved = True
    
    best = list(selected_set)
    
    # Try exact via scipy milp if time permits
    if time_left() > 30:
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            from scipy.sparse import csr_matrix
            import numpy as np
            
            n_vars = n_nodes  # for nodes 2..N
            c = np.array([-v for v in nodes], dtype=float)  # minimize -sum
            
            # Constraints: x_i + x_j <= 1 for conflicts
            rows = []
            cols = []
            data = []
            for (i, j) in conflicts:
                rows.append(len(data))
                cols.append(i)
                data.append(1.0)
                rows.append(len(data))
                cols.append(j)
                data.append(1.0)
            
            A = csr_matrix((data, (rows, cols)), shape=(len(conflicts), n_vars))
            cons = LinearConstraint(A, ub=1)
            bounds = Bounds(0, 1)
            integrality = np.ones(n_vars)
            
            res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
            
            if res.status == 0 and res.x is not None:
                chosen = [1]
                for i, x in enumerate(res.x):
                    if x > 0.5:
                        chosen.append(nodes[i])
                s = sum(chosen)
                if s > best_sum:
                    best = chosen
                    best_sum = s
        except Exception:
            pass
    
    # Ensure types
    result = [int(x) for x in best]
    # dedupe, ensure distinct, in range
    seen = set()
    out = []
    for x in result:
        if isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= N and x not in seen:
            seen.add(x)
            out.append(x)
    if not out:
        out = [1]
    return out
