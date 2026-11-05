import time
import random
import numpy as np
import scipy.optimize as opt

_N = 256
_DEADLINE = 85.0

def _build_conflict():
    """Build conflict adjacency: conflict[i][j] = True iff (i*j)%(i+j)==0 for i!=j."""
    conflict = [[False] * (_N + 1) for _ in range(_N + 1)]
    neighbors = [set() for _ in range(_N + 1)]
    for i in range(1, _N + 1):
        for j in range(i + 1, _N + 1):
            if (i * j) % (i + j) == 0:
                conflict[i][j] = True
                conflict[j][i] = True
                neighbors[i].add(j)
                neighbors[j].add(i)
    return conflict, neighbors

def _validate(S):
    """Check that S is a valid independent set."""
    S_set = set(S)
    for a in S:
        for b in S:
            if a != b and (a * b) % (a + b) == 0:
                return False
    return len(S) > 0

def _can_add(x, S_set, neighbors):
    return not (neighbors[x] & S_set)

def _greedy_init(neighbors, mode='degree'):
    if mode == 'degree':
        degrees = [len(neighbors[i]) for i in range(_N + 1)]
        order = list(range(1, _N + 1))
        order.sort(key=lambda x: (degrees[x], x))
    elif mode == 'random':
        order = list(range(1, _N + 1))
        random.shuffle(order)
    else:
        order = list(range(1, _N + 1))
    S = []
    S_set = set()
    for x in order:
        if _can_add(x, S_set, neighbors):
            S.append(x)
            S_set.add(x)
    return S, S_set

def _local_search(S, neighbors, time_limit):
    S = list(S)
    S_set = set(S)
    complement = set(range(1, _N + 1)) - S_set
    start = time.time()
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # Add any available node
        for x in list(complement):
            if _can_add(x, S_set, neighbors):
                S.append(x)
                S_set.add(x)
                complement.discard(x)
                improved = True
        if time.time() >= time_limit:
            break
        # Try 1-for-k swaps
        if not improved:
            best_size = len(S)
            for u in list(S):
                if time.time() >= time_limit:
                    break
                S_set.remove(u)
                new_S_set = set(S_set)
                added = 0
                for x in list(complement | {u}):
                    if _can_add(x, new_S_set, neighbors):
                        new_S_set.add(x)
                        added += 1
                if len(new_S_set) > best_size:
                    S = list(new_S_set)
                    S_set = new_S_set
                    complement = set(range(1, _N + 1)) - S_set
                    best_size = len(S)
                    improved = True
                    break
                S_set.add(u)
    return S

def _milp_max_is(conflict, neighbors, time_limit):
    """
    Exact maximum independent set via MILP.
    Maximize sum(x_i) s.t. x_i + x_j <= 1 for each edge, x_i in {0,1}.
    """
    start = time.time()
    n = _N
    # Objective: maximize sum x_i  =>  minimize -sum x_i
    c = np.full(n, -1.0)
    
    # Constraints: for each conflict edge (i,j), x_i + x_j <= 1
    # Build sparse constraint matrix
    rows = []
    cols = []
    vals = []
    rhs = []
    for i in range(1, _N + 1):
        for j in neighbors[i]:
            if i < j:  # avoid duplicates
                rows.append(len(rhs))
                cols.append(i - 1)
                vals.append(1.0)
                rows.append(len(rhs))
                cols.append(j - 1)
                vals.append(1.0)
                rhs.append(1.0)
    
    if len(rhs) == 0:
        # No conflicts: take all
        return list(range(1, _N + 1))
    
    from scipy.sparse import csr_matrix
    A = csr_matrix((vals, (rows, cols)), shape=(len(rhs), n))
    
    # Bounds: 0 <= x_i <= 1
    bounds = [(0, 1)] * n
    
    # Integrality: x_i integer
    integrality = [1] * n
    
    try:
        res = opt.milp(
            c=c,
            constraints=opt.LinearConstraint(A, lb=-np.inf, ub=rhs),
            integrality=integrality,
            bounds=bounds,
            options={'time_limit': max(1, time_limit - (time.time() - start))}
        )
        if res.success or res.status in (1, 2, 3, 4):  # optimal or near-optimal
            x = res.x
            S = [i + 1 for i in range(n) if x[i] > 0.5]
            if _validate(S):
                return S
    except Exception:
        pass
    
    # Fallback: greedy
    S, _ = _greedy_init(neighbors, 'degree')
    return S

def _two_swap_search(S, neighbors, time_limit):
    S = list(S)
    S_set = set(S)
    remaining = set(range(1, _N + 1)) - S_set
    start = time.time()
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            if time.time() >= time_limit:
                break
            for j in range(i + 1, n):
                if time.time() >= time_limit:
                    break
                r1, r2 = S_list[i], S_list[j]
                S_set.discard(r1)
                S_set.discard(r2)
                new_S_set = set(S_set)
                added = 0
                for x in list(remaining):
                    if added >= 3:
                        break
                    if _can_add(x, new_S_set, neighbors):
                        new_S_set.add(x)
                        added += 1
                if added > 2:
                    S = list(new_S_set)
                    S_set = new_S_set
                    remaining = set(range(1, _N + 1)) - S_set
                    improved = True
                    break
                else:
                    S_set.add(r1)
                    S_set.add(r2)
            if improved:
                break
    return S

def construct():
    t0 = time.time()
    conflict, neighbors = _build_conflict()
    
    # Phase 1: Try MILP for exact maximum independent set
    best_S = []
    if time.time() - t0 < _DEADLINE - 15:
        milp_time = _DEADLINE - 20  # generous time for MILP
        S_milp = _milp_max_is(conflict, neighbors, milp_time)
        if _validate(S_milp):
            best_S = S_milp
    
    # Phase 2: If MILP failed or gave small result, try greedy + local search
    if len(best_S) < 128:
        S_greedy, _ = _greedy_init(neighbors, 'degree')
        if len(S_greedy) > len(best_S):
            best_S = S_greedy
    
    # Phase 3: Local search polish
    if time.time() - t0 < _DEADLINE - 10:
        S_final = _local_search(best_S, neighbors, _DEADLINE - 10)
        if len(S_final) > len(best_S):
            best_S = S_final
    
    # Phase 4: Two-swap search
    if time.time() - t0 < _DEADLINE - 5:
        S_final = _two_swap_search(best_S, neighbors, _DEADLINE - 5)
        if len(S_final) > len(best_S):
            best_S = S_final
    
    # Phase 5: Final local search
    if time.time() - t0 < _DEADLINE - 2:
        S_final = _local_search(best_S, neighbors, _DEADLINE - 2)
        if len(S_final) > len(best_S):
            best_S = S_final
    
    # Ensure validity
    if not _validate(best_S):
        # Fallback to all odds
        best_S = list(range(1, _N + 1, 2))
    
    return best_S
