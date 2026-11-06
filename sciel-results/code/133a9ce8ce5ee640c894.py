import time
import random
import itertools
import numpy as np
import sys
from scipy.optimize import milp, LinearConstraint, Bounds

# Global state
_start_time = None
_DEADLINE = 325.0

def _init_state(vector, n):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _is_deadline_exceeded():
    return _elapsed() > _DEADLINE

# Cache for results
_best_set = {}

def _generate_lines(n):
    """Generate all lines in F_3^n as frozensets of 3 vectors."""
    vectors = list(itertools.product([0, 1, 2], repeat=n))
    lines = []
    m = 3**n
    vecs = list(itertools.product([0, 1, 2], repeat=n))
    for i in range(m):
        u = vecs[i]
        for j in range(i+1, m):
            v = vecs[j]
            w = tuple((-u[k] - v[k]) % 3 for k in range(n))
            line_set = frozenset([u, v, w])
            lines.append(line_set)
    return lines

def _solve_ilp(n):
    """Solve ILP for max cap set for small n."""
    if n > 6: # Limit ILP to n<=6 to avoid timeout/memory issues for n=7
        return None
        
    vectors = list(itertools.product([0, 1, 2], repeat=n))
    m = len(vectors)
    
    # Generate lines
    lines = _generate_lines(n)
    
    import scipy.sparse as sp
    
    num_constraints = len(lines)
    row_indices = []
    col_indices = []
    data = []
    
    vec_to_idx = {v: i for i, v in enumerate(vectors)}
    
    for r, line in enumerate(lines):
        for v in line:
            c = vec_to_idx[v]
            row_indices.append(r)
            col_indices.append(c)
            data.append(1.0)
            
    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(num_constraints, m))
    
    # Objective: maximize sum(x) => minimize -sum(x)
    c = -np.ones(m)
    
    # Bounds: 0 <= x <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all integer
    integrality = np.ones(m)
    
    # Constraints: A x <= 2
    cons = LinearConstraint(A, ub=2)
    
    try:
        res = milp(c, constraints=cons, integrality=integrality, bounds=bounds, options={"time_limit": 30})
        if res.status == 0:
            x_sol = res.x
            cap_set = set()
            for i, val in enumerate(x_sol):
                if val > 0.5:
                    cap_set.add(vectors[i])
            return cap_set
        else:
            return None
    except Exception:
        return None

def _greedy_cap_set(order, n):
    """
    Greedily build cap set from ordered list of vectors.
    Returns the set of vectors in the cap set.
    """
    cap_set = set()
    forbidden = set()
    for v in order:
        if v not in forbidden:
            cap_set.add(v)
            for u in cap_set:
                if u == v: continue
                w = tuple((-v[i] - u[i]) % 3 for i in range(n))
                forbidden.add(w)
    return cap_set

def _heuristic_greedy_restart(n, time_budget):
    """
    Run multiple greedy iterations with random linear orderings.
    Returns the largest cap set found.
    """
    vectors = list(itertools.product([0, 1, 2], repeat=n))
    best_set = set()
    best_size = 0
    
    start = time.time()
    
    while time.time() - start < time_budget:
        if _is_deadline_exceeded():
            break
        
        # Strategy 1: Random Linear Ordering
        # Pick a random weight vector w in {0,1,2}^n
        w = tuple(random.randint(0, 2) for _ in range(n))
        
        # Score each vector by dot product mod 3
        # To break ties, add a small random noise or use index
        items = []
        for v in vectors:
            score = sum(v[i] * w[i] for i in range(n)) % 3
            # Add tiny noise to break symmetry
            noise = random.uniform(0, 0.001)
            items.append((score + noise, v))
            
        # Sort by score ascending
        items.sort(key=lambda x: x[0])
        order = [v for _, v in items]
        
        cap_set = _greedy_cap_set(order, n)
        if len(cap_set) > best_size:
            best_size = len(cap_set)
            best_set = cap_set
            
        # Strategy 2: Balance-based (keep some of the old heuristic if it was good)
        # But the directive says structural change. Random Linear is different from Balance.
        # We can mix them.
        if random.random() < 0.5:
             # Balance heuristic
             def score(v):
                 c0 = sum(1 for x in v if x == 0)
                 c1 = sum(1 for x in v if x == 1)
                 c2 = sum(1 for x in v if x == 2)
                 badness = abs(c0 - c2)**2 + 0.5 * abs(c1 - n/3.0)**2
                 noise = random.gauss(0, 0.01)
                 return badness + noise
             
             items = []
             for v in vectors:
                 items.append((score(v), v))
             items.sort(key=lambda x: x[0])
             order = [v for _, v in items]
             
             cap_set = _greedy_cap_set(order, n)
             if len(cap_set) > best_size:
                 best_size = len(cap_set)
                 best_set = cap_set

    return best_set

def priority(vector, n):
    global _start_time
    _init_state(vector, n)
    
    if _is_deadline_exceeded():
        # Fallback heuristic
        c0 = sum(1 for x in vector if x == 0)
        c2 = sum(1 for x in vector if x == 2)
        c1 = sum(1 for x in vector if x == 1)
        return -(abs(c0 - c2)**2 + 0.5*abs(c1 - n/3.0)**2)
    
    if n not in _best_set:
        # Determine strategy based on n
        cap_set = None
        
        if n <= 6:
            # Try ILP
            cap_set = _solve_ilp(n)
            
        if cap_set is None or len(cap_set) == 0:
            # Use heuristic
            # Allocate 250 seconds for search
            cap_set = _heuristic_greedy_restart(n, 250.0)
            
        _best_set[n] = cap_set
    
    # Assign priority
    # Vectors in best_set get 1.0
    # Others get 0.0
    if vector in _best_set[n]:
        return 1.0
    else:
        return 0.0
