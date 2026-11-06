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

def _local_search_cap_set(n, time_budget):
    """
    Structural Change: Local Search / Hill Climbing on the Cap Set.
    Starts with random greedy sets and improves them via swaps.
    """
    vectors = list(itertools.product([0, 1, 2], repeat=n))
    m = len(vectors)
    vec_to_idx = {v: i for i, v in enumerate(vectors)}
    idx_to_vec = {i: v for v, i in vec_to_idx.items()}
    
    # Precompute line neighbors for fast conflict checking?
    # For each vector, which other vectors form a line with it?
    # If we have a set S, adding v is valid if no u in S is a "neighbor" of v.
    # Neighbors of v are {u | u, v, w form a line}.
    # There are 2*(3^n - 1)/2? No.
    # For a fixed v, the lines containing v are determined by the direction d.
    # Line is {v, v+d, v+2d}.
    # So for each v, we can precompute the set of vectors that conflict with it (i.e., form a line with it).
    # Actually, if we add v, we forbid all u such that v, u, w is a line.
    # The number of such u is 2 * (number of lines through v) = 2 * (3^(n-1) - 1).
    # For n=8, 3^7=2187, so ~4372 neighbors. This is manageable.
    
    conflict_map = {}
    for v in vectors:
        neighbors = set()
        for d in vectors:
            if d == (0,)*n: continue
            u1 = tuple((v[i] + d[i]) % 3 for i in range(n))
            u2 = tuple((v[i] + 2*d[i]) % 3 for i in range(n))
            neighbors.add(u1)
            neighbors.add(u2)
        conflict_map[v] = neighbors

    best_set = set()
    best_size = 0
    start = time.time()
    
    while time.time() - start < time_budget:
        if _is_deadline_exceeded():
            break
        
        # 1. Initialize with a random greedy set
        # Random order
        order = vectors[:]
        random.shuffle(order)
        current_set = _greedy_cap_set(order, n)
        current_size = len(current_set)
        
        # 2. Local Search Loop
        # We try to improve the set.
        # Move: 
        #   a) If there is a vector not in the set that is not in the "forbidden" set of current_set, add it.
        #   b) If stuck, try a swap: remove a random v from set, then try to add a random u not in set.
        
        # Precompute forbidden set for current_set
        def get_forbidden(s):
            forb = set()
            for v in s:
                forb.update(conflict_map[v])
            return forb
        
        forb = get_forbidden(current_set)
        
        improved = True
        while improved and time.time() - start < time_budget:
            improved = False
            
            # Try to add any available vector
            # We can iterate through all vectors not in set and not in forb
            # To speed up, we can pick a random subset or just iterate if n is small
            # For n=8, 6561 vectors. Iterating is fast.
            added_any = False
            # Sort candidates by some heuristic? Or just random?
            # Let's try random sampling to avoid bias
            candidates = [v for v in vectors if v not in current_set and v not in forb]
            if candidates:
                # Pick a random candidate
                v_add = random.choice(candidates)
                current_set.add(v_add)
                current_size += 1
                # Update forbidden set
                # We only need to add conflicts of v_add
                forb.update(conflict_map[v_add])
                improved = True
                added_any = True
            
            if not added_any:
                # Stuck. Try a swap.
                # Remove a random vector from the set
                if current_size == 0:
                    break
                v_remove = random.choice(list(current_set))
                current_set.remove(v_remove)
                current_size -= 1
                
                # Recalculate forbidden set? 
                # Removing v_remove might allow new vectors.
                # But calculating full forbidden set is O(|S| * |neighbors|).
                # For simplicity and robustness, let's just recompute if set is small, 
                # or better, maintain a count of conflicts.
                
                # Actually, a full recompute is safer for correctness in this local search context 
                # given the small time budget per restart.
                # However, recompute is O(N * 3^n). 
                # Let's try a cheaper update: 
                # We don't easily remove conflicts from the forbidden set without counts.
                # So, for the swap move, we will recompute the forbidden set.
                forb = get_forbidden(current_set)
                
                # Now try to add a vector that is not in set and not in forb
                candidates = [v for v in vectors if v not in current_set and v not in forb]
                if candidates:
                    # Pick the one that adds the most "potential"? 
                    # Or just random.
                    v_add = random.choice(candidates)
                    current_set.add(v_add)
                    current_size += 1
                    forb.update(conflict_map[v_add])
                    improved = True
                    # Note: We might have reduced size if we removed one and didn't add one, 
                    # but we only accept the swap if we added one. 
                    # If we removed one and added one, size is same. 
                    # We should track if size increased.
                    # Actually, if we remove one, size drops by 1. If we add one, size goes up by 1. Net 0.
                    # This local search might cycle. 
                    # To escape cycles, we can require that the new vector has higher priority 
                    # or we can just accept it and hope the random walk finds a better basin.
                    # Better: Only accept swap if it allows adding MORE vectors later? 
                    # That's hard to know.
                    # Alternative: Simulated Annealing. 
                    # Given the simplicity required, let's stick to "Add if possible, else Swap (Remove+Add)".
                    # This is a random walk on the space of cap sets.
                    
                else:
                    # Could not add any vector after removal. 
                    # This means the set was "maximal" in a sense, but removing one didn't help immediately.
                    # We might want to revert the removal if we can't add? 
                    # No, let's keep the removal and try another swap.
                    # But we need to update forb again if we removed and didn't add?
                    # Yes, forb is already updated for the removal.
                    # If we can't add, we are stuck in this state. 
                    # Let's break the inner loop and restart the whole greedy+local search.
                    break
        
        if current_size > best_size:
            best_size = current_size
            best_set = set(current_set)
            
        # If we hit the deadline, break
        if _is_deadline_exceeded():
            break

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
            # Use Local Search
            # Allocate remaining time minus safety margin
            elapsed = _elapsed()
            budget = _DEADLINE - elapsed - 5.0
            if budget < 1.0:
                # Not enough time, use fast greedy
                order = list(itertools.product([0, 1, 2], repeat=n))
                random.shuffle(order)
                cap_set = _greedy_cap_set(order, n)
            else:
                cap_set = _local_search_cap_set(n, budget)
            
        _best_set[n] = cap_set
    
    # Assign priority
    # Vectors in best_set get 1.0
    # Others get 0.0
    if vector in _best_set[n]:
        return 1.0
    else:
        return 0.0
