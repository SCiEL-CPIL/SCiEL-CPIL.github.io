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
    """Generate all lines in F_3^n as tuples of 3 vectors."""
    vectors = list(itertools.product([0, 1, 2], repeat=n))
    lines = []
    # Map vector to index for faster lookup?
    # We just need to generate constraints.
    # Iterate all pairs u, v.
    # To avoid duplicates and self-pairs, use u < v in index order.
    # w = -(u+v) mod 3.
    # If w is distinct from u and v, add line.
    # Note: u, v, w are distinct if u != v.
    # Because if u = v, w = -2u = u (mod 3). So u=v implies w=u.
    # So for any u != v, w is distinct from u and v?
    # u + v + w = 0.
    # If w = u, 2u + v = 0 => v = -2u = u. Contradiction.
    # If w = v, u + 2v = 0 => u = -2v = v. Contradiction.
    # So yes, distinct.
    
    # For efficiency, we can just iterate u, v where index(u) < index(v).
    # But we need to map vectors to indices or just use tuples as keys.
    # For n <= 7, 3^n is small.
    
    L = []
    m = 3**n
    # Precompute vectors list
    vecs = list(itertools.product([0, 1, 2], repeat=n))
    for i in range(m):
        u = vecs[i]
        for j in range(i+1, m):
            v = vecs[j]
            w = tuple((-u[k] - v[k]) % 3 for k in range(n))
            # w is a valid vector.
            # We need to ensure we don't add duplicate lines.
            # The line is {u, v, w}.
            # The pair (u, v) defines the line uniquely?
            # Yes, the line through u and v is unique.
            # So each pair (u, v) with u != v generates a unique line.
            # We just iterate i < j to cover each line exactly once?
            # Wait, the line contains 3 points. There are 3 pairs: (u,v), (u,w), (v,w).
            # We only want to add the constraint once.
            # The constraint is x_u + x_v + x_w <= 2.
            # If we iterate all i < j, we will generate the same constraint 3 times?
            # No, (u,v) gives w. (u,w) gives v. (v,w) gives u.
            # So the set {u,v,w} is the same.
            # To avoid duplicates, we can identify the line by its sorted tuple.
            line_set = frozenset([u, v, w])
            L.append(line_set)
            
    # This is O(3^{2n}). For n=7, 4.7M iterations. OK.
    return L

def _solve_ilp(n):
    """Solve ILP for max cap set for small n."""
    if n > 7:
        return None
        
    vectors = list(itertools.product([0, 1, 2], repeat=n))
    m = len(vectors)
    
    # Generate lines
    lines = _generate_lines(n)
    
    # Build constraint matrix A and bounds
    # x_u + x_v + x_w <= 2
    # A is sparse matrix.
    # scipy.optimize.milp expects LinearConstraint.
    
    # For m up to 2187 (n=7) and L up to 800k, this is large.
    # Let's check if we can do it faster.
    # Actually, for n=7, it might take too long to build the dense matrix.
    # Use sparse matrix?
    # scipy.optimize.milp supports sparse matrices.
    
    import scipy.sparse as sp
    
    rows = []
    cols = []
    vals = []
    
    # Map vector to index
    vec_to_idx = {v: i for i, v in enumerate(vectors)}
    
    for line in lines:
        indices = [vec_to_idx[v] for v in line]
        for idx in indices:
            rows.append(len(vals)) # This is wrong, need to track row index properly
            # Actually, each constraint is a row.
            pass
            
    # Re-do constraint building
    num_constraints = len(lines)
    row_indices = []
    col_indices = []
    data = []
    
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
        res = milp(c, constraints=cons, integrality=integrality, bounds=bounds, options={"time_limit": 60})
        if res.status == 0:
            x_sol = res.x
            # Threshold
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

def _is_valid_addition(v, cap_set, n):
    """Check if v can be added to cap_set without forming a line."""
    for u in cap_set:
        if u == v:
            continue
        w = tuple((-v[i] - u[i]) % 3 for i in range(n))
        if w in cap_set:
            return False
    return True

def _simulated_annealing(initial_set, n, time_budget):
    """
    Improve the cap set using Simulated Annealing.
    """
    cap_set = set(initial_set)
    best_set = set(initial_set)
    best_size = len(cap_set)
    
    # All vectors for random sampling
    all_vectors = list(itertools.product([0, 1, 2], repeat=n))
    
    start_time = time.time()
    T = 1.0
    cooling_rate = 0.9999 # Tune this
    min_T = 1e-4
    
    # Precompute indices for faster access? 
    # all_vectors is a list of tuples.
    
    while time.time() - start_time < time_budget:
        if _is_deadline_exceeded():
            break
            
        # Random move
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to ADD a random vector
            if len(cap_set) < len(all_vectors):
                v = random.choice(all_vectors)
                if v not in cap_set:
                    if _is_valid_addition(v, cap_set, n):
                        # Accept
                        cap_set.add(v)
                        if len(cap_set) > best_size:
                            best_size = len(cap_set)
                            best_set = set(cap_set)
                    else:
                        # Reject (Delta = 0, but we didn't change state)
                        pass
                # If v in cap_set, do nothing
            else:
                # Set is full, try remove
                pass
        elif move_type < 0.8:
            # Try to REMOVE a random vector
            if cap_set:
                u = random.choice(list(cap_set))
                cap_set.remove(u)
                # Delta = -1
                # Accept if random() < exp(-1/T)
                if random.random() < np.exp(-1.0 / T):
                    if len(cap_set) < best_size:
                        # We might have lost the best, but we keep tracking best_set separately
                        pass
                else:
                    # Revert
                    cap_set.add(u)
        else:
            # Try to SWAP: Remove u, Add v
            if cap_set and len(cap_set) < len(all_vectors):
                u = random.choice(list(cap_set))
                cap_set.remove(u)
                
                # Try to find a valid v to add
                # Randomly pick a few v's to try
                found_valid_v = False
                for _ in range(10):
                    v = random.choice(all_vectors)
                    if v not in cap_set and v != u:
                        if _is_valid_addition(v, cap_set, n):
                            cap_set.add(v)
                            found_valid_v = True
                            break
                
                if found_valid_v:
                    # Size unchanged, but structure changed
                    # Accept with some probability? Or always accept if valid?
                    # Usually swap is neutral, so accept if it doesn't hurt.
                    # But we want to explore.
                    if random.random() < 0.5: # Random acceptance for neutral moves
                         pass # Keep the change
                    else:
                         # Revert
                         cap_set.remove(v)
                         cap_set.add(u)
                else:
                    # Revert
                    cap_set.add(u)
        
        T *= cooling_rate
        if T < min_T:
             T = min_T
             
    return best_set

def _heuristic_greedy_restart(n, time_budget):
    """
    Run multiple greedy iterations with random perturbations.
    Returns the largest cap set found.
    """
    vectors = list(itertools.product([0, 1, 2], repeat=n))
    best_set = set()
    best_size = 0
    
    start = time.time()
    iterations = 0
    
    while time.time() - start < time_budget:
        if _is_deadline_exceeded():
            break
            
        # Generate a heuristic score with noise
        # Base heuristic: balance of 0s, 1s, 2s
        def score(v):
            c0 = sum(1 for x in v if x == 0)
            c1 = sum(1 for x in v if x == 1)
            c2 = sum(1 for x in v if x == 2)
            # Prefer balanced
            base = abs(c0 - c2)**2 + 0.5 * abs(c1 - n/3.0)**2
            noise = random.uniform(0, 0.1)
            return base - noise # Lower is better for sorting asc, but we want high priority.
            # Wait, priority is descending. So we want high score to be picked first?
            # No, the evaluator sorts by priority DESCENDING.
            # So we want "good" vectors to have HIGH priority.
            # So we want to minimize "badness".
            # Let's define priority = -badness + noise.
        
        # Create list of (priority, vector)
        items = []
        for v in vectors:
            c0 = sum(1 for x in v if x == 0)
            c1 = sum(1 for x in v if x == 1)
            c2 = sum(1 for x in v if x == 2)
            badness = abs(c0 - c2)**2 + 0.5 * abs(c1 - n/3.0)**2
            # Add noise to explore
            noise = random.gauss(0, 0.01)
            items.append((badness + noise, v))
            
        # Sort by badness ascending (lowest badness first -> highest priority)
        items.sort(key=lambda x: x[0])
        order = [v for _, v in items]
        
        cap_set = _greedy_cap_set(order, n)
        if len(cap_set) > best_size:
            best_size = len(cap_set)
            best_set = cap_set
            iterations += 1
            
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
        
        if n <= 7:
            # Try ILP
            cap_set = _solve_ilp(n)
            
        if cap_set is None or len(cap_set) == 0:
            # Use heuristic
            # Allocate 100 seconds for initial greedy search
            cap_set = _heuristic_greedy_restart(n, 100.0)
            
            # If we have time left, use Simulated Annealing to improve
            if cap_set and not _is_deadline_exceeded():
                # Allocate remaining time minus safety margin to SA
                time_left = _DEADLINE - _elapsed() - 5.0
                if time_left > 10:
                    improved_set = _simulated_annealing(cap_set, n, time_left)
                    if len(improved_set) > len(cap_set):
                        cap_set = improved_set
                        
        _best_set[n] = cap_set
    
    # Assign priority
    # Vectors in best_set get 1.0
    # Others get 0.0
    # To break ties and ensure stable/deterministic order if needed, we can add tiny bit
    # But 1.0 vs 0.0 is clean.
    if vector in _best_set[n]:
        return 1.0
    else:
        return 0.0
