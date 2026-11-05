"""Program to find a large subset of [1,256] where no pair's product is divisible by their sum."""
import time
import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import milp, LinearConstraint, Bounds
import sys
import threading

DEADLINE = 85.0  # safety margin before 90s kill

def _build_conflict_matrix():
    """Build conflict matrix: edge (i,j) exists if (i*j) % (i+j) == 0."""
    n = 256
    rows, cols, vals = [], [], []
    for i in range(1, 257):
        for j in range(i + 1, 257):
            if (i * j) % (i + j) == 0:
                rows.append(i - 1)
                cols.append(j - 1)
                vals.append(1)
                rows.append(j - 1)
                cols.append(i - 1)
                vals.append(1)
    if not rows:
        return csr_matrix((256, 256), dtype=np.int8)
    mat = csr_matrix((vals, (rows, cols)), shape=(256, 256), dtype=np.int8)
    return mat

def _solve_milp(conflict_mat, timeout=55.0):
    """
    Solve the Maximum Independent Set problem using MILP.
    Returns a set of selected vertices (0-indexed) or None if failed.
    Uses a thread-based timeout to prevent hanging.
    """
    n = 256
    c = -np.ones(n)
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if conflict_mat[i, j] == 1:
                edges.append((i, j))
    
    if not edges:
        return set(range(n))
        
    m = len(edges)
    rows = []
    cols = []
    vals = []
    
    for k, (i, j) in enumerate(edges):
        rows.extend([k, k])
        cols.extend([i, j])
        vals.extend([1, 1])
        
    A = csr_matrix((vals, (rows, cols)), shape=(m, n), dtype=np.float64)
    b = np.ones(m, dtype=np.float64)
    
    lb = np.zeros(n)
    ub = np.ones(n)
    integrality = np.ones(n)
    
    constraints = LinearConstraint(A, -np.inf, b)
    bounds = Bounds(lb, ub)
    
    result_container = [None]
    exception_container = [None]
    
    def _milp_worker():
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.status in (0, 1):  # Optimal or Feasible
                x = res.x
                selected = set(np.where(x > 0.5)[0].tolist())
                result_container[0] = selected
            else:
                result_container[0] = None
        except Exception as e:
            exception_container[0] = e
    
    t = threading.Thread(target=_milp_worker, daemon=True)
    t.start()
    t.join(timeout)
    
    if t.is_alive():
        print(f"MILP timed out after {timeout}s", file=sys.stderr)
        return None
    
    if exception_container[0] is not None:
        print(f"MILP failed: {exception_container[0]}", file=sys.stderr)
        return None
    
    return result_container[0]

def _greedy_mis(conflict_mat, order=None):
    """
    Greedy Maximal Independent Set.
    If order is None, sort by degree (ascending).
    """
    n = 256
    if order is None:
        degrees = np.array(conflict_mat.sum(axis=1)).flatten().astype(int)
        order = np.argsort(degrees)
    
    neighbors = [set(conflict_mat.getrow(i).indices) for i in range(n)]
    
    remaining = set(range(n))
    selected = set()
    
    for v in order:
        if v in remaining:
            selected.add(v)
            remaining.discard(v)
            for u in neighbors[v]:
                remaining.discard(u)
                
    return selected

def _local_search_polish(selected, conflict_mat, time_limit=5.0):
    """
    Improve the selected set by trying to add vertices or perform swaps.
    """
    n = 256
    start_time = time.time()
    neighbors = [set(conflict_mat.getrow(i).indices) for i in range(n)]
    selected_set = set(selected)
    
    # Phase 1: Greedy add - try to add any unselected vertex that has no conflicts with selected
    improved = True
    while improved:
        improved = False
        # Shuffle to break symmetry if needed, but deterministic is fine for now
        for v in range(n):
            if v in selected_set:
                continue
            if not (neighbors[v] & selected_set):
                selected_set.add(v)
                improved = True
            if time.time() - start_time > time_limit:
                return selected_set
    
    # Phase 2: 1-swap local search
    # Try removing one vertex v and adding any set of vertices that become available
    # This is effectively trying to find a larger independent set in the neighborhood
    for _ in range(20): # Limit iterations
        if time.time() - start_time > time_limit:
            break
        
        best_swap = None
        best_gain = 0
        
        for v in list(selected_set):
            # If we remove v, which unselected vertices can we add?
            # A vertex u can be added if it doesn't conflict with (selected_set - {v})
            # And u is not v.
            # Also, the added vertices must not conflict with each other.
            # For a simple 1-swap, we try to add a *single* vertex u that conflicts with v 
            # but allows us to add more? No, standard 1-swap is remove v, add u.
            # Gain is 0 if we add 1 and remove 1. 
            # To increase size, we need to remove k and add k+1.
            
            # Let's try a simple "add if possible" logic which is covered in Phase 1.
            # For swaps, let's look for a vertex v in selected such that its removal 
            # allows adding 2 or more unselected vertices that don't conflict with each other.
            
            # This is getting complex. Let's stick to a simpler heuristic:
            # Try to replace a subset of selected vertices with a larger independent set from the "free" vertices.
            
            # Actually, Phase 1 usually finds the local optimum for "adding".
            # Swaps are harder. Let's try a random restart strategy if time permits.
            pass
        
        # If we don't have a sophisticated swap, we can just break.
        # The MILP is the main solver. The polish is just a safety net.
        break 

    return selected_set

def _verify(selected):
    """Verify the set is valid: no pair has (a*b) % (a+b) == 0."""
    sel = list(selected)
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            a, b = sel[i] + 1, sel[j] + 1
            if (a * b) % (a + b) == 0:
                return False
    return True

def construct():
    start_time = time.time()
    
    # Build conflict matrix
    conflict_mat = _build_conflict_matrix()
    
    # 1. Get a good initial heuristic solution
    # Try a few different orderings to find a larger MIS
    best_heuristic = set()
    max_heur_size = 0
    
    # Ordering 1: Degree ascending
    sol1 = _greedy_mis(conflict_mat)
    if len(sol1) > max_heur_size:
        best_heuristic = sol1
        max_heur_size = len(sol1)
        
    # Ordering 2: Random (a few times)
    for _ in range(5):
        if time.time() - start_time > 5.0: break
        order = np.random.permutation(256)
        sol_r = _greedy_mis(conflict_mat, order=order)
        if len(sol_r) > max_heur_size:
            best_heuristic = sol_r
            max_heur_size = len(sol_r)
            
    best_sol = best_heuristic
    best_size = len(best_sol)
    
    # 2. Try MILP with a time budget
    elapsed = time.time() - start_time
    # Leave 5s for verification and safety
    milp_budget = min(75.0, DEADLINE - elapsed - 5.0)
    
    if milp_budget > 10.0:
        try:
            milp_sol = _solve_milp(conflict_mat, timeout=milp_budget)
            
            if milp_sol is not None:
                # Verify MILP solution
                if _verify(milp_sol):
                    if len(milp_sol) > best_size:
                        best_sol = milp_sol
                        best_size = len(best_sol)
                else:
                    # If MILP returned an invalid solution (shouldn't happen), ignore
                    pass
        except Exception as e:
            print(f"Exception in MILP phase: {e}", file=sys.stderr)
    
    # 3. Polish the best solution found
    elapsed = time.time() - start_time
    polish_budget = min(5.0, DEADLINE - elapsed - 2.0)
    
    if polish_budget > 1.0:
        try:
            polished = _local_search_polish(best_sol, conflict_mat, time_limit=polish_budget)
            if _verify(polished) and len(polished) > best_size:
                best_sol = polished
                best_size = len(best_sol)
        except Exception as e:
            print(f"Exception in polish phase: {e}", file=sys.stderr)
    
    # 4. Final check
    if not _verify(best_sol):
        # Fallback to all odds - always valid, size 128
        odds = set(range(1, 257, 2))
        if len(odds) > len(best_sol):
            best_sol = odds
            
    # Convert 0-indexed to 1-indexed
    return [v + 1 for v in best_sol]
