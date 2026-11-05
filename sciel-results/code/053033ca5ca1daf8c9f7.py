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
    # Objective: maximize sum(x_i) -> minimize -sum(x_i)
    c = -np.ones(n)
    
    # Build constraint matrix from conflict edges
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
    
    # Run MILP in a daemon thread with timeout
    t = threading.Thread(target=_milp_worker, daemon=True)
    t.start()
    t.join(timeout)
    
    if t.is_alive():
        # Timeout - thread is still running but we can't kill it easily.
        # Return None to use fallback.
        print(f"MILP timed out after {timeout}s", file=sys.stderr)
        return None
    
    if exception_container[0] is not None:
        print(f"MILP failed: {exception_container[0]}", file=sys.stderr)
        return None
    
    return result_container[0]


def _heuristic_fallback(conflict_mat):
    """
    A fast heuristic to provide a valid solution if MILP fails or times out.
    Simple greedy: sort by degree, pick lowest degree, remove neighbors.
    """
    n = 256
    degrees = np.array(conflict_mat.sum(axis=1)).flatten().astype(int)
    
    remaining = set(range(n))
    selected = set()
    
    neighbors = [set(conflict_mat.getrow(i).indices) for i in range(n)]
    
    while remaining:
        min_deg = float('inf')
        best_v = None
        for v in remaining:
            if degrees[v] < min_deg:
                min_deg = degrees[v]
                best_v = v
        
        if best_v is None:
            break
            
        selected.add(best_v)
        remaining.discard(best_v)
        for u in neighbors[best_v]:
            remaining.discard(u)
            
    return selected


def _local_search_polish(selected, conflict_mat, time_limit=5.0):
    """
    Try to improve the selected set by:
    1. Adding unselected vertices that don't conflict with any selected vertex.
    2. Trying swaps: remove one selected, add one or more unselected.
    Returns an improved set.
    """
    n = 256
    start_time = time.time()
    neighbors = [set(conflict_mat.getrow(i).indices) for i in range(n)]
    
    selected_set = set(selected)
    
    # Phase 1: Greedy add - try to add any unselected vertex that has no conflicts with selected
    improved = True
    while improved:
        improved = False
        for v in range(n):
            if v in selected_set:
                continue
            # Check if v conflicts with any selected vertex
            if neighbors[v] & selected_set:
                continue
            selected_set.add(v)
            improved = True
            if time.time() - start_time > time_limit:
                return selected_set
    
    # Phase 2: Try swaps - for each selected vertex, try removing it and adding multiple unselected
    for _ in range(10):  # Limit iterations
        if time.time() - start_time > time_limit:
            break
        
        best_swap = None
        best_gain = 0
        
        for v in list(selected_set):
            # If we remove v, which unselected vertices can we add?
            candidates = set()
            for u in range(n):
                if u in selected_set:
                    continue
                # u can be added if it doesn't conflict with any selected except possibly v
                if (neighbors[u] & selected_set) - {v}:
                    continue
                candidates.add(u)
            
            # Try adding all candidates (they shouldn't conflict with each other if they don't conflict with selected-v)
            # Actually they might conflict with each other. Let's do greedy on candidates.
            gain = len(candidates) - 1  # Remove 1, add up to len(candidates)
            if gain > best_gain:
                best_gain = gain
                best_swap = (v, candidates)
        
        if best_swap is None or best_gain <= 0:
            break
        
        v, candidates = best_swap
        selected_set.discard(v)
        # Greedily add from candidates
        for u in candidates:
            if not (neighbors[u] & selected_set):
                selected_set.add(u)
    
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
    
    # First, get a heuristic solution to have a baseline
    heuristic_sol = _heuristic_fallback(conflict_mat)
    
    best_sol = heuristic_sol
    best_size = len(best_sol)
    
    # Try MILP with a time budget
    elapsed = time.time() - start_time
    milp_budget = min(55.0, DEADLINE - elapsed - 10.0)  # Leave 10s for polish + safety
    
    if milp_budget > 5.0:
        try:
            milp_sol = _solve_milp(conflict_mat, timeout=milp_budget)
            
            if milp_sol is not None:
                if len(milp_sol) > best_size:
                    best_sol = milp_sol
                    best_size = len(best_sol)
                    
        except Exception as e:
            print(f"Exception in MILP phase: {e}", file=sys.stderr)
    
    # Local search polish with remaining time
    elapsed = time.time() - start_time
    polish_budget = min(5.0, DEADLINE - elapsed - 2.0)
    
    if polish_budget > 1.0:
        try:
            polished = _local_search_polish(best_sol, conflict_mat, time_limit=polish_budget)
            if len(polished) > best_size:
                best_sol = polished
                best_size = len(best_sol)
        except Exception as e:
            print(f"Exception in polish phase: {e}", file=sys.stderr)
    
    # Final check
    if not _verify(best_sol):
        # Fallback to all odds - always valid, size 128
        odds = set(range(1, 257, 2))
        if len(odds) > len(best_sol):
            best_sol = odds
            
    # Convert 0-indexed to 1-indexed
    return [v + 1 for v in best_sol]
