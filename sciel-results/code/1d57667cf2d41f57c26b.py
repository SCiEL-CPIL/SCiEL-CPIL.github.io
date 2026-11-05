"""Program to find a large subset of [1,256] where no pair's product is divisible by their sum."""
import time
import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import milp, LinearConstraint, Bounds
import sys

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


def _solve_milp(conflict_mat):
    """
    Solve the Maximum Independent Set problem using MILP.
    Returns a set of selected vertices (0-indexed) or None if failed.
    """
    n = 256
    # Objective: maximize sum(x_i) -> minimize -sum(x_i)
    c = -np.ones(n)
    
    # Constraints: x_i + x_j <= 1 for all edges (i,j)
    # conflict_mat is symmetric. We need to extract unique edges.
    # Using the sparse matrix structure.
    # The matrix has 1s at (i,j) and (j,i) for each conflict.
    # We can iterate over the non-zero elements of the upper triangle.
    
    # Efficient way to get rows and cols of constraints
    # We need A @ x <= b
    # A is a sparse matrix where A[k, i] = 1 and A[k, j] = 1 for edge (i,j)
    
    # Let's build the constraint matrix directly from the conflict list
    # To avoid double counting, we only consider i < j
    constraint_rows = []
    constraint_cols = []
    constraint_vals = []
    
    # Get indices from the sparse matrix
    # conflict_mat.indices gives column indices for each row
    # We need to be careful not to include duplicates or self-loops (though none exist here)
    
    # A more robust way: iterate i from 0 to n-1, j from i+1 to n-1
    # But that's O(N^2). N=256, so 32k checks. Very fast.
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if conflict_mat[i, j] == 1:
                edges.append((i, j))
    
    if not edges:
        # No conflicts, take all
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
    
    # Bounds: 0 <= x_i <= 1
    lb = np.zeros(n)
    ub = np.ones(n)
    
    # Integrality: x_i must be integer
    integrality = np.ones(n)
    
    # Define problem
    # Minimize c^T x
    # Subject to:
    #   A @ x <= b  (LinearConstraint(A, -inf, b))
    #   lb <= x <= ub
    #   x is integer
    
    constraints = LinearConstraint(A, -np.inf, b)
    bounds = Bounds(lb, ub)
    
    try:
        # Set time limit to 60 seconds to be safe, though we check deadline in construct
        # scipy.optimize.milp does not have a direct timeout parameter in older versions,
        # but we can wrap it or just rely on the fact that it's small.
        # However, to be safe against infinite loops, we should ideally have a timeout.
        # Since we can't easily interrupt scipy.optimize.milp, we rely on the problem size.
        # If it takes too long, it's a risk. But 256 vars is small for modern solvers.
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0:  # Optimal
            # res.x contains the solution
            # Due to floating point, we should threshold
            x = res.x
            selected = set(np.where(x > 0.5)[0].tolist())
            return selected
        elif res.status == 1:  # Feasible
            x = res.x
            selected = set(np.where(x > 0.5)[0].tolist())
            return selected
        else:
            return None
    except Exception as e:
        print(f"MILP failed: {e}", file=sys.stderr)
        return None


def _heuristic_fallback(conflict_mat):
    """
    A fast heuristic to provide a valid solution if MILP fails or times out.
    Simple greedy: sort by degree, pick lowest degree, remove neighbors.
    """
    n = 256
    # Calculate degrees
    degrees = np.array(conflict_mat.sum(axis=1)).flatten().astype(int)
    
    # Sort vertices by degree (ascending)
    order = np.argsort(degrees)
    
    selected = set()
    # We need to check conflicts dynamically or precompute
    # For simplicity and speed, let's just do a greedy pass
    # This might not be optimal but it's valid.
    
    # Actually, a better heuristic is random restarts with greedy
    # But for a fallback, let's just pick the vertex with min degree, add it, remove its neighbors.
    
    remaining = set(range(n))
    selected = set()
    
    # Precompute neighbors for fast removal
    neighbors = [set(conflict_mat.getrow(i).indices) for i in range(n)]
    
    while remaining:
        # Find vertex in remaining with min degree in the induced subgraph?
        # That's expensive. Just use original degree as a proxy or just pick first in sorted order.
        # Let's pick the one with min original degree among remaining.
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
        # Remove neighbors
        for u in neighbors[best_v]:
            remaining.discard(u)
            
    return selected


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
    # This is fast and ensures we have *some* valid answer
    heuristic_sol = _heuristic_fallback(conflict_mat)
    
    best_sol = heuristic_sol
    best_size = len(best_sol)
    
    # Try MILP
    # We need to be careful about time.
    # If we are already close to deadline, skip MILP.
    if time.time() - start_time < 10: # Give it 10s to setup, then try MILP with remaining time
        
        # We cannot easily timeout scipy.optimize.milp.
        # However, for N=256, it should be fast.
        # If it hangs, the whole process might be killed.
        # To be safe, we could run it in a thread and join with timeout,
        # but that's complex. 
        # Given the problem size, direct call is likely safe.
        
        try:
            milp_sol = _solve_milp(conflict_mat)
            
            if milp_sol is not None:
                if len(milp_sol) > best_size:
                    best_sol = milp_sol
                    best_size = len(best_sol)
                    
        except Exception as e:
            print(f"Exception in MILP phase: {e}", file=sys.stderr)
            
    # Final check
    if not _verify(best_sol):
        # This shouldn't happen, but just in case
        # Fallback to all odds? (Odd*Odd = Odd, Odd+Odd = Even. Odd % Even != 0 unless Odd=0 which isn't in [1,256])
        # Wait, if a,b are odd, a+b is even. a*b is odd. Odd % Even is never 0.
        # So the set of all odd numbers is always a valid independent set.
        # Size = 128.
        odds = set(range(1, 257, 2))
        if len(odds) > len(best_sol):
            best_sol = odds
            
    # Convert 0-indexed to 1-indexed
    return [v + 1 for v in best_sol]
