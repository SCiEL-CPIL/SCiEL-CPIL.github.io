import numpy as np
import time
from scipy.optimize import linprog

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0
    
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    
    # Fallback scores
    degrees = adj.sum(axis=1).astype(float)
    nbr_deg_sums = adj @ degrees
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_nbr_deg = np.where(degrees > 0, nbr_deg_sums / degrees, 0.0)
    fallback_scores = -degrees + avg_nbr_deg

    # Try LP Relaxation
    try:
        if time.time() > deadline - 10:
            return fallback_scores
            
        # Get edges
        # adj is symmetric bool/int
        # We need unique edges i < j
        # np.triu_indices
        i, j = np.triu_indices(n, k=1)
        # Filter where adj[i, j] == 1
        mask = adj[i, j] == 1
        i = i[mask]
        j = j[mask]
        m = len(i)
        
        if m == 0:
            # No edges, all vertices can be picked
            return np.ones(n)
            
        # Build A_ub (or A_eq/A_ub depending on linprog interface)
        # linprog minimizes c^T x s.t. A_ub x <= b_ub, A_eq x == b_eq, bounds
        # Our constraint: y_i + y_j >= 1  =>  -y_i - y_j <= -1
        # So A_ub has -1 at i and j, b_ub has -1.
        
        # For efficiency, use sparse matrix
        from scipy.sparse import coo_matrix
        row_indices = np.concatenate([i, j])
        col_indices = np.concatenate([i, j])
        data = np.ones(len(row_indices)) * -1.0
        
        A_ub = coo_matrix((data, (row_indices, col_indices)), shape=(m, n)).tocsr()
        b_ub = -np.ones(m)
        
        c = np.ones(n)
        
        bounds = [(0, None)] * n
        
        # Solve
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            y_opt = res.x
            # Score: lower y is better
            # Add a small degree penalty to break ties?
            # If y_i is 0 for many nodes, we need a tie-breaker.
            # Low degree is better.
            scores = -y_opt - 0.1 * degrees
            return scores
        else:
            return fallback_scores
            
    except Exception:
        return fallback_scores
