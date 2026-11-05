"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def priority(adjacency):
    """
    Compute vertex scores for greedy independent set selection.
    Uses a combination of degree, neighborhood structure, and LP relaxation
    to estimate vertex 'value' for maximum independent set.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Basic features
    degrees = A.sum(axis=1)
    
    # Neighborhood degree sum (sum of degrees of neighbors)
    nbr_deg_sum = A @ degrees
    
    # OutwardExp: how much a vertex's neighborhood expands
    # For each vertex, compute the number of unique neighbors of its neighbors
    # (excluding itself)
    # This is expensive for large n, so we approximate
    # Use: sum of (degree of neighbor - 1) / degree as a proxy
    # Actually, let's compute a cheaper measure: 
    # For vertex i, outward expansion = sum over neighbors j of (deg(j) - 1)
    # This captures how "spread out" the neighborhood is
    
    # Compute mean degree of neighbors
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_nbr_deg = np.where(degrees > 0, nbr_deg_sum / degrees, 0)
    
    # Min neighbor degree
    # For each vertex, find the minimum degree among its neighbors
    # This is O(n^2) but n=200 so it's fine
    min_nbr_deg = np.zeros(n)
    for i in range(n):
        neighbors = np.where(A[i] > 0)[0]
        if len(neighbors) > 0:
            min_nbr_deg[i] = degrees[neighbors].min()
        else:
            min_nbr_deg[i] = 0
    
    # Second-order: sum of degrees of neighbors of neighbors
    # A^2 @ degrees gives for each vertex the sum of degrees of its neighbors' neighbors
    A2_deg = (A @ A) @ degrees
    
    # Normalize features
    def normalize(x):
        x = np.asarray(x, dtype=float)
        rng = x.max() - x.min()
        if rng > 1e-12:
            return (x - x.min()) / rng
        return np.zeros_like(x)
    
    deg_norm = normalize(degrees)
    nbr_deg_norm = normalize(nbr_deg_sum)
    mean_nbr_deg_norm = normalize(mean_nbr_deg)
    min_nbr_deg_norm = normalize(min_nbr_deg)
    A2_deg_norm = normalize(A2_deg)
    
    # Try to use LP relaxation if time permits
    # The LP relaxation of MIS: maximize sum(x_i) subject to x_i + x_j <= 1 for edges, 0 <= x <= 1
    # For n=200, this might be feasible with linprog
    use_lp = n <= 250 and (time.time() - start_time) < 20.0
    
    if use_lp:
        try:
            # Build constraint matrix: for each edge (i,j), x_i + x_j <= 1
            # Also 0 <= x_i <= 1
            # Use scipy.optimize.linprog with method='highs'
            
            # Get edges
            edges = np.argwhere(A > 0)
            m = len(edges)
            
            if m == 0:
                # No edges, all vertices can be picked
                return np.ones(n)
            
            # Constraint matrix: m rows, n cols
            # For each edge (i,j), row has 1 at i and 1 at j
            C = np.zeros((m, n))
            C[np.arange(m), edges[:, 0]] = 1
            C[np.arange(m), edges[:, 1]] = 1
            
            # Objective: maximize sum(x) => minimize -sum(x)
            c_obj = -np.ones(n)
            
            # Bounds: 0 <= x_i <= 1
            bounds = [(0, 1)] * n
            
            # Run LP
            result = linprog(c_obj, A_ub=C, b_ub=np.ones(m), bounds=bounds, method='highs')
            
            if result.success:
                lp_solution = result.x
            else:
                lp_solution = None
            
            if lp_solution is not None and (time.time() - start_time) < deadline:
                lp_norm = normalize(lp_solution)
                # Combine LP solution with structural features
                # LP solution should be the primary signal
                score = 1.5 * lp_norm + 0.3 * (1 - deg_norm) + 0.2 * min_nbr_deg_norm
                return score
                
        except Exception:
            pass
    
    # Fallback: weighted combination of features
    # Based on knowledge: OutwardExp and MinNbrDeg provide new signal
    # Weighted sum with tuned weights
    
    # Use a combination that favors low-degree vertices with high min-neighbor-degree
    # (i.e., vertices whose neighbors are also low-degree, suggesting sparse local structure)
    
    score = (
        2.5 * (1 - deg_norm) +        # Lower degree is better
        1.5 * min_nbr_deg_norm +       # Higher min neighbor degree (neighbors are well-connected among themselves? No, wait)
        1.0 * (1 - mean_nbr_deg_norm) + # Lower mean neighbor degree is better
        0.5 * (1 - nbr_deg_norm) +      # Lower total neighbor degree is better
        0.3 * (1 - A2_deg_norm)         # Lower second-order degree is better
    )
    
    return score
# EVOLVE-BLOCK-END
