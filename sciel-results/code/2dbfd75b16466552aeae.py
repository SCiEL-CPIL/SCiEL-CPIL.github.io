"""
Semidefinite Programming (SDP) Relaxation + Spectral Rounding + Polish for Maximum Cut.
Strategy:
1. Formulate the SDP relaxation of Maximum Cut: maximize Tr(X) subject to X_ii=1, X >= 0.
2. Solve the SDP using SciPy's MILP solver (which handles quadratic constraints via linearization or direct QP if available, but here we use a specific trick: 
   Actually, scipy.optimize.milp is for integer linear. For SDP, we usually need cvxpy. 
   Since we only have standard libs + numpy/scipy, we cannot run a full SDP solver directly without cvxpy.
   
   Alternative "Different Family": **Spectral Clustering / Eigenvalue-based Partitioning**.
   The Laplacian of the graph provides a spectral embedding. The Fiedler vector (second smallest eigenvalue of the Laplacian) gives a near-optimal partition for many graph structures (bipartite-like).
   
   Let's combine:
   1. Compute the Graph Laplacian L = D - A.
   2. Find the Fiedler vector (eigenvector corresponding to the second smallest eigenvalue).
   3. Use the sign of this vector as an initial partition.
   4. Refine using a fast, vectorized Local Search (Hill Climbing) to fix any suboptimal flips.
   
   This is materially different from ILS because the initialization is driven by global spectral properties rather than random or perturbed local states. Spectral methods often find the "global" structure that local search misses if started randomly.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def partition(adjacency):
    """
    Solve Maximum Cut using Spectral Initialization + Local Search Polish.
    Returns a 0/1 label per vertex.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Leave 5s safety margin
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array
    adj = np.array(adjacency, dtype=np.int32)
    # Ensure symmetry and no self-loops for Laplacian calculation
    adj = (adj + adj.T) // 2
    np.fill_diagonal(adj, 0)
    
    # Precompute degrees
    degrees = adj.sum(axis=1)
    total_degree = degrees.sum()
    
    def get_cut(s):
        """Calculate cut size for spin vector s (values -1, 1)."""
        return 0.5 * (total_degree - s @ (adj @ s))
    
    def local_search(s, max_iters=200):
        """
        Greedy Local Search: repeatedly flip the vertex that gives the largest
        positive gain in cut size until no improvement is possible.
        Optimized for speed with numpy vectorization.
        """
        best_s = s.copy()
        best_cut = get_cut(best_s)
        
        for _ in range(max_iters):
            if time.time() > DEADLINE:
                break
                
            # Calculate gains for all vertices
            # Gain from flipping v: s[v] * (adj @ s)[v]
            adj_s = adj @ best_s
            gains = best_s * adj_s
            
            max_gain = np.max(gains)
            
            if max_gain <= 0:
                break
            
            # Find vertex to flip
            v = np.argmax(gains)
            best_s[v] *= -1
            best_cut += max_gain
            
        return best_s, best_cut

    # --- Spectral Initialization ---
    
    # Build Laplacian L = D - A
    # For sparse efficiency, though n=100 is small, we can use dense or sparse.
    # Dense is fine for n=100.
    D = np.diag(degrees)
    L = D - adj
    
    # Find eigenvalues and eigenvectors
    # We want the second smallest eigenvalue (Fiedler value) and its eigenvector.
    try:
        # np.linalg.eigh is stable for symmetric matrices
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # Sort by eigenvalue (eigh usually returns sorted, but ensure)
        # The smallest eigenvalue is 0 (constant vector).
        # The second smallest is the Fiedler value.
        fiedler_idx = 1
        fiedler_vec = eigenvectors[:, fiedler_idx]
        
        # Convert to spin vector (-1, 1)
        # If fiedler_vec[i] > 0, assign 1, else -1.
        s_initial = np.where(fiedler_vec > 0, 1, -1).astype(np.int32)
        
    except Exception:
        # Fallback to random if spectral fails
        s_initial = np.random.choice([-1, 1], size=n).astype(np.int32)
    
    # --- Polish with Local Search ---
    
    s_best, cut_best = local_search(s_initial)
    
    # --- Multi-Start Refinement with Spectral Variants ---
    # Sometimes the Fiedler vector is not enough. We can try other eigenvectors or random restarts 
    # but initialized with a mix of spectral and random to explore.
    
    # Strategy: Run a few more local searches starting from:
    # 1. Negated Fiedler vector (should be same cut, but verify)
    # 2. Random restarts (standard, to catch cases where spectral fails)
    # 3. "Noisy" Spectral: Add small noise to Fiedler vector before rounding
    
    rng = np.random.default_rng(42)
    
    # Try negated Fiedler (just in case of numerical quirks, though cut is same)
    s_neg = -s_initial
    s_try, cut_try = local_search(s_neg)
    if cut_try > cut_best:
        s_best = s_try
        cut_best = cut_try
        
    # Try noisy spectral variants
    for _ in range(5):
        if time.time() > DEADLINE:
            break
        noise = rng.normal(0, 0.1, size=n)
        s_noisy = np.where(fiedler_vec + noise > 0, 1, -1).astype(np.int32)
        s_try, cut_try = local_search(s_noisy)
        if cut_try > cut_best:
            s_best = s_try
            cut_best = cut_try
            
    # Try standard random restarts to ensure we don't miss a better basin
    # The spectral method is strong, but random restarts with fast local search
    # can find better local optima in complex landscapes.
    
    attempts = 0
    while time.time() < DEADLINE:
        # Random restart
        s_random = rng.choice([-1, 1], size=n).astype(np.int32)
        s_try, cut_try = local_search(s_random)
        
        if cut_try > cut_best:
            s_best = s_try
            cut_best = cut_try
            
        attempts += 1
        # Limit random restarts if we've spent too much time or if n is large
        # For n=100, local_search is very fast (<1ms). 
        # We can afford thousands of restarts.
        
        # Check deadline
        if time.time() > DEADLINE:
            break

    # Convert spin vector (-1, 1) back to labels (0, 1)
    labels = (s_best + 1) // 2
    labels = labels.astype(int)
    
    return labels
# EVOLVE-BLOCK-END
