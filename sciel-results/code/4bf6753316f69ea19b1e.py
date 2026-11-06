"""
Heuristic Maximum Cut solver using Spectral Initialization and Deterministic Best-Improvement Local Search.
Strategy:
1. Initialize using the Fiedler vector (second eigenvector of the Laplacian) for a strong starting point.
2. Perform a deterministic, vectorized best-improvement local search.
   - In each pass, compute the gain for flipping every vertex.
   - Flip the vertex with the maximum positive gain.
   - Repeat until no positive gain exists (local optimum).
   - This is deterministic given the initialization.
3. To handle cases where a single local optimum might be suboptimal, we can use a "spectral perturbation" or simply rely on the fact that for n=100, the spectral initialization is often very close to global optimum, and the local search is extremely fast, allowing us to run the search on multiple spectral variants or simply run the search to convergence.
   However, since the current program uses random restarts, and the directive asks for a structural change, we will stick to a single, highly optimized deterministic path but potentially use the time budget to verify or refine.
   Actually, a better structural change is to use the time budget for *multiple* spectral initializations with different random perturbations or simply run the deterministic search on the spectral init. 
   Wait, the inspiration 1 (score 1.0120) uses "deterministic, vectorized Best-Improvement Local Search (BILS) loop seeded by the spectral initialization".
   Let's implement exactly that. It is faster and more reliable than SA for n=100.
   
   To maximize score, we should ensure we reach a local optimum. 
   We can also try a few random initializations if time permits, but the spectral one is the key structural difference from "random init + SA".
   
   Let's refine: 
   1. Compute Laplacian L = D - A.
   2. Find Fiedler vector.
   3. Threshold to get initial spin vector.
   4. Run Best-Improvement Local Search (BILS).
   5. If time remains, we might try other eigenvectors or random restarts, but BILS is very fast. 
   Let's just do the spectral init + BILS. It's a clean, different strategy.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def partition(adjacency):
    """
    Solve Maximum Cut using Spectral Initialization and Best-Improvement Local Search.
    Returns a 0/1 label per vertex.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Leave 5s safety margin
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for fast access
    adj = np.array(adjacency, dtype=np.int32)
    adj = (adj + adj.T) // 2
    
    # Precompute degrees
    degrees = adj.sum(axis=1)
    total_degree = degrees.sum()
    
    def get_cut(s):
        """Calculate cut size for spin vector s (values -1, 1)."""
        return 0.5 * (total_degree - s @ (adj @ s))
    
    def best_improvement_local_search(s):
        """
        Deterministic Best-Improvement Local Search.
        Repeatedly flips the vertex that gives the largest positive gain.
        """
        s = s.copy()
        current_cut = get_cut(s)
        
        # We'll loop until no improvement is found in a full pass of checking all vertices?
        # No, "Best Improvement" usually means: find the single best move, take it, repeat.
        # This is O(n^2) per flip, O(n) flips worst case? No, can be more.
        # But for n=100, it's very fast.
        
        while True:
            # Calculate gains for all vertices
            adj_s = adj @ s
            gains = s * adj_s
            
            max_gain = np.max(gains)
            
            if max_gain <= 0:
                break
            
            # Find vertex to flip
            v = np.argmax(gains)
            s[v] *= -1
            current_cut += max_gain
            
            # Time check
            if time.time() > DEADLINE:
                break
                
        return s, current_cut

    # 1. Spectral Initialization
    # Build Laplacian L = D - A
    # L is symmetric
    # We want the Fiedler vector (2nd smallest eigenvector)
    
    # For n=100, dense eigendecomposition is fast enough and simpler than sparse iterative methods which might have convergence issues or overhead.
    # Let's use np.linalg.eigh for dense matrix.
    
    # L = D - A
    # D is diagonal matrix of degrees
    L = np.diag(degrees) - adj
    
    # Compute eigenvalues and eigenvectors
    # We only need the second smallest eigenvalue's vector.
    # eigh returns eigenvalues in ascending order.
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is eigvecs[:, 1] (index 1 is the 2nd smallest)
        fiedler_vec = eigvecs[:, 1]
        
        # Threshold to get spin vector
        # If fiedler_vec[i] > 0 -> 1, else -1
        s_init = np.where(fiedler_vec > 0, 1, -1).astype(np.int8)
        
    except Exception:
        # Fallback to random if spectral fails
        s_init = np.random.choice([-1, 1], size=n).astype(np.int8)
        
    # 2. Local Search
    s_best, cut_best = best_improvement_local_search(s_init)
    
    # 3. If time permits, we can try a few random restarts to see if we find a better local optimum.
    # But the directive says "Make a STRUCTURAL change... a different design or modelling family".
    # Spectral + BILS is a distinct family from SA.
    # Adding random restarts makes it hybrid. Let's see if the spectral one is strong enough.
    # For n=100, spectral is usually very good.
    
    # However, to be safe and potentially beat 1.0046, we can use the remaining time for a few random restarts with BILS.
    # This is still structurally different from SA because the core optimizer is deterministic BILS, not stochastic SA.
    
    rng = np.random.default_rng(42)
    
    # If we have significant time left, do a few random restarts with BILS
    # BILS is much faster than SA, so we can do more restarts.
    
    while time.time() < DEADLINE - 5.0:
        s_rand = rng.choice([-1, 1], size=n).astype(np.int8)
        s_rand_best, cut_rand_best = best_improvement_local_search(s_rand)
        
        if cut_rand_best > cut_best:
            cut_best = cut_rand_best
            s_best = s_rand_best.copy()
            
        # Check if we are stuck in the same local optimum? 
        # If not improving, we might just keep going until time is up.
        # BILS is fast, so this is fine.

    # Convert spin vector (-1, 1) back to labels (0, 1)
    labels = (s_best + 1) // 2
    labels = labels.astype(int)
    
    return labels
# EVOLVE-BLOCK-END
