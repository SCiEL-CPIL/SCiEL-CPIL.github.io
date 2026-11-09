"""
Alternative strategy: Goemans-Williamson SDP-based Max Cut with randomized rounding,
combined with local search refinement (flip individual nodes to improve cut).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = labels[None, :] != labels[:, None]
    return int((adjacency & diff).sum() // 2)

def _local_search(adjacency, labels, deadline):
    """
    Local search: repeatedly flip a node if it improves the cut.
    Use vectorized gain computation.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    adj_float = adjacency.astype(float)
    
    # Precompute degrees
    degrees = adjacency.sum(axis=1)
    
    # For each node i, the gain of flipping i is:
    # If i is in group 0 (label 0), flipping to 1 changes cut by:
    #   (# neighbors in group 1) - (# neighbors in group 0)
    # More precisely, using the 0/1 formulation:
    #   gain_i = sum_{j: adj[i,j]=1} (1 if labels[j] != labels[i] else -1)
    #           = sum_{j: adj[i,j]=1} (2*labels[j] - 1) * (1 if labels[i]==0 else -1)
    # Let's compute this vectorized.
    
    # labels is 0/1. For node i:
    #   gain_i = sum_j A[i,j] * (1 if labels[i] != labels[j] else -1)
    #          = sum_j A[i,j] * (2*(labels[i] != labels[j]) - 1)
    #          = 2 * sum_j A[i,j]*(labels[i] != labels[j]) - degrees[i]
    
    # sum_j A[i,j]*(labels[i] != labels[j]) is the number of edges from i to nodes with different label.
    # Let cross[i] = sum_j A[i,j] * (labels[i] != labels[j])
    # Then gain_i = 2*cross[i] - degrees[i]
    
    # cross[i] = sum_j A[i,j] * (labels[i] != labels[j])
    # This is the number of edges from i to the "other" side.
    
    # We can compute cross = adjacency @ (labels != labels[None,:]).T ? 
    # Actually, let's compute:
    # diff = labels[None, :] != labels[:, None]  # (n,n) boolean
    # cross = adjacency @ diff.sum(axis=1) ? No.
    # cross[i] = sum_j A[i,j] * diff[i,j]
    # So cross = adjacency @ diff.sum(axis=1) is wrong.
    # cross = (adjacency * diff).sum(axis=1)
    
    while True:
        if time.time() > deadline:
            break
        
        diff = labels[None, :] != labels[:, None]
        cross = (adjacency * diff).sum(axis=1)
        gain = 2 * cross - degrees
        
        # Find the node with the largest positive gain
        max_gain = np.max(gain)
        if max_gain <= 0:
            break
        
        idx = np.argmax(gain)
        labels[idx] = 1 - labels[idx]
        current_cut += int(max_gain)
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
    
    return best_labels, best_cut

def _sdp_max_cut(adjacency, deadline):
    """
    Solve the SDP relaxation of Max Cut using Goemans-Williamson.
    Minimize trace(Y L) subject to Y_ii = 1, Y >= 0 (PSD).
    Use scipy.optimize.minimize on the vectorized form.
    """
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1)
    L = np.diag(degrees) - adjacency.astype(float)
    
    # Objective: minimize trace(Y L) = sum_{i,j} L_{ij} Y_{ij}
    # Since L is symmetric, trace(Y L) = sum_i sum_j L_{ij} Y_{ij}
    # But Y is PSD and Y_ii = 1.
    
    # We'll parameterize Y as a full symmetric matrix with Y_ii = 1.
    # Variables: the off-diagonal entries Y_{ij} for i < j.
    # Number of variables: n*(n-1)/2
    
    def _vec_to_Y(vec):
        Y = np.zeros((n, n))
        idx = 0
        for i in range(n):
            Y[i, i] = 1.0
            for j in range(i+1, n):
                Y[i, j] = vec[idx]
                Y[j, i] = vec[idx]
                idx += 1
        return Y
    
    def _Y_to_vec(Y):
        vec = np.zeros(n*(n-1)//2)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                vec[idx] = Y[i, j]
                idx += 1
        return vec
    
    # Objective function: trace(Y L) = sum_{i,j} L_{ij} Y_{ij}
    # Since L is symmetric and Y is symmetric:
    # trace(Y L) = 2 * sum_{i<j} L_{ij} Y_{ij} + sum_i L_{ii} Y_{ii}
    #            = 2 * sum_{i<j} L_{ij} Y_{ij} + sum_i L_{ii}
    # sum_i L_{ii} is constant (it's 2*|E|... actually sum L_ii = sum degrees = 2|E|).
    # So we just need to minimize 2 * sum_{i<j} L_{ij} Y_{ij}.
    
    # Precompute the linear coefficients for the objective.
    # obj = 2 * sum_{i<j} L_{ij} * Y_{ij}
    # Let c be the vector of coefficients: c_k = 2 * L_{ij} for the k-th variable.
    
    c = np.zeros(n*(n-1)//2)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c[idx] = 2.0 * L[i, j]
            idx += 1
    
    def objective(vec):
        return np.dot(c, vec)
    
    # Initial guess: random Y with Y_ii = 1, and try to make it PSD.
    # A simple PSD matrix with Y_ii=1: Y = u u^T where u is a unit vector.
    # Or Y = I (identity), which is PSD.
    # Let's start with Y = I, so all off-diagonals are 0.
    vec0 = np.zeros(n*(n-1)//2)
    
    # Use SLSQP or L-BFGS-B. Since we have no explicit constraints in the vectorized form
    # (we're not enforcing PSD), we need to be careful.
    # Actually, scipy.optimize.minimize doesn't enforce PSD.
    # We'll use a penalty approach or just accept that the solution may not be PSD.
    # For the purpose of this heuristic, let's try to optimize and then project to PSD.
    
    # Alternatively, use a different parameterization that ensures PSD.
    # Y = X X^T where X is n x n, but that's more variables.
    # Let's just use the simple approach and hope the optimizer finds a reasonable solution.
    
    # Actually, let's use a different approach: random initialization with random vectors.
    # The GW algorithm uses random vectors v_i in R^n, then Y_{ij} = v_i^T v_j / ||v_i|| ||v_j||.
    # But we're solving the SDP, not using the random rounding directly.
    
    # Let's try a simpler approach: use the spectral initialization as a starting point,
    # then refine with local search. The SDP might be overkill for n=100.
    
    # Actually, let's just do multiple random initializations + local search.
    # The key insight from the inspirations is that local search is important.
    
    # Let's do: spectral init + local search, then random restarts + local search.
    
    return None  # We'll handle this in the main function

def partition(adjacency):
    """
    Main partition function.
    Strategy: 
    1. Spectral initialization (Fiedler vector) + local search
    2. Multiple random restarts + local search
    3. Keep the best solution found.
    """
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Spectral Initialization + Local Search
    degrees = adjacency.sum(axis=1)
    L = np.diag(degrees) - adjacency.astype(float)
    
    try:
        vals, vecs = np.linalg.eigh(L)
        fiedler = vecs[:, 1]
        labels = (fiedler >= 0).astype(np.int8)
    except Exception:
        labels = np.random.randint(0, 2, size=n, dtype=np.int8)
    
    labels, cut = _local_search(adjacency, labels, deadline - 5.0)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()
    
    # Phase 2: Random Restarts + Local Search
    remaining = deadline - time.time() - 5.0
    if remaining > 1.0:
        # Estimate time per restart
        # Local search is fast for n=100, so we can do many restarts.
        num_restarts = 50
        time_per_restart = remaining / num_restarts
        
        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Random initialization
            labels = np.random.randint(0, 2, size=n, dtype=np.int8)
            
            # Run local search with a time limit
            sub_deadline = time.time() + time_per_restart
            labels, cut = _local_search(adjacency, labels, sub_deadline)
            
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = np.random.randint(0, 2, size=n, dtype=np.int8)
    
    return best_labels
# EVOLVE-BLOCK-END
