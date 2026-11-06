# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _cut_size(adj, labels):
    """Compute the number of edges crossing the 0/1 partition."""
    diff = (labels[:, None] != labels[None, :])
    return np.sum(adj * diff)


def _spectral_init(adj, n):
    """Generate initial labels using spectral method (Fiedler vector)."""
    deg = np.sum(adj, axis=1)
    L = np.diag(deg) - adj
    
    try:
        evals, evecs = np.linalg.eigh(L)
        fiedler = evecs[:, 1]
    except:
        return np.random.randint(0, 2, n).astype(np.float64)
    
    labels = (fiedler > 0).astype(np.float64)
    
    if np.all(labels == 0) or np.all(labels == 1):
        labels = np.random.randint(0, 2, n).astype(np.float64)
    
    return labels


def _sdr_init(adj, n, rng, deadline):
    """
    Generate initial labels using SDP relaxation with projected gradient ascent
    and Goemans-Williamson randomized rounding.
    """
    # The SDP relaxation for MaxCut:
    # max trace(Y A) subject to Y_ii = 1, Y >= 0
    # where Y = s s^T for the binary case
    
    # We'll use a simplified approach: optimize a matrix Y with diagonal 1
    # to maximize trace(Y A), then use GW rounding.
    
    # Start with Y = I (identity)
    Y = np.eye(n)
    
    # Number of iterations for gradient ascent
    # Each iteration: compute gradient, project back to feasible set
    # Gradient of trace(Y A) is A (for symmetric A)
    # But we need to handle the constraint Y_ii = 1
    
    # Projected gradient: Y_new = Y + eta * (A - diag(A) * I)
    # Then project to ensure Y >= 0 and Y_ii = 1
    
    # For efficiency, we'll do a limited number of iterations
    max_iter = 50
    eta = 0.1
    
    for i in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        
        # Gradient step
        # The objective is trace(Y A) = sum_ij Y_ij A_ij
        # Gradient w.r.t. Y is A (for symmetric A)
        # But we need to keep Y_ii = 1, so we zero out the diagonal of the gradient
        grad = adj.copy()
        np.fill_diagonal(grad, 0)
        
        # Update Y
        Y = Y + eta * grad
        
        # Project to PSD: use eigenvalue clipping
        try:
            evals, evecs = np.linalg.eigh(Y)
            evals_clipped = np.maximum(evals, 0)
            Y = evecs @ np.diag(evals_clipped) @ evecs.T
        except:
            Y = np.eye(n)
        
        # Enforce diagonal = 1
        np.fill_diagonal(Y, 1.0)
    
    # Goemans-Williamson randomized rounding
    # Sample a random vector u from N(0, I_n)
    # For each vertex i, label = 1 if u^T Y[:, i] > 0 else 0
    
    # Generate multiple random vectors to get diverse solutions
    num_roundings = 5
    best_labels = None
    best_cut = -1
    
    for r in range(num_roundings):
        if time.time() > deadline - 2.0:
            break
        
        u = rng.normal(0, 1, n)
        # labels_i = 1 if u^T Y[:, i] > 0
        projections = Y @ u
        labels = (projections > 0).astype(np.float64)
        
        # Ensure both classes non-empty
        if np.all(labels == 0) or np.all(labels == 1):
            continue
        
        # Evaluate cut
        cut = _cut_size(adj, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels
    
    if best_labels is None:
        # Fallback to random
        best_labels = rng.randint(0, 2, n).astype(np.float64)
    
    return best_labels


def _greedy_refine(adj, labels, deadline):
    """
    Vectorized best-improvement local search with incremental updates.
    Repeatedly find the vertex whose flip increases the cut the most.
    """
    n = len(labels)
    adj = np.asarray(adj, dtype=np.float64)
    labels = labels.astype(np.float64).copy()
    
    s = 2 * labels - 1
    As = adj @ s
    best_cut = 0.5 * (np.sum(adj) - s @ As)
    
    while True:
        if time.time() > deadline:
            break
        
        gains = s * As
        
        max_gain = np.max(gains)
        
        if max_gain <= 0:
            break
        
        v = np.argmax(gains)
        
        # Flip vertex v
        s[v] = -s[v]
        labels[v] = 1 - labels[v]
        
        # Update As incrementally
        # As[u] changes by: -2 * A[v,u] * s[v] (since s[v] flipped)
        # Actually: As[u] = sum_j A[u,j] s[j]
        # When s[v] flips, As[u] changes by A[u,v] * (new_s[v] - old_s[v]) = A[u,v] * (-2 * old_s[v])
        # Since A is symmetric, A[u,v] = A[v,u]
        As += -2 * s[v] * adj[:, v]  # adj[:, v] is the column (same as row for symmetric)
        
        # Update best_cut
        best_cut += max_gain
    
    return labels, best_cut


def _random_init(n, rng):
    """Generate random initial labels."""
    return rng.randint(0, 2, n).astype(np.float64)


def _simulated_annealing_polish(adj, labels, deadline):
    """
    Simulated annealing to escape local optima.
    """
    n = len(labels)
    adj = np.asarray(adj, dtype=np.float64)
    labels = labels.astype(np.float64).copy()
    
    s = 2 * labels - 1
    As = adj @ s
    current_cut = 0.5 * (np.sum(adj) - s @ As)
    
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Temperature schedule
    T_start = 2.0
    T_end = 0.01
    n_iter = 200
    
    for i in range(n_iter):
        if time.time() > deadline:
            break
        
        T = T_start * (T_end / T_start) ** (i / n_iter)
        
        # Pick a random vertex to flip
        v = np.random.randint(0, n)
        
        # Compute gain from flipping v
        gain = s[v] * As[v]
        
        # Metropolis criterion
        if gain > 0 or np.random.random() < np.exp(gain / T):
            # Accept the move
            s[v] = -s[v]
            labels[v] = 1 - labels[v]
            As += -2 * s[v] * adj[:, v]
            current_cut += gain
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
    
    return best_labels, best_cut


def partition(adjacency):
    """
    Find a high-quality cut using spectral initialization, SDP-based initialization,
    and vectorized best-improvement local search with multiple restarts.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    n = len(adj)
    
    deadline = time.time() + 110.0
    
    best_labels = None
    best_cut = -1
    
    rng = np.random.RandomState(42)
    
    # Strategy:
    # 1. Spectral init + refine
    # 2. SDP init + refine (multiple roundings)
    # 3. Multiple random restarts + refine
    # 4. Simulated annealing polish on best solution
    
    # First try spectral initialization
    spec_labels = _spectral_init(adj, n)
    refined_labels, cut_val = _greedy_refine(adj, spec_labels, deadline)
    
    if cut_val > best_cut:
        best_cut = cut_val
        best_labels = refined_labels.copy()
    
    # Try SDP-based initialization
    sdp_labels = _sdr_init(adj, n, rng, deadline)
    refined_labels, cut_val = _greedy_refine(adj, sdp_labels, deadline)
    
    if cut_val > best_cut:
        best_cut = cut_val
        best_labels = refined_labels.copy()
    
    # Now do random restarts
    restart_count = 0
    while time.time() < deadline - 5.0:  # Leave 5s margin for SA polish
        random_labels = _random_init(n, rng)
        refined_labels, cut_val = _greedy_refine(adj, random_labels, deadline)
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = refined_labels.copy()
        
        restart_count += 1
        
        if restart_count > 2000:
            break
    
    # Final polish: simulated annealing on the best solution
    if best_labels is not None and time.time() < deadline - 2.0:
        sa_labels, sa_cut = _simulated_annealing_polish(adj, best_labels, deadline)
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
    
    if best_labels is None:
        best_labels = _random_init(n, rng)
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
