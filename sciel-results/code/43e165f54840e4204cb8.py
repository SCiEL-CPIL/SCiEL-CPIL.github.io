"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def _cut_size(labels, adj):
    """Compute cut size for given 0/1 labels."""
    s = 2.0 * labels - 1.0
    E = 0.5 * np.sum(adj)
    sAs = s @ (adj @ s)
    return E - 0.5 * sAs

def _greedy_improve(labels, adj, deadline):
    """First-improvement local search: flip vertices if it improves cut."""
    curr_labels = labels.copy()
    curr_cut = _cut_size(curr_labels, adj)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        s = 2.0 * curr_labels - 1.0
        As = adj @ s
        deltas = s * As
        # Find any vertex with positive delta (first-improvement)
        pos_indices = np.where(deltas > 1e-9)[0]
        if len(pos_indices) > 0:
            # Pick the one with max delta for efficiency
            best_v = pos_indices[np.argmax(deltas[pos_indices])]
            curr_labels[best_v] = 1 - curr_labels[best_v]
            curr_cut += deltas[best_v]
            improved = True
    return curr_labels, curr_cut

def _simulated_annealing(labels, adj, deadline, T0=10.0, alpha=0.995):
    """Simulated annealing to escape local optima."""
    curr_labels = labels.copy().astype(np.float64)
    curr_cut = _cut_size(curr_labels.astype(int), adj)
    T = T0
    n = len(labels)
    
    while time.time() < deadline and T > 0.01:
        # Random vertex flip
        v = random.randint(0, n - 1)
        s = 2.0 * curr_labels - 1.0
        As = adj @ s
        delta = s[v] * As[v]
        
        if delta > 0 or random.random() < np.exp(delta / T):
            curr_labels[v] = 1.0 - curr_labels[v]
            curr_cut += delta
        
        T *= alpha
    
    # Convert to int and run greedy
    curr_labels_int = (curr_labels >= 0.5).astype(int)
    curr_labels_int, curr_cut = _greedy_improve(curr_labels_int, adj, deadline)
    return curr_labels_int, curr_cut

def _spectral_init(adj, n):
    """Spectral initialization using Fiedler vector."""
    A_sp = sp.csr_matrix(adj)
    D = np.sum(adj, axis=1)
    L = sp.diags(D) - A_sp
    
    try:
        vals, vecs = spla.eigsh(L, k=2, which='SM')
        idx = np.argsort(vals)
        fiedler_vec = vecs[:, idx[1]]
        labels = (fiedler_vec >= 0).astype(int)
        if np.all(labels == 0) or np.all(labels == 1):
            labels = np.arange(n) % 2
        return labels
    except Exception:
        return np.arange(n) % 2

def _sdp_relaxation(adj, n, deadline):
    """SDP relaxation with hyperplane rounding for initialization."""
    # For n=100, we can use a simple SDP-like approach via semidefinite programming
    # But scipy doesn't have a built-in SDP solver, so we'll use a heuristic:
    # Use multiple random hyperplane projections on a random matrix correlated with A
    
    # Create a random matrix R and compute R^T A R to get a "relaxed" view
    rng = np.random.RandomState(42)
    
    best_labels = None
    best_cut = -1
    
    for trial in range(5):
        if time.time() > deadline:
            break
        
        # Random orthogonal matrix
        Q = rng.randn(n, n)
        Q, _ = np.linalg.qr(Q)
        
        # Project adjacency into random space
        A_proj = Q.T @ adj @ Q
        
        # Use sign of diagonal of a related matrix
        # Actually, let's use a simpler approach: random hyperplane rounding
        # on the eigenvectors of A
        
        # Get top eigenvectors of A
        try:
            vals, vecs = spla.eigsh(sp.csr_matrix(adj), k=min(5, n-1), which='LA')
            # Use the first few eigenvectors for random hyperplane rounding
            for k in range(min(3, len(vals))):
                v = vecs[:, k]
                # Random hyperplane
                r = rng.randn(n)
                proj = v * (r @ v)
                labels = (proj >= 0).astype(int)
                if not (np.all(labels == 0) or np.all(labels == 1)):
                    cut = _cut_size(labels, adj)
                    if cut > best_cut:
                        best_cut = cut
                        best_labels = labels.copy()
        except Exception:
            pass
    
    if best_labels is not None:
        return best_labels
    else:
        return np.random.randint(0, 2, n)

def partition(adjacency):
    """
    Maximum Cut via SDP-inspired Initialization + Multi-Start with Greedy and Simulated Annealing.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.asarray(adjacency, dtype=np.float64)
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Generate diverse initializations
    initializations = []
    
    # Spectral init
    spec_labels = _spectral_init(adj, n)
    initializations.append(spec_labels)
    
    # SDP-inspired init
    sdp_labels = _sdp_relaxation(adj, n, deadline)
    initializations.append(sdp_labels)
    
    # Random inits
    rng = np.random.RandomState(123)
    for _ in range(5):
        rand_labels = rng.randint(0, 2, n)
        initializations.append(rand_labels)
    
    # Phase 2: For each initialization, run greedy then SA
    for init_labels in initializations:
        if time.time() > deadline - 10.0:
            break
        
        # Run greedy first
        greedy_labels, greedy_cut = _greedy_improve(init_labels, adj, deadline)
        
        if greedy_cut > best_cut:
            best_cut = greedy_cut
            best_labels = greedy_labels.copy()
        
        # Run SA from greedy result
        if time.time() < deadline - 5.0:
            sa_labels, sa_cut = _simulated_annealing(greedy_labels, adj, deadline)
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
    
    # Phase 3: Iterated local search with random perturbations
    if best_labels is not None and time.time() < deadline - 5.0:
        rng2 = np.random.RandomState(456)
        for _ in range(10):
            if time.time() > deadline - 3.0:
                break
            
            # Perturb best solution
            perturbed = best_labels.copy()
            num_flips = rng2.randint(1, 5)
            for _ in range(num_flips):
                v = rng2.randint(0, n)
                perturbed[v] = 1 - perturbed[v]
            
            # Run greedy + SA
            g_labels, g_cut = _greedy_improve(perturbed, adj, deadline)
            if g_cut > best_cut:
                best_cut = g_cut
                best_labels = g_labels.copy()
            
            if time.time() < deadline - 3.0:
                s_labels, s_cut = _simulated_annealing(g_labels, adj, deadline)
                if s_cut > best_cut:
                    best_cut = s_cut
                    best_labels = s_labels.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
