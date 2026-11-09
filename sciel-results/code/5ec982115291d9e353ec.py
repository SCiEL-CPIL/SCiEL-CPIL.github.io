"""Maximum Cut using Spectral Initialization, 1-Flip, 2-Swap, and Perturbation Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # Vectorized: each crossing edge is counted twice (u,v) and (v,u)
    mask = (labels[:, None] != labels[None, :]) & adj
    return mask.sum() // 2


def _spectral_init(adj, n, rng):
    """
    Generate initial partition using spectral embedding.
    Use a random projection of the top eigenvectors of the Laplacian.
    """
    degrees = adj.sum(axis=1)
    L = np.diag(degrees) - adj
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return rng.integers(0, 2, size=n)
    
    # Use the top k eigenvectors (excluding the constant one)
    k = min(n, 8)
    V = eigenvectors[:, 1:k+1]
    
    # Random projection
    w = rng.standard_normal(k)
    v = V @ w
    
    labels = (v > 0).astype(int)
    return labels


def _greedy_1_flip(adj, labels, max_iter=200):
    """
    Greedy local search: repeatedly flip the vertex that gives the best gain.
    """
    n = len(labels)
    labels = labels.copy()
    degrees = adj.sum(axis=1).astype(float)
    
    for _ in range(max_iter):
        improved = False
        best_gain = 0
        best_v = -1
        
        # Vectorized gain computation
        # opposite[v] = number of neighbors of v in the opposite partition
        opposite = adj @ (1 - labels)
        same = degrees - opposite
        gains = same - opposite  # = degrees - 2*opposite
        
        best_v = np.argmax(gains)
        best_gain = gains[best_v]
        
        if best_gain <= 0:
            break
        
        labels[best_v] = 1 - labels[best_v]
        improved = True
    
    return labels


def _greedy_2_swap(adj, labels, max_iter=100):
    """
    Greedy local search: try swapping pairs of vertices (one from each side).
    """
    n = len(labels)
    labels = labels.copy()
    degrees = adj.sum(axis=1).astype(float)
    
    for _ in range(max_iter):
        improved = False
        
        # Get lists of vertices in each partition
        side0 = np.where(labels == 0)[0]
        side1 = np.where(labels == 1)[0]
        
        if len(side0) == 0 or len(side1) == 0:
            break
        
        best_gain = 0
        best_pair = None
        
        # For each pair (u in side0, v in side1), compute gain of swapping
        # This is O(n^2) per iteration, but n=100 so it's manageable
        # Vectorized: for each u in side0, compute gain of swapping u with each v in side1
        
        # Gain of swapping u and v:
        # u moves from 0 to 1, v moves from 1 to 0
        # For u: was in side 0, now in side 1
        #   Lost: neighbors in side 1 (they were crossing, now not)
        #   Gained: neighbors in side 0 (they were not crossing, now crossing)
        #   But we need to account for u and v being neighbors of each other
        # For v: was in side 1, now in side 0
        #   Lost: neighbors in side 0 (they were crossing, now not)
        #   Gained: neighbors in side 1 (they were not crossing, now crossing)
        
        # Let's compute this more carefully
        # Let A be the adjacency matrix
        # Let S0 = {u: labels[u]=0}, S1 = {u: labels[u]=1}
        # For u in S0, v in S1:
        #   Before swap: u in S0, v in S1
        #   After swap: u in S1, v in S0
        #   Edges affected:
        #     (u, x) for x != v: if x in S0, edge was internal, now crossing (+1); if x in S1, edge was crossing, now internal (-1)
        #     (v, x) for x != u: if x in S0, edge was crossing, now internal (-1); if x in S1, edge was internal, now crossing (+1)
        #     (u, v): if they are neighbors, edge was crossing, now internal (-1)
        
        # Gain = [neighbors of u in S0 (excluding v)] - [neighbors of u in S1 (excluding v)]
        #       + [neighbors of v in S1 (excluding u)] - [neighbors of v in S0 (excluding u)]
        #       - adj[u,v]  (if u and v are neighbors, the edge between them stops crossing)
        
        # neighbors of u in S0 (excluding v) = adj[u] @ (labels == 0) - adj[u,v] * (labels[v]==0)
        # But v is in S1, so labels[v] != 0, so adj[u,v] * (labels[v]==0) = 0
        # So neighbors of u in S0 = adj[u] @ (labels == 0)
        # neighbors of u in S1 (excluding v) = adj[u] @ (labels == 1) - adj[u,v]
        
        # Similarly for v.
        
        # Let's compute this vectorized for all pairs
        # This is complex, so let's do it with a double loop but limit the number of pairs checked
        
        # For efficiency, let's only check a random subset of pairs if the partitions are large
        if len(side0) * len(side1) > 5000:
            # Random sampling
            num_samples = 2000
            idx_u = rng.integers(0, len(side0), size=num_samples)
            idx_v = rng.integers(0, len(side1), size=num_samples)
            candidates = list(zip(side0[idx_u], side1[idx_v]))
        else:
            candidates = list(zip(side0, side1))
        
        for u, v in candidates:
            if u == v:
                continue
            
            # Compute gain
            # neighbors of u in S0 (excluding v)
            n_u_s0 = adj[u] @ (labels == 0)  # v is in S1, so this doesn't include v
            # neighbors of u in S1 (excluding v)
            n_u_s1 = adj[u] @ (labels == 1) - adj[u, v]
            
            # neighbors of v in S1 (excluding u)
            n_v_s1 = adj[v] @ (labels == 1)  # u is in S0, so this doesn't include u
            # neighbors of v in S0 (excluding u)
            n_v_s0 = adj[v] @ (labels == 0) - adj[v, u]
            
            gain = (n_u_s0 - n_u_s1) + (n_v_s1 - n_v_s0) - adj[u, v]
            
            if gain > best_gain:
                best_gain = gain
                best_pair = (u, v)
        
        if best_pair is None or best_gain <= 0:
            break
        
        u, v = best_pair
        labels[u] = 1 - labels[u]
        labels[v] = 1 - labels[v]
        improved = True
    
    return labels


def _perturbation_search(adj, labels, rng, num_restarts=10):
    """
    Perturbation search: flip a random subset of vertices and re-run local search.
    """
    n = len(labels)
    best_labels = labels.copy()
    best_cut = _cut_size(adj, best_labels)
    
    for _ in range(num_restarts):
        # Flip a random subset (10-30% of vertices)
        frac = rng.uniform(0.1, 0.3)
        num_flips = int(n * frac)
        flip_indices = rng.choice(n, size=num_flips, replace=False)
        
        perturbed = labels.copy()
        perturbed[flip_indices] = 1 - perturbed[flip_indices]
        
        # Local search
        improved = _greedy_1_flip(adj, perturbed, max_iter=100)
        improved = _greedy_2_swap(adj, improved, max_iter=50)
        
        cut = _cut_size(adj, improved)
        if cut > best_cut:
            best_cut = cut
            best_labels = improved.copy()
    
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    adj = np.array(adjacency, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    rng = np.random.RandomState(42)
    
    best_labels = None
    best_cut = -1
    
    # Strategy:
    # 1. Generate multiple initial solutions using spectral embedding
    # 2. Apply 1-flip local search
    # 3. Apply 2-swap local search
    # 4. Apply perturbation search to the best solution
    # 5. Return the best found
    
    num_spectral_inits = 30
    
    for i in range(num_spectral_inits):
        if time.time() > deadline - 10.0:
            break
        
        # Generate initial partition
        if i == 0:
            # First one: use Fiedler vector directly
            try:
                degrees = adj.sum(axis=1)
                L = np.diag(degrees) - adj
                eigenvalues, eigenvectors = np.linalg.eigh(L)
                fiedler = eigenvectors[:, 1]
                labels = (fiedler > 0).astype(int)
            except:
                labels = rng.integers(0, 2, size=n)
        else:
            labels = _spectral_init(adj, n, rng)
        
        # 1-flip local search
        labels = _greedy_1_flip(adj, labels, max_iter=200)
        # 2-swap local search
        labels = _greedy_2_swap(adj, labels, max_iter=50)
        
        cut = _cut_size(adj, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # If we have time, apply perturbation search to the best solution
    if best_labels is not None and time.time() < deadline - 5.0:
        pert_labels, pert_cut = _perturbation_search(adj, best_labels, rng, num_restarts=5)
        if pert_cut > best_cut:
            best_cut = pert_cut
            best_labels = pert_labels
    
    # Final polish
    if best_labels is not None and time.time() < deadline - 1.0:
        best_labels = _greedy_1_flip(adj, best_labels, max_iter=50)
        best_labels = _greedy_2_swap(adj, best_labels, max_iter=20)
    
    if best_labels is None:
        # Fallback
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
