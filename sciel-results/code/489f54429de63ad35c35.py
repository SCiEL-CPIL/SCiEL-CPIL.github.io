"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

# Safety margin before the 150 s wall-clock kill.
DEADLINE_MARGIN = 5.0


def _cut_size(labels, adj):
    """Compute the cut size (number of crossing edges) for a labeling."""
    diff = labels[None, :] != labels[:, None]
    return int(np.sum(adj & diff))


def _vectorized_gains(labels, adj, Ax):
    """
    Compute flip gains for all vertices simultaneously.
    Ax = adj @ labels (vector of neighbor label sums).
    
    gain[v] = (2*labels[v] - 1) * (2*Ax[v] - deg[v])
    """
    deg = adj.sum(axis=1)
    sign = 2.0 * labels.astype(np.float64) - 1.0
    return sign * (2.0 * Ax - deg)


def _update_Ax(v, labels, adj, Ax):
    """Update Ax when vertex v is flipped. Ax = adj @ labels."""
    delta = 1.0 - 2.0 * labels[v]
    Ax[adj[v, :]] += delta
    return Ax


def _greedy_polish(labels, adj, deadline):
    """
    Fast greedy local search using vectorized gain computation.
    """
    n = len(labels)
    labels = labels.copy().astype(np.float64)
    Ax = adj.astype(np.float64) @ labels
    best_cut = _cut_size(labels.astype(int), adj)
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        gains = _vectorized_gains(labels, adj, Ax)
        best_gain = np.max(gains)
        if best_gain > 1e-10:
            best_v = int(np.argmax(gains))
            _update_Ax(best_v, labels, adj, Ax)
            labels[best_v] = 1.0 - labels[best_v]
            best_cut += int(best_gain)
            improved = True
    
    return labels.astype(int), best_cut


def _pairwise_flip_polish(labels, adj, deadline):
    """
    Local search with pairwise flip moves to escape single-flip local optima.
    """
    n = len(labels)
    labels = labels.copy().astype(np.float64)
    Ax = adj.astype(np.float64) @ labels
    current_cut = _cut_size(labels.astype(int), adj)
    best_cut = current_cut
    best_labels = labels.copy()
    
    deg = adj.sum(axis=1)
    
    # Precompute adjacency as list of lists for faster neighbor iteration
    adj_list = [np.where(adj[i, :])[0] for i in range(n)]
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        
        # Try single flips first
        gains = _vectorized_gains(labels, adj, Ax)
        best_gain = np.max(gains)
        if best_gain > 1e-10:
            best_v = int(np.argmax(gains))
            _update_Ax(best_v, labels, adj, Ax)
            labels[best_v] = 1.0 - labels[best_v]
            current_cut += int(best_gain)
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
            improved = True
            continue
        
        # Try pairwise flips
        # Sample a few random pairs to find improving moves
        rng = np.random.default_rng()
        found_pair = False
        for _ in range(50):
            if time.time() > deadline:
                break
            i = rng.integers(n)
            j = rng.integers(n)
            if i == j:
                continue
            
            # Compute gain for flipping both i and j
            # Current: labels[i], labels[j]
            # After flip: 1-labels[i], 1-labels[j]
            
            # Gain from flipping i alone (given current state)
            gain_i = gains[i]
            # Gain from flipping j alone (given current state)
            gain_j = gains[j]
            
            # Interaction term: if i and j are neighbors, flipping both has an extra effect
            # If adj[i,j] == 1:
            #   If labels[i] == labels[j]: after flip, they're still same side, edge doesn't cross. Before: didn't cross. No change.
            #   If labels[i] != labels[j]: after flip, they're still different side, edge crosses. Before: crossed. No change.
            # Wait, that's not right. Let me think more carefully.
            
            # Actually, the gain formula for flipping vertex v assumes other vertices stay fixed.
            # When flipping two vertices i and j simultaneously, the gain is not simply gain_i + gain_j
            # because the interaction between i and j changes.
            
            # Let's compute the exact gain for flipping both i and j:
            # For each neighbor u of i (u != j):
            #   If u is on same side as i currently, flipping i makes it cross (+1)
            #   If u is on other side as i currently, flipping i makes it not cross (-1)
            #   But if u == j, we need to account for j also flipping
            # Similarly for neighbors of j (u != i)
            # And the edge (i,j) itself
            
            # Simpler: compute cut before and after for just the affected edges
            # Affected edges: all edges incident to i or j
            
            # Let's just compute the gain directly
            li = labels[i]
            lj = labels[j]
            
            # Edges from i to all neighbors except j
            ni = adj_list[i]
            nj = adj_list[j]
            
            # Gain from i's edges (excluding j)
            gain_i_excl_j = 0
            for u in ni:
                if u == j:
                    continue
                if labels[u] == li:
                    gain_i_excl_j += 1  # was same side, now different
                else:
                    gain_i_excl_j -= 1  # was different side, now same
            
            # Gain from j's edges (excluding i)
            gain_j_excl_i = 0
            for u in nj:
                if u == i:
                    continue
                if labels[u] == lj:
                    gain_j_excl_i += 1
                else:
                    gain_j_excl_i -= 1
            
            # Edge (i,j) if it exists
            if adj[i, j]:
                if li == lj:
                    # Was same side, after flip still same side (both flipped)
                    edge_gain = 0
                else:
                    # Was different side, after flip still different side
                    edge_gain = 0
            else:
                edge_gain = 0
            
            total_gain = gain_i_excl_j + gain_j_excl_i + edge_gain
            
            if total_gain > 0:
                # Apply the flip
                _update_Ax(i, labels, adj, Ax)
                labels[i] = 1.0 - labels[i]
                _update_Ax(j, labels, adj, Ax)
                labels[j] = 1.0 - labels[j]
                current_cut += int(total_gain)
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
                improved = True
                found_pair = True
                break
        
        if found_pair:
            continue
    
    return best_labels.astype(int), best_cut


def _simulated_annealing(labels, adj, deadline, t0=10.0, t1=0.1, alpha=0.99):
    """
    Simulated annealing for MaxCut.
    labels: 0/1 array (will be converted to float internally)
    Returns improved labels and cut size.
    """
    n = len(labels)
    labels = labels.copy().astype(np.float64)
    Ax = adj.astype(np.float64) @ labels
    current_cut = _cut_size(labels.astype(int), adj)
    best_cut = current_cut
    best_labels = labels.copy()
    
    t = t0
    rng = np.random.default_rng()
    
    while t > t1:
        if time.time() > deadline:
            break
        
        # Pick a random vertex to flip
        v = rng.integers(n)
        
        # Compute gain for flipping v
        gains = _vectorized_gains(labels, adj, Ax)
        gain = gains[v]
        
        # Accept if gain > 0, or with probability exp(gain/t) if gain < 0
        if gain >= 0 or rng.random() < np.exp(gain / t):
            _update_Ax(v, labels, adj, Ax)
            labels[v] = 1.0 - labels[v]
            current_cut += int(gain)
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        
        t *= alpha
    
    # Final greedy polish from best state
    best_labels_int, best_cut = _greedy_polish(best_labels.astype(int), adj, deadline)
    
    return best_labels_int, best_cut


def _spectral_init(adj, n):
    """
    Spectral initialization using the leading eigenvector of the adjacency matrix.
    """
    try:
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        
        # Use the leading eigenvector of the adjacency matrix
        # For MaxCut, we want to partition based on the sign of the leading eigenvector
        # of the Laplacian or adjacency matrix
        
        # Simple approach: use random projection with a few iterations
        # Or use the fact that for regular graphs, the leading eigenvector of A gives a good cut
        
        # Let's use a simpler spectral method: power iteration on A
        x = np.random.randn(n)
        x /= np.linalg.norm(x)
        
        for _ in range(50):
            x_new = adj.astype(np.float64) @ x
            norm = np.linalg.norm(x_new)
            if norm < 1e-10:
                break
            x_new /= norm
            x = x_new
        
        # Partition by sign
        labels = (x >= 0).astype(int)
        
        # Ensure both sides are non-empty
        if labels.sum() == 0 or labels.sum() == n:
            labels = (np.arange(n) % 2).astype(int)
        
        return labels
    except Exception:
        return (np.arange(n) % 2).astype(int)


def partition(adjacency):
    """
    Partition vertices into two sets to maximize the number of crossing edges.
    Uses spectral initialization, simulated annealing, greedy polish, and pairwise flip polish.
    """
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    best_labels = None
    best_cut = -1
    
    # Strategy 1: Spectral initialization + greedy polish
    labels_spec = _spectral_init(adj, n)
    labels_spec, cut_spec = _greedy_polish(labels_spec, adj, deadline)
    best_labels = labels_spec
    best_cut = cut_spec
    
    # Strategy 2: Alternating initialization + greedy polish
    labels_alt = (np.arange(n) % 2).astype(int)
    labels_alt, cut_alt = _greedy_polish(labels_alt, adj, deadline)
    if cut_alt > best_cut:
        best_cut = cut_alt
        best_labels = labels_alt.copy()
    
    # Strategy 3: Multiple SA restarts with random initialization
    n_sa_restarts = 5
    for r in range(n_sa_restarts):
        if time.time() > deadline:
            break
        
        # Random initial labeling
        labels = np.random.randint(0, 2, size=n).astype(int)
        
        # Run simulated annealing
        labels_sa, cut_sa = _simulated_annealing(labels, adj, deadline, t0=5.0, t1=0.05, alpha=0.995)
        
        if cut_sa > best_cut:
            best_cut = cut_sa
            best_labels = labels_sa.copy()
    
    # Strategy 4: Use remaining time for pairwise flip polish from best solution
    if time.time() < deadline - 1.0:
        labels_final, cut_final = _pairwise_flip_polish(best_labels, adj, deadline)
        if cut_final > best_cut:
            best_cut = cut_final
            best_labels = labels_final
    
    # Strategy 5: Final greedy polish
    if time.time() < deadline - 0.5:
        labels_final2, cut_final2 = _greedy_polish(best_labels, adj, deadline)
        if cut_final2 > best_cut:
            best_cut = cut_final2
            best_labels = labels_final2
    
    return best_labels
# EVOLVE-BLOCK-END
