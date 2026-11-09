"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(labels, adj):
    """Compute cut size: number of edges crossing the partition."""
    # Edge crosses if labels[i] != labels[j] and adj[i,j] == 1
    # We count each edge once (undirected), so divide by 2
    diff = (labels[:, None] != labels[None, :]).astype(np.int8)
    return int((adj * diff).sum() // 2)


def _spectral_init(adj, n, rng):
    """Initialize using spectral embedding (Fiedler vector)."""
    try:
        # Compute degree matrix
        degrees = adj.sum(axis=1).astype(np.float64)
        d_max = max(degrees.max(), 1.0)
        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        # But for sparse graphs, use unnormalized: L = D - A
        D = np.diag(degrees)
        L = D - adj.astype(np.float64)
        
        # Find smallest non-zero eigenvalue (Fiedler vector)
        # Use eigvalsh for symmetric matrix
        eigvals, eigvecs = np.linalg.eigh(L)
        # The second smallest eigenvalue's eigenvector is the Fiedler vector
        # eigvals are sorted in ascending order
        fiedler = eigvecs[:, 1]
        
        # Partition: vertices with positive Fiedler value go to 0, negative to 1
        # Add small random perturbation to break ties
        perturbation = rng.normal(0, 0.001, size=n)
        fiedler_perturbed = fiedler + perturbation
        labels = (fiedler_perturbed > 0).astype(np.int8)
        return labels
    except Exception:
        # Fallback to random
        return rng.integers(0, 2, size=n).astype(np.int8)


def _local_improvement(labels, adj, n, deadline, rng, tabu_length=10):
    """
    Cerny tabu search for maximum cut.
    Uses a fixed-length tabu list to prohibit recent vertex flips,
    allowing the search to accept worsening moves to escape local optima.
    """
    labels = labels.copy()
    current_cut = _cut_size(labels, adj)
    best_cut = current_cut
    best_labels = labels.copy()
    
    # Precompute neighbor lists for efficiency
    # For each vertex, store list of neighbors
    neighbors = [np.where(adj[i] > 0)[0].tolist() for i in range(n)]
    
    # Tabu list: store (vertex, iteration) for recent moves
    tabu_list = []
    iteration = 0
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
        
        # Find best move (flip a vertex)
        best_gain = -np.inf
        best_vertex = -1
        best_is_tabu = True
        
        # Random order to break symmetry
        order = rng.permutation(n)
        
        for v in order:
            # Check if v is in tabu list
            is_tabu = any(tabu[0] == v for tabu in tabu_list)
            
            if is_tabu:
                # Allow tabu move only if it improves best known
                continue
            
            # Compute gain from flipping v
            # Current: v contributes (n_other) to cut where n_other = neighbors on other side
            # After flip: v contributes (n_same) to cut where n_same = neighbors on same side
            # Gain = n_same - n_other = 2*n_same - degree(v)
            same_side = labels == labels[v]
            n_same = adj[v, same_side].sum()
            n_other = adj[v, ~same_side].sum()
            gain = int(n_same - n_other)
            
            if gain > best_gain:
                best_gain = gain
                best_vertex = v
                best_is_tabu = is_tabu
        
        if best_vertex == -1:
            # All moves are tabu or no improvement possible
            # Clear tabu list and try again
            tabu_list = []
            no_improve_count += 1
            continue
        
        # Apply the move
        old_label = labels[best_vertex]
        labels[best_vertex] = 1 - old_label
        current_cut += best_gain
        
        # Update tabu list
        tabu_list.append((best_vertex, iteration))
        if len(tabu_list) > tabu_length:
            tabu_list.pop(0)
        
        # Update best
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        iteration += 1
    
    return best_labels, best_cut


def _two_swap_improvement(labels, adj, n, deadline, rng):
    """
    2-swap local search: try swapping pairs of vertices.
    This can escape 1-flip local optima.
    """
    labels = labels.copy()
    current_cut = _cut_size(labels, adj)
    best_cut = current_cut
    best_labels = labels.copy()
    
    # Precompute for efficiency
    improved = True
    max_iterations = 10000
    iteration = 0
    
    while improved and time.time() < deadline and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try random pairs of vertices
        for _ in range(50):
            if time.time() > deadline:
                break
            
            # Pick two random vertices from different sides
            side0 = np.where(labels == 0)[0]
            side1 = np.where(labels == 1)[0]
            
            if len(side0) == 0 or len(side1) == 0:
                break
            
            i = rng.integers(0, len(side0))
            j = rng.integers(0, len(side1))
            i = side0[i]
            j = side1[j]
            
            # Compute gain from swapping i and j
            # This is complex; approximate by checking if it helps
            # Simple heuristic: if i has more neighbors on side 1 and j has more on side 0
            i_same = adj[i, labels == 0].sum()
            i_other = adj[i, labels == 1].sum()
            j_same = adj[j, labels == 1].sum()
            j_other = adj[j, labels == 0].sum()
            
            # After swap: i moves to side 1, j moves to side 0
            # i's new same-side count: i_other (neighbors now on side 1)
            # j's new same-side count: j_other (neighbors now on side 0)
            # But we need to account for the edge between i and j
            has_edge_ij = adj[i, j]
            
            # Approximate gain
            gain_i = i_other - i_same  # i gains i_other - i_same
            gain_j = j_other - j_same  # j gains j_other - j_same
            # If there's an edge between i and j, it was crossing before, now it's not
            edge_correction = -has_edge_ij
            
            total_gain = gain_i + gain_j + 2 * edge_correction
            
            if total_gain > 0:
                # Apply swap
                labels[i], labels[j] = labels[j], labels[i]
                current_cut += total_gain
                improved = True
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
    
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for faster operations
    adj = np.array(adjacency, dtype=np.int8)
    
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_labels = None
    best_cut = -1
    
    rng = np.random.default_rng()
    
    # Multiple restarts with different strategies
    n_restarts = 30
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Choose initialization strategy
        if restart % 3 == 0:
            # Spectral initialization
            labels = _spectral_init(adj, n, rng)
        elif restart % 3 == 1:
            # Random initialization
            labels = rng.integers(0, 2, size=n).astype(np.int8)
        else:
            # Random with bias
            p = rng.uniform(0.3, 0.7)
            labels = (rng.random(n) < p).astype(np.int8)
        
        # Phase 1: Local improvement with tabu search
        labels_tabu, cut_tabu = _local_improvement(labels, adj, n, deadline, rng, tabu_length=15)
        
        # Phase 2: 2-swap improvement
        labels_swap, cut_swap = _two_swap_improvement(labels_tabu, adj, n, deadline, rng)
        
        # Evaluate
        c = cut_swap
        if c > best_cut:
            best_cut = c
            best_labels = labels_swap.copy()
    
    # If we didn't find any valid solution, use alternating
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
