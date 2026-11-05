"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Safety margin before the 150 s wall-clock kill.
DEADLINE_MARGIN = 5.0


def _cut_size(labels, adj):
    """Compute the cut size (number of crossing edges) for a labeling."""
    # adj is a bool matrix; crossing edges = adj where labels differ
    diff = labels[None, :] != labels[:, None]
    return int(np.sum(adj & diff))


def _vectorized_gains(labels, adj):
    """
    Compute the gain for flipping each vertex in a vectorized manner.
    gain[v] = (# neighbors on same side as v) - (# neighbors on other side)
    If gain > 0, flipping v increases the cut size.
    """
    n = len(labels)
    # For each vertex v, gain = 2 * (num neighbors with same label) - deg(v)
    # Wait, let's re-derive:
    # Let s_v be the label of v (0 or 1).
    # Let N_v be the set of neighbors of v.
    # Current contribution to cut from edges incident to v:
    #   sum_{u in N_v} [s_u != s_v]
    # If we flip v, new contribution:
    #   sum_{u in N_v} [s_u != (1-s_v)] = sum_{u in N_v} [s_u == s_v]
    # Change = new - old = sum_{u in N_v} [s_u == s_v] - sum_{u in N_v} [s_u != s_v]
    # Let same = sum_{u in N_v} [s_u == s_v]
    # Let other = sum_{u in N_v} [s_u != s_v]
    # deg(v) = same + other
    # Change = same - other = same - (deg - same) = 2*same - deg
    
    # Vectorized computation:
    # same[v] = sum_u adj[v,u] * (labels[u] == labels[v])
    # We can compute this as:
    # same = adj @ (labels == labels[:, None]).T  -- no, that's not right.
    
    # Let's use:
    # same_matrix[v, u] = adj[v, u] * (labels[v] == labels[u])
    # same[v] = sum_u same_matrix[v, u]
    
    # More efficient:
    # Let one_hot be a matrix where one_hot[v, 0] = 1 if labels[v]==0 else 0, one_hot[v, 1] = 1 if labels[v]==1 else 0.
    # Then same[v] = sum_u adj[v, u] * one_hot[u, labels[v]]
    
    # Even more efficient using matrix multiplication:
    # Let L be a vector of labels.
    # Let A be the adjacency matrix.
    # same[v] = sum_u A[v, u] * (L[u] == L[v])
    
    # We can compute:
    # A @ L gives for each v: sum_u A[v, u] * L[u]
    # But we need sum_u A[v, u] * (L[u] == L[v])
    
    # Note that (L[u] == L[v]) is 1 if L[u] and L[v] are equal, 0 otherwise.
    # This is not linear in L.
    
    # Alternative approach:
    # gain[v] = 2 * same[v] - deg[v]
    # same[v] = sum_{u: L[u]==L[v]} A[v, u]
    
    # We can compute this as:
    # same = np.zeros(n)
    # for label in [0, 1]:
    #     mask = (labels == label)
    #     same[mask] = (adj[mask, :] @ mask)
    
    # This is O(n^2) but with low constant.
    
    # Let's try a fully vectorized approach:
    # Create a matrix M where M[v, u] = 1 if labels[v] == labels[u] else 0.
    # Then same = adj @ M (element-wise? No, M is not a simple matrix product).
    
    # Actually, M[v, u] = 1 - |labels[v] - labels[u]|
    # This is not helpful for matrix multiplication.
    
    # Let's stick with the loop over labels:
    same = np.zeros(len(labels), dtype=int)
    for lbl in (0, 1):
        mask = (labels == lbl)
        if np.any(mask):
            same[mask] = (adj[mask, :] @ mask.astype(int))
    
    deg = adj.sum(axis=1)
    gains = 2 * same - deg
    return gains


def _greedy_polish(labels, adj, deadline):
    """
    Greedy local search: repeatedly flip the vertex with the best positive gain.
    Uses vectorized gain computation for speed.
    """
    n = len(labels)
    labels = labels.copy()
    best_cut = _cut_size(labels, adj)
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        gains = _vectorized_gains(labels, adj)
        best_gain = np.max(gains)
        if best_gain > 0:
            best_v = np.argmax(gains)
            labels[best_v] = 1 - labels[best_v]
            best_cut += int(best_gain)
            improved = True
    
    return labels, best_cut


def _simulated_annealing(labels, adj, deadline, t_start=10.0, t_end=0.01, cooling=0.99):
    """
    Simulated annealing for Maximum Cut.
    Treats vertex labels as spins and edge weights as coupling constants.
    """
    n = len(labels)
    labels = labels.copy()
    best_labels = labels.copy()
    best_cut = _cut_size(labels, adj)
    
    t = t_start
    
    while t > t_end:
        if time.time() > deadline:
            break
        
        # Pick a random vertex
        v = np.random.randint(n)
        
        # Compute gain for flipping v
        gains = _vectorized_gains(labels, adj)
        delta = gains[v]
        
        # Accept with probability 1 if delta > 0, else exp(-delta/t)
        if delta > 0 or np.random.rand() < np.exp(-delta / t):
            labels[v] = 1 - labels[v]
            new_cut = best_cut + int(delta)
            if new_cut > best_cut:
                best_cut = new_cut
                best_labels = labels.copy()
        
        t *= cooling
    
    return best_labels, best_cut


def partition(adjacency):
    """
    Partition vertices into two sets to maximize the number of crossing edges.
    Uses a combination of greedy local search and simulated annealing with random restarts.
    """
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    best_labels = None
    best_cut = -1
    
    # Strategy 1: Alternating initialization + greedy polish
    labels_alt = np.arange(n) % 2
    labels_alt, cut_alt = _greedy_polish(labels_alt, adj, deadline)
    best_labels = labels_alt.copy()
    best_cut = cut_alt
    
    # Strategy 2: Random restarts with greedy polish and simulated annealing
    remaining = deadline - time.time()
    if remaining > 2.0:
        for r in range(50):  # Up to 50 restarts, but check deadline
            if time.time() > deadline:
                break
            
            # Random initial labeling
            labels = np.random.randint(0, 2, size=n)
            
            # Greedy polish
            labels_greedy, cut_greedy = _greedy_polish(labels, adj, deadline)
            
            # Simulated annealing from the greedy solution
            labels_sa, cut_sa = _simulated_annealing(labels_greedy, adj, deadline)
            
            # Update best
            if cut_sa > best_cut:
                best_cut = cut_sa
                best_labels = labels_sa.copy()
            elif cut_greedy > best_cut:
                best_cut = cut_greedy
                best_labels = labels_greedy.copy()
    
    return best_labels
# EVOLVE-BLOCK-END
