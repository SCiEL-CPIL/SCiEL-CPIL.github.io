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


def _flip_gain(v, labels, adj):
    """
    Gain in cut size if vertex v is flipped.
    gain = 2 * (# neighbors on other side) - deg(v)
         = 2 * other - (same + other) = other - same
    Wait: if we flip v, edges from v to same-side neighbors become crossing (+1 each),
          edges from v to other-side neighbors stop crossing (-1 each).
    So gain = same - other.
    """
    same = int(adj[v, labels == labels[v]].sum())
    other = int(adj[v, labels != labels[v]].sum())
    return same - other


def _local_search(labels, adj, deadline):
    """
    Greedy local search: repeatedly flip the vertex with the best positive gain.
    Returns improved labels and the cut size.
    """
    n = len(labels)
    labels = labels.copy()
    best_cut = _cut_size(labels, adj)
    
    # Precompute row sums (degrees)
    deg = adj.sum(axis=1)
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        best_gain = 0
        best_v = -1
        for v in range(n):
            gain = _flip_gain(v, labels, adj)
            if gain > best_gain:
                best_gain = gain
                best_v = v
        if best_v != -1:
            labels[best_v] = 1 - labels[best_v]
            best_cut += best_gain
            improved = True
    
    return labels, best_cut


def _random_restarts(adj, n_restarts, deadline):
    """
    Run multiple random restarts of local search, keeping the best solution.
    """
    n = len(adj)
    best_labels = None
    best_cut = -1
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        # Random initial labeling
        labels = np.random.randint(0, 2, size=n)
        labels, cut = _local_search(labels, adj, deadline)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        # Fallback: alternating start
        labels = np.arange(n) % 2
        labels, best_cut = _local_search(labels, adj, deadline)
        best_labels = labels
    
    return best_labels


def partition(adjacency):
    """
    Partition vertices into two sets to maximize the number of crossing edges.
    Uses greedy local search with random restarts.
    """
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Try a few different strategies within the time budget
    # Start with alternating initialization
    labels_alt = np.arange(n) % 2
    labels_alt, cut_alt = _local_search(labels_alt, adj, deadline)
    best_labels = labels_alt
    best_cut = cut_alt
    
    # Now do random restarts with the remaining time
    remaining = deadline - time.time()
    if remaining > 1.0:
        # Estimate how many restarts we can do
        # Each restart takes roughly n * (some constant) time
        # Be conservative: try up to 20 restarts, but check deadline each time
        for r in range(20):
            if time.time() > deadline:
                break
            labels = np.random.randint(0, 2, size=n)
            labels, cut = _local_search(labels, adj, deadline)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
    
    return best_labels
# EVOLVE-BLOCK-END
