"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Fixed safety margin so we return well before the 150 s wall-clock kill.
DEADLINE_MARGIN = 5.0


def _cut_size(labels, adj):
    """Compute the cut size for a given labeling."""
    diff = (labels[None, :] != labels[:, None])
    return int(np.sum(adj * diff))


def _flip_gain(v, labels, adj, row_sums):
    """
    Compute the gain in cut size if vertex v is flipped.
    gain = 2 * (deg(v) - same_side_neighbors) - deg(v)
         = 2 * other_side_neighbors - deg(v)
    where same_side_neighbors = sum(adj[v, labels == labels[v]])
    """
    same = int(adj[v, labels == labels[v]].sum())
    deg = int(row_sums[v])
    return 2 * (deg - same) - deg


def _local_search(labels, adj, row_sums, deadline):
    """
    Greedy local search: repeatedly flip the vertex with the best positive gain
    until no improving flip exists or deadline is reached.
    """
    n = len(labels)
    improved = True
    while improved:
        improved = False
        for v in range(n):
            if time.time() > deadline:
                return labels
            g = _flip_gain(v, labels, adj, row_sums)
            if g > 0:
                labels[v] = 1 - labels[v]
                improved = True
    return labels


def _random_restarts(adj, row_sums, deadline, n_restarts=20):
    """
    Perform multiple random restarts with local search, keeping the best result.
    """
    n = len(adj)
    best_labels = None
    best_cut = -1
    rng = np.random.default_rng(42)

    # Start with a few structured initializations
    initializations = [
        np.arange(n) % 2,  # alternating
        np.zeros(n, dtype=int),  # all zeros
        np.ones(n, dtype=int),  # all ones
    ]
    # Add random initializations
    for _ in range(n_restarts - len(initializations)):
        initializations.append(rng.integers(0, 2, size=n))

    for init_labels in initializations:
        if time.time() > deadline:
            break
        labels = init_labels.copy()
        labels = _local_search(labels, adj, row_sums, deadline)
        cut = _cut_size(labels, adj)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    return best_labels


def partition(adjacency):
    """
    Partition vertices into two groups to maximize the cut size.
    Uses multi-start greedy local search with a deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    # Convert to numpy array for efficient operations
    adj = np.asarray(adjacency, dtype=bool)
    n = len(adj)

    # Precompute row sums (degrees)
    row_sums = adj.sum(axis=1)

    # Run multi-start local search
    best_labels = _random_restarts(adj, row_sums, deadline, n_restarts=50)

    # Final check: ensure we haven't exceeded deadline
    if time.time() > deadline:
        # Return whatever we have
        pass

    return best_labels
# EVOLVE-BLOCK-END
