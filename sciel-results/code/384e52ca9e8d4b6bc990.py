"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    cross_mask = labels[:, None] != labels[None, :]
    return int((adj * cross_mask).sum() // 2)


def _spectral_init(adj, n, k=1):
    """
    Initialize using the (k+1)-th eigenvector of the graph Laplacian.
    Returns a 0/1 label array.
    """
    degrees = np.sum(adj, axis=1).astype(float)
    D = np.diag(degrees)
    A = adj.astype(float)
    L = D - A

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        if k >= len(eigenvectors):
            k = len(eigenvectors) - 1
        fiedler = eigenvectors[:, k]
        labels = (fiedler > 0).astype(np.int8)
    except Exception:
        labels = np.random.randint(0, 2, size=n).astype(np.int8)

    return labels


def _random_init(n):
    """Random 0/1 initialization."""
    return np.random.randint(0, 2, size=n).astype(np.int8)


def _vectorized_greedy_improve(adj, labels, max_passes=100):
    """
    Vectorized greedy local search: repeatedly flip all vertices that improve the cut simultaneously.
    This is a parallel greedy approach that converges fast.
    Returns improved labels.
    """
    n = len(adj)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()
    degree = adj_int.sum(axis=1)

    for _ in range(max_passes):
        # For each vertex v, count edges to same side
        same_side_mask = (labels[None, :] == labels[:, None]).astype(np.int8)
        same_counts = (adj_int * same_side_mask).sum(axis=1)
        other_counts = degree - same_counts

        # Flipping v: gain = same_counts[v] - other_counts[v]
        # If gain > 0, flipping v improves the cut
        gains = same_counts - other_counts
        to_flip = np.where(gains > 0)[0]

        if len(to_flip) == 0:
            break

        labels[to_flip] = 1 - labels[to_flip]

    return labels


def _single_flip_greedy(adj, labels, max_passes=200):
    """
    Sequential single-flip greedy: flip the vertex with the best positive gain each step.
    More thorough than parallel greedy but slower.
    """
    n = len(adj)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()
    degree = adj_int.sum(axis=1)

    for _ in range(max_passes):
        same_side_mask = (labels[None, :] == labels[:, None]).astype(np.int8)
        same_counts = (adj_int * same_side_mask).sum(axis=1)
        other_counts = degree - same_counts
        gains = same_counts - other_counts

        max_gain = gains.max()
        if max_gain <= 0:
            break

        idx = np.argmax(gains)
        labels[idx] = 1 - labels[idx]

    return labels


def _simulated_annealing(adj, labels, start_temp=8.0, cool_rate=0.998, max_steps=8000):
    """
    Simulated annealing with precomputed neighbor structure for speed.
    Returns improved labels.
    """
    n = len(adj)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    temp = start_temp

    # Precompute neighbor lists as arrays for fast access
    neighbors = []
    for i in range(n):
        nbrs = np.where(adj_int[i] == 1)[0]
        neighbors.append(nbrs)

    for step in range(max_steps):
        v = np.random.randint(0, n)
        old_label = labels[v]
        new_label = 1 - old_label

        # Compute change in cut size by looking at neighbors
        nbrs = neighbors[v]
        if len(nbrs) == 0:
            delta = 0
        else:
            nbr_labels = labels[nbrs]
            same_count = (nbr_labels == old_label).sum()
            other_count = len(nbrs) - same_count
            delta = same_count - other_count

        new_cut = current_cut + delta

        if new_cut > current_cut:
            accept = True
        else:
            if temp < 1e-10:
                accept = False
            else:
                accept = np.random.random() < np.exp(delta / temp)

        if accept:
            labels[v] = new_label
            current_cut = new_cut
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()

        temp *= cool_rate

    return best_labels


def partition(adjacency):
    """
    Partition the graph into two sets to maximize the cut.
    Uses multiple spectral initializations + vectorized greedy + single-flip greedy + simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=bool)

    # Generate multiple diverse initializations
    inits = []

    # Spectral initializations with different eigenvectors
    for k in [1, 2, 3]:
        if time.time() > deadline - 60:
            break
        spec_labels = _spectral_init(adj, n, k=k)
        inits.append(spec_labels)

    # Alternating initialization
    alt_labels = np.arange(n) % 2
    inits.append(alt_labels)

    # Several random initializations
    for _ in range(4):
        if time.time() > deadline - 50:
            break
        inits.append(_random_init(n))

    best_labels = None
    best_cut = -1

    for init_labels in inits:
        if time.time() > deadline - 30:
            break

        # Vectorized greedy local search (fast, parallel)
        greedy_labels = _vectorized_greedy_improve(adj, init_labels, max_passes=80)
        greedy_cut = _cut_size(adj, greedy_labels)

        if greedy_cut > best_cut:
            best_cut = greedy_cut
            best_labels = greedy_labels.copy()

        # Single-flip greedy refinement (more thorough)
        if time.time() < deadline - 25:
            sf_labels = _single_flip_greedy(adj, greedy_labels, max_passes=100)
            sf_cut = _cut_size(adj, sf_labels)

            if sf_cut > best_cut:
                best_cut = sf_cut
                best_labels = sf_labels.copy()

    # Simulated annealing from the best result found so far
    if best_labels is not None and time.time() < deadline - 20:
        remaining = deadline - 20 - time.time()
        # Estimate steps: each SA step is ~1-2 microseconds for n=100
        # Be conservative: assume 5us per step
        max_steps = int(min(15000, max(3000, remaining * 1000000 / 5)))
        sa_labels = _simulated_annealing(adj, best_labels, start_temp=6.0, cool_rate=0.999, max_steps=max_steps)
        sa_cut = _cut_size(adj, sa_labels)

        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()

    # Final vectorized greedy pass on the best found
    if best_labels is not None and time.time() < deadline - 5:
        final_labels = _vectorized_greedy_improve(adj, best_labels, max_passes=100)
        final_cut = _cut_size(adj, final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels.copy()

        # One more single-flip pass if time permits
        if time.time() < deadline - 2:
            sf_final = _single_flip_greedy(adj, best_labels, max_passes=50)
            sf_final_cut = _cut_size(adj, sf_final)
            if sf_final_cut > best_cut:
                best_cut = sf_final_cut
                best_labels = sf_final.copy()

    if best_labels is None:
        # Fallback
        best_labels = np.arange(n) % 2

    return best_labels
# EVOLVE-BLOCK-END
