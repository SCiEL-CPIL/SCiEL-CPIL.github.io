"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

DEADLINE_MARGIN = 5.0


def _cut_size(labels, adj):
    """Compute the cut size (number of crossing edges) for a labeling."""
    diff = labels[None, :] != labels[:, None]
    return int(np.sum(adj & diff))


def _vectorized_gains(labels, adj):
    """
    Compute flip gain for every vertex in a vectorized way.
    gain[v] = same_side_neighbors[v] - other_side_neighbors[v]
    where same_side = neighbors of v that share v's label,
          other_side = neighbors of v that have the opposite label.
    """
    # adj @ labels gives, for each v, the count of neighbors with label 1
    # (since adj is bool, adj[v, :] @ labels = sum of labels of neighbors of v)
    # Let's think carefully:
    #   labels is 0/1.
    #   For vertex v with label l_v:
    #     neighbors with label 1: sum(adj[v, :] * labels)
    #     neighbors with label 0: deg[v] - sum(adj[v, :] * labels)
    #   If l_v == 1:
    #     same = neighbors with label 1 = sum(adj[v,:] * labels)
    #     other = neighbors with label 0 = deg[v] - sum(adj[v,:] * labels)
    #     gain = same - other = 2*sum(adj[v,:] * labels) - deg[v]
    #   If l_v == 0:
    #     same = neighbors with label 0 = deg[v] - sum(adj[v,:] * labels)
    #     other = neighbors with label 1 = sum(adj[v,:] * labels)
    #     gain = same - other = deg[v] - 2*sum(adj[v,:] * labels)
    #
    # Unified: gain[v] = (2*labels[v] - 1) * (2*sum(adj[v,:] * labels) - deg[v])
    #   Check: if labels[v]=1: (2*1-1)=1, gain = 2*sum - deg. ✓
    #   if labels[v]=0: (2*0-1)=-1, gain = -(2*sum - deg) = deg - 2*sum. ✓

    deg = adj.sum(axis=1)  # shape (n,)
    neighbor_label_sum = adj @ labels  # shape (n,): for each v, sum of labels of its neighbors
    # adj is bool, labels is int, so this works as matrix-vector product
    sign = 2 * labels - 1  # +1 for label 1, -1 for label 0
    gains = sign * (2 * neighbor_label_sum - deg)
    return gains


def _simulated_annealing(labels, adj, deadline):
    """
    Simulated annealing on the cut problem.
    labels: initial 0/1 array
    Returns improved labels and cut size.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(labels, adj)
    best_labels = labels.copy()
    best_cut = current_cut

    # Temperature schedule: geometric cooling
    # Start high enough to accept many moves, cool down to near 0
    T_start = 10.0
    T_end = 0.01
    # We'll do a fixed number of iterations, but check deadline
    # Estimate: each iteration is O(n) vectorized, should be fast
    # Try to do as many as possible within deadline
    n_iter = 2000  # will be capped by deadline

    rng = np.random.default_rng()

    for it in range(n_iter):
        if time.time() > deadline:
            break

        # Temperature for this iteration
        if it < n_iter - 1:
            T = T_start * (T_end / T_start) ** (it / max(n_iter - 1, 1))
        else:
            T = T_end

        # Compute gains for all vertices vectorized
        gains = _vectorized_gains(labels, adj)

        # Pick a random vertex to flip
        v = rng.integers(0, n)
        delta = gains[v]  # change in cut if we flip v

        if delta >= 0:
            # Always accept improving or neutral moves
            labels[v] = 1 - labels[v]
            current_cut += int(delta)
        else:
            # Accept with probability exp(delta / T)
            if T > 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    labels[v] = 1 - labels[v]
                    current_cut += int(delta)

        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()

    # Final greedy polish from the best found state
    best_labels, best_cut = _greedy_polish(best_labels, adj, deadline)

    return best_labels, best_cut


def _greedy_polish(labels, adj, deadline):
    """
    Greedy local search: repeatedly flip the vertex with the best positive gain.
    Vectorized gain computation makes each pass fast.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(labels, adj)
    best_cut = current_cut
    best_labels = labels.copy()

    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        gains = _vectorized_gains(labels, adj)
        max_gain = int(gains.max())
        if max_gain > 0:
            best_v = int(np.argmax(gains))
            labels[best_v] = 1 - labels[best_v]
            current_cut += max_gain
            improved = True
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()

    return best_labels, best_cut


def partition(adjacency):
    """
    Partition vertices into two sets to maximize the number of crossing edges.
    Uses simulated annealing with greedy polish and multi-start.
    """
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=bool)

    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    best_labels = None
    best_cut = -1

    rng = np.random.default_rng(42)

    # Phase 1: Alternating start + greedy polish (fast baseline)
    labels_alt = np.arange(n) % 2
    labels_alt, cut_alt = _greedy_polish(labels_alt, adj, deadline)
    best_labels = labels_alt.copy()
    best_cut = cut_alt

    # Phase 2: Random restarts with simulated annealing
    # Each SA run does many iterations; check deadline before each restart
    n_restarts_max = 50
    for r in range(n_restarts_max):
        if time.time() > deadline - 2.0:  # leave a bit more margin for polish
            break

        # Random initial labeling
        labels = rng.integers(0, 2, size=n)
        labels, cut = _simulated_annealing(labels, adj, deadline)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Final: one more greedy polish on the best found
    if time.time() < deadline - 1.0:
        best_labels, best_cut = _greedy_polish(best_labels, adj, deadline)

    return best_labels
# EVOLVE-BLOCK-END
