"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

DEADLINE_MARGIN = 5.0


def _cut_size(labels, adj):
    """Compute the cut size (number of crossing edges) for a labeling."""
    # cut = sum over edges (u,v) of |l_u - l_v| = sum(adj) - sum(adj * (l_u == l_v))
    # Equivalent: for each vertex v, count neighbors with opposite label
    # Vectorized: cut = sum_v sum_u adj[v,u] * |l_v - l_u|
    # = sum_v (deg[v] - sum_u adj[v,u] * (2*l_v - 1)*(2*l_u - 1) / 2 ... )
    # Simpler: use the sign formulation
    s = 2 * labels - 1  # ±1
    # cut = (M - s^T A s) / 4 where M = total edge count * 2 (since symmetric)
    # Actually for undirected: cut = sum_{u<v} A[u,v] * (1 - s_u * s_v) / 2
    # = (sum_{u,v} A[u,v] - sum_{u,v} A[u,v]*s_u*s_v) / 4
    # Since A is symmetric with 0 diagonal: sum_{u,v} A[u,v] = 2*E
    # s^T A s = sum_{u,v} s_u A[u,v] s_v
    total = int(adj.sum())
    s_as = int(s @ adj @ s)
    return (total - s_as) // 4


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
    Simulated annealing on the cut problem with batched multi-flip steps.
    labels: initial 0/1 array
    Returns improved labels and cut size.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(labels, adj)
    best_labels = labels.copy()
    best_cut = current_cut

    # Temperature schedule: geometric cooling
    T_start = 8.0
    T_end = 0.005
    n_iter = 3000  # will be capped by deadline

    rng = np.random.default_rng()

    # Precompute degree for gain updates
    deg = adj.sum(axis=1)

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

        # Batch: pick k random vertices and process them
        # For small n, batch of 1 is fine; for efficiency, try batch of 2-4
        batch_size = min(n, max(1, n // 50))
        
        for _ in range(batch_size):
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
    Includes a final high-precision polish with multiple passes.
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

    # High-precision polish: multiple full passes over all vertices
    # Try flipping each vertex in order if it improves, repeat until no improvement
    for polish_pass in range(5):
        if time.time() > deadline:
            break
        any_improved = False
        for v in range(n):
            if time.time() > deadline:
                break
            gains_v = _vectorized_gains(labels, adj)
            if gains_v[v] > 0:
                labels[v] = 1 - labels[v]
                current_cut += int(gains_v[v])
                any_improved = True
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
        if not any_improved:
            break

    return best_labels, best_cut


def _spectral_init(adj, n):
    """
    Use spectral bisection to get a good initial partition.
    Compute the Fiedler vector (2nd smallest eigenvector of L = D - A)
    and split at the median.
    """
    deg = adj.sum(axis=1).astype(float)
    L = np.diag(deg) - adj.astype(float)
    
    # Use eigendecomposition for small n (100 nodes is fine)
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector corresponding to the 2nd smallest eigenvalue
        # eigvals are sorted ascending
        fiedler = eigvecs[:, 1]
        # Split at median
        median_val = np.median(fiedler)
        labels = (fiedler > median_val).astype(int)
        return labels
    except Exception:
        return np.random.default_rng(42).integers(0, 2, size=n)


def partition(adjacency):
    """
    Partition vertices into two sets to maximize the number of crossing edges.
    Uses spectral initialization, simulated annealing with greedy polish and multi-start.
    """
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=bool)

    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    best_labels = None
    best_cut = -1

    rng = np.random.default_rng(42)

    # Phase 1: Spectral initialization + greedy polish (fast baseline)
    labels_spec = _spectral_init(adj, n)
    labels_spec, cut_spec = _greedy_polish(labels_spec, adj, deadline)
    best_labels = labels_spec.copy()
    best_cut = cut_spec

    # Also try alternating start as a second baseline
    labels_alt = np.arange(n) % 2
    labels_alt, cut_alt = _greedy_polish(labels_alt, adj, deadline)
    if cut_alt > best_cut:
        best_cut = cut_alt
        best_labels = labels_alt.copy()

    # Phase 2: Random restarts with simulated annealing
    # Each SA run does many iterations; check deadline before each restart
    n_restarts_max = 50
    for r in range(n_restarts_max):
        if time.time() > deadline - 2.0:  # leave a bit more margin for polish
            break

        # Mix of random and perturbed spectral starts
        if r % 3 == 0:
            # Start from perturbed spectral solution
            labels = labels_spec.copy()
            # Flip a random subset of vertices
            flip_mask = rng.random(n) < 0.3
            labels[flip_mask] = 1 - labels[flip_mask]
        elif r % 3 == 1:
            # Pure random
            labels = rng.integers(0, 2, size=n)
        else:
            # Start from best found so far with small perturbation
            labels = best_labels.copy()
            flip_mask = rng.random(n) < 0.1
            labels[flip_mask] = 1 - labels[flip_mask]
        
        labels, cut = _simulated_annealing(labels, adj, deadline)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Final: one more greedy polish on the best found
    if time.time() < deadline - 1.0:
        best_labels, best_cut = _greedy_polish(best_labels, adj, deadline)

    return best_labels
# EVOLVE-BLOCK-END
