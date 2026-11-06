"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition (each edge counted once)."""
    n = len(labels)
    cross = (labels[:, None] != labels[None, :])
    return int((adj * cross).sum() // 2)


def _greedy_improve(adj, labels, deadline):
    """
    Greedy local improvement using gain vectors.
    Returns (labels, cut_size).
    """
    n = len(labels)
    adj_bool = np.asarray(adj, dtype=bool)
    labels = np.array(labels, dtype=np.int8)

    # gain[v] = (neighbors of v in the OTHER set) - (neighbors of v in the SAME set)
    # Flipping v changes cut by: gain[v] becomes -gain[v] after flip,
    # and for each neighbor u, gain[u] changes by 2 if u and v were in same set, -2 otherwise.
    # Actually: gain[v] = sum over neighbors u of (2 if labels[u]!=labels[v] else -2) ... 
    # Let me redefine cleanly.
    # Let s(v) = labels[v] in {0,1}.
    # cut contribution of edge (v,u): 1 if s(v)!=s(u) else 0.
    # If we flip v, for each neighbor u:
    #   before: contributes 1 if s(v)!=s(u), else 0
    #   after:  contributes 1 if s(v)==s(u), else 0  (since s(v) flipped)
    #   delta for edge (v,u) = +1 if s(v)==s(u) before, -1 if s(v)!=s(u) before
    # So delta_cut = (# neighbors u with s(u)==s(v)) - (# neighbors u with s(u)!=s(v))
    # Let same[v] = # neighbors in same set, other[v] = # neighbors in other set
    # delta_cut = same[v] - other[v]
    # We flip if delta_cut > 0, i.e., same[v] > other[v].

    # Precompute adjacency as list of neighbors for fast iteration
    neighbors = [np.where(adj_bool[v])[0] for v in range(n)]
    deg = np.array([len(nei) for nei in neighbors], dtype=np.int32)

    # same[v] = number of neighbors of v with same label as v
    # other[v] = deg[v] - same[v]
    # delta = same[v] - other[v] = 2*same[v] - deg[v]
    same = np.zeros(n, dtype=np.int32)
    for v in range(n):
        for u in neighbors[v]:
            if labels[u] == labels[v]:
                same[v] += 1

    best_labels = labels.copy()
    best_cut = _cut_size(adj_bool, labels)

    for _pass in range(200):
        if time.time() > deadline:
            break
        improved = False
        # Process vertices in random order to break symmetry
        order = list(range(n))
        random.shuffle(order)
        for v in order:
            # delta = same[v] - (deg[v] - same[v]) = 2*same[v] - deg[v]
            delta = 2 * same[v] - deg[v]
            if delta > 0:
                # Flip v
                labels[v] = 1 - labels[v]
                improved = True
                # Update same[] for v and its neighbors
                # After flip, v's same becomes deg[v] - same[v]
                old_same_v = same[v]
                same[v] = deg[v] - old_same_v
                for u in neighbors[v]:
                    # Before flip: u and v had some relationship
                    # If labels[u] == old labels[v] (i.e., same before), then after flip they differ
                    # We need to update same[u]: 
                    #   before: if labels[u]==labels[v] (old), same[u] included u's contribution
                    #   after:  if labels[u]!=labels[v] (new), same[u] should not include it
                    # So: if labels[u] == old_label_v (same before), same[u] -= 1
                    #     else (different before), same[u] += 1
                    if labels[u] == (1 - labels[v]):  # labels[u] equals the OLD label of v
                        same[u] -= 1
                    else:
                        same[u] += 1

        if not improved:
            break

    cut = _cut_size(adj_bool, labels)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()

    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    best_labels = None
    best_cut = -1

    # Try alternating start first
    labels_alt = np.arange(n) % 2
    labels_alt, cut_alt = _greedy_improve(adjacency, labels_alt, deadline)
    best_labels = labels_alt.copy()
    best_cut = cut_alt

    # Random restarts
    rng = np.random.default_rng(42)
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        # Random initial labeling
        labels_rand = rng.integers(0, 2, size=n).astype(np.int8)
        labels_out, cut_out = _greedy_improve(adjacency, labels_rand, deadline)
        if cut_out > best_cut:
            best_cut = cut_out
            best_labels = labels_out.copy()

    return best_labels
# EVOLVE-BLOCK-END
