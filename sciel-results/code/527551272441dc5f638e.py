"""Combined program for Maximum Cut: correct boolean masking, multi-restart, 2-swap polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses correct boolean-mask gain computation, multiple restarts,
    and a 2-swap neighborhood for escaping local optima.
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)

    if n <= 1:
        return np.zeros(n, dtype=np.int8)

    # Precompute degrees
    degrees = adj.sum(axis=1)

    deadline = time.time() + 115.0

    best_labels = None
    best_cut = -1

    def compute_cut(labels):
        """Number of edges crossing the partition using boolean masks."""
        mask0 = labels == 0
        mask1 = labels == 1
        cut = adj[mask0][:, mask1].sum()
        return int(cut)

    def local_search(labels, max_passes=20):
        """
        Greedy 1-flip local search with correct boolean-mask gain.
        gain = degree[v] - 2 * other, where other = neighbors in opposite group.
        """
        labels = labels.copy()
        improved = True
        passes = 0

        while improved and passes < max_passes:
            improved = False
            passes += 1

            for v in range(n):
                if time.time() > deadline - 2.0:
                    return labels

                # Correct: use boolean mask for opposite group
                if labels[v] == 0:
                    other = adj[v, labels == 1].sum()
                else:
                    other = adj[v, labels == 0].sum()

                gain = degrees[v] - 2 * other

                if gain > 0:
                    labels[v] = 1 - labels[v]
                    improved = True

        return labels

    def two_swap_polish(labels, max_rounds=50):
        """
        Try pairwise swaps (flip two vertices) to escape 1-flip local optima.
        A swap of vertices a and b changes the cut by:
          delta = sum of edge changes for a and b considering each other.
        """
        labels = labels.copy()
        improved = True
        rounds = 0

        while improved and rounds < max_rounds:
            improved = False
            rounds += 1

            for a in range(n):
                if time.time() > deadline - 2.0:
                    return labels
                for b in range(a + 1, n):
                    if labels[a] == labels[b]:
                        continue  # Swapping same-label vertices is a no-op

                    # Compute gain of swapping a and b
                    # After swap: a gets label of b, b gets label of a
                    # For vertex a: neighbors in new opposite group
                    new_label_a = 1 - labels[a]
                    new_label_b = 1 - labels[b]

                    # Count neighbors of a in group new_label_a (excluding b's contribution change)
                    # Actually, let's just compute the cut delta directly
                    # Old contribution: edges from a to opposite(a) + edges from b to opposite(b)
                    # New contribution: edges from a to opposite(new_label_a) + edges from b to opposite(new_label_b)
                    # But a and b are connected, so their mutual edge changes too.

                    # Simpler: compute old and new cut contributions
                    # For a: old other = neighbors not in group a
                    # For b: old other = neighbors not in group b

                    old_a = adj[a, labels != labels[a]].sum()
                    old_b = adj[b, labels != labels[b]].sum()

                    # Temporarily flip both
                    labels[a] = new_label_a
                    labels[b] = new_label_b

                    new_a = adj[a, labels != labels[a]].sum()
                    new_b = adj[b, labels != labels[b]].sum()

                    delta = (new_a + new_b) - (old_a + old_b)

                    if delta > 0:
                        improved = True
                    else:
                        # Revert
                        labels[a] = 1 - new_label_a
                        labels[b] = 1 - new_label_b

        return labels

    rng = np.random.default_rng(seed=42)

    # Strategy 1: Alternating start + 1-flip LS + 2-swap polish
    labels_alt = np.arange(n) % 2
    labels_improved = local_search(labels_alt)
    labels_polished = two_swap_polish(labels_improved)
    cut = compute_cut(labels_polished)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels_polished.copy()

    # Strategy 2: All-zeros start
    labels_zero = np.zeros(n, dtype=np.int8)
    labels_improved = local_search(labels_zero)
    labels_polished = two_swap_polish(labels_improved)
    cut = compute_cut(labels_polished)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels_polished.copy()

    # Strategy 3: Random restarts with 1-flip LS + 2-swap polish
    for restart in range(200):
        if time.time() > deadline - 5.0:
            break

        labels_rand = rng.integers(0, 2, size=n).astype(np.int8)
        labels_improved = local_search(labels_rand)
        labels_polished = two_swap_polish(labels_improved)
        cut = compute_cut(labels_polished)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels_polished.copy()

        if time.time() > deadline - 3.0 and restart > 20:
            break

    if best_labels is None:
        best_labels = np.zeros(n, dtype=np.int8)

    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
