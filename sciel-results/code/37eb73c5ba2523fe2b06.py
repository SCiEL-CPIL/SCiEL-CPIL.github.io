# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)

    adj = np.array(adjacency, dtype=np.int8)
    degrees = adj.sum(axis=1)  # degree of each vertex
    deadline = time.time() + 140  # safety margin before 150s kill

    def cut_size(labels):
        return 0.5 * np.sum(adj @ (1 - labels))

    def greedy_flip(labels):
        labels = labels.copy()
        improved = True
        while improved:
            improved = False
            edges_to_other = adj @ (1 - labels)
            deltas = 2 * edges_to_other - degrees
            flip_mask = deltas > 0
            if np.any(flip_mask):
                labels[flip_mask] = 1 - labels[flip_mask]
                improved = True
        return labels

    def simulated_annealing(labels, T_start=10.0, T_end=0.01, cooling=0.95, max_iters=50000):
        labels = labels.copy()
        T = T_start
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Compute deltas for all vertices
            edges_to_other = adj @ (1 - labels)
            deltas = 2 * edges_to_other - degrees

            v = random.randint(0, n - 1)
            delta = deltas[v]

            if delta >= 0 or (T > 0 and random.random() < np.exp(delta / T)):
                labels[v] = 1 - labels[v]
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()

            T *= cooling
            if T < T_end:
                break

        return best_labels, best_cut

    best_labels = None
    best_cut = -1

    # Greedy from alternating start
    labels_alt = np.arange(n) % 2
    labels_greedy = greedy_flip(labels_alt)
    c = cut_size(labels_greedy)
    if c > best_cut:
        best_cut = c
        best_labels = labels_greedy.copy()

    if time.time() > deadline:
        return best_labels

    # Multiple random restarts with SA
    for r in range(30):
        if time.time() > deadline - 5:
            break

        labels_init = np.random.randint(0, 2, n).astype(np.int8)
        labels_g = greedy_flip(labels_init)
        cg = cut_size(labels_g)
        if cg > best_cut:
            best_cut = cg
            best_labels = labels_g.copy()

        if time.time() > deadline - 2:
            break
        sa_labels, sa_cut = simulated_annealing(labels_g)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()

    return best_labels
# EVOLVE-BLOCK-END
