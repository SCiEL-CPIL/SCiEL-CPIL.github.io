"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    deadline = time.time() + 145.0  # safety margin before 150s kill

    # Spectral initialization using the Fiedler vector of the graph Laplacian
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj

    # Compute eigenvalues and eigenvectors of the Laplacian
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        fiedler = eigvecs[:, 1]
        # Threshold at the mean to get initial 0/1 labels
        labels = (fiedler >= np.median(fiedler)).astype(np.int8)
    except Exception:
        # Fallback to alternating initialization
        labels = np.arange(n) % 2

    # Helper: compute cut size for current labels
    def cut_size(labels):
        # Cut edges are those connecting vertices with different labels
        # For each edge (i,j), it's in the cut if labels[i] != labels[j]
        # We can compute this as: sum over all edges where labels differ
        # Efficient: for each vertex, count neighbors with different label
        total = 0
        for v in range(n):
            if labels[v] == 0:
                total += adj[v, labels == 1].sum()
            else:
                total += adj[v, labels == 0].sum()
        return total // 2  # Each edge counted twice

    # Helper: compute gain of flipping vertex v
    def flip_gain(v, labels):
        # If we flip v, edges from v to same-side become cut, and cut edges become non-cut
        current_side = labels[v]
        same_side = (labels == current_side) & (labels != 0)  # placeholder
        # Actually: neighbors with same label as v will now be on opposite side (gain +1 each)
        # Neighbors with different label as v will now be on same side (lose 1 each)
        same_count = adj[v, labels == current_side].sum()
        diff_count = adj[v, labels != current_side].sum()
        # Flipping: gain = same_count - diff_count (net change in cut size)
        return same_count - diff_count

    # Local search: 1-flip improvement
    def local_search(labels, max_rounds=20):
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            for v in range(n):
                if time.time() > deadline:
                    return labels
                gain = flip_gain(v, labels)
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    improved = True
        return labels

    # Simulated annealing for escaping local optima
    def simulated_annealing(labels, start_temp=10.0, end_temp=0.1, cooling=0.99, max_steps=5000):
        temp = start_temp
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        steps = 0
        while temp > end_temp and steps < max_steps:
            if time.time() > deadline:
                break
            # Pick a random vertex to flip
            v = np.random.randint(n)
            labels[v] = 1 - labels[v]
            new_cut = cut_size(labels)
            delta = new_cut - current_cut
            if delta > 0 or np.random.random() < np.exp(delta / temp):
                current_cut = new_cut
                if new_cut > best_cut:
                    best_cut = new_cut
                    best_labels = labels.copy()
            else:
                labels[v] = 1 - labels[v]  # revert
            temp *= cooling
            steps += 1
        return best_labels

    # First pass: local search from spectral init
    labels = local_search(labels.copy())
    best_labels = labels.copy()
    best_cut = cut_size(best_labels)

    # Multiple restarts with perturbation if time allows
    num_restarts = 0
    while time.time() < deadline - 10.0:
        num_restarts += 1
        # Perturb the best solution: flip k random vertices
        k = max(1, n // 20)
        perturbed = best_labels.copy()
        flip_indices = np.random.choice(n, size=k, replace=False)
        perturbed[flip_indices] = 1 - perturbed[flip_indices]

        # Local search from perturbed state
        candidate = local_search(perturbed, max_rounds=10)
        candidate_cut = cut_size(candidate)

        if candidate_cut > best_cut:
            best_cut = candidate_cut
            best_labels = candidate.copy()

        # Try simulated annealing from time to time
        if num_restarts % 3 == 0 and time.time() < deadline - 20.0:
            sa_candidate = simulated_annealing(best_labels.copy(), start_temp=5.0, end_temp=0.1, cooling=0.995, max_steps=3000)
            sa_cut = cut_size(sa_candidate)
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_candidate.copy()

    return best_labels
# EVOLVE-BLOCK-END
