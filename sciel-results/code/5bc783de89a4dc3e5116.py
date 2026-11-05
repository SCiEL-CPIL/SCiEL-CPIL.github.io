"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def partition(adjacency):
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    start_time = time.time()
    DEADLINE = start_time + 140.0

    def time_left():
        return DEADLINE - time.time()

    E = int(np.sum(adj)) // 2
    degrees = adj.sum(axis=1).astype(np.int32)

    def cut_size(labels):
        x = 2 * labels.astype(np.float64) - 1
        xTadjx = x @ (adj @ x)
        return int(E - 0.5 * xTadjx)

    def compute_deltas(labels):
        """Compute flip delta for each vertex.
        delta[v] = gain from flipping v = (# neighbors on opposite side) - (# neighbors on same side)
        """
        same_side = (labels == labels[None, :]).astype(np.int8)
        diff_side = 1 - same_side
        # For each v, count neighbors on same side (excluding self) and opposite side
        # same_count[v] = sum of adj[v, w] where labels[w] == labels[v]
        # diff_count[v] = sum of adj[v, w] where labels[w] != labels[v]
        same_count = (adj * same_side).sum(axis=1)
        # Remove self-contribution (adj[v,v] is 0 for simple graphs, but be safe)
        same_count -= np.diag(adj * same_side)
        diff_count = degrees - same_count
        return diff_count - same_count

    def local_search(labels, max_rounds=50):
        """Full-pass hill climbing until no improvement."""
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut

        for rnd in range(max_rounds):
            if time_left() < 2.0:
                break
            improved = False
            deltas = compute_deltas(labels)
            # Find all positive deltas
            pos = np.where(deltas > 0)[0]
            if len(pos) == 0:
                break
            # Apply all positive flips simultaneously (they may interact, but this is a heuristic)
            for v in pos:
                labels[v] = 1 - labels[v]
                current_cut += deltas[v]
                improved = True
            # Recompute after batch
            if improved:
                actual_cut = cut_size(labels)
                if actual_cut > best_cut:
                    best_cut = actual_cut
                    best_labels = labels.copy()
                current_cut = actual_cut

        return best_labels, best_cut

    def local_search_sequential(labels, max_rounds=50):
        """Sequential hill climbing: flip one vertex at a time if it improves."""
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut

        for rnd in range(max_rounds):
            if time_left() < 2.0:
                break
            improved = False
            for v in range(n):
                if time_left() < 1.0:
                    break
                # Compute delta for v
                same = (labels == labels[v])
                same_count = adj[v, same].sum()
                diff_count = degrees[v] - same_count
                delta = diff_count - same_count
                if delta > 0:
                    labels[v] = 1 - labels[v]
                    current_cut += delta
                    improved = True

            if not improved:
                break
            # Update best
            actual_cut = cut_size(labels)
            if actual_cut > best_cut:
                best_cut = actual_cut
                best_labels = labels.copy()
            current_cut = actual_cut

        return best_labels, best_cut

    def spectral_partition():
        """Spectral partition using Fiedler vector."""
        if n < 3:
            return np.zeros(n, dtype=int)
        L = np.diag(degrees.astype(np.float64)) - adj.astype(np.float64)
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            fiedler = eigvecs[:, 1]
            labels = (fiedler > 0).astype(int)
            return labels
        except np.linalg.LinAlgError:
            return np.zeros(n, dtype=int)

    def simulated_annealing(labels, max_iters=200000, T_start=5.0, T_end=0.01):
        """Simulated annealing with single-flip moves."""
        labels = labels.copy()
        best_labels = labels.copy()
        best_cut = cut_size(labels)
        current_cut = best_cut
        T = T_start
        num_iters = max_iters

        # Precompute cooling factor
        if num_iters > 1:
            alpha = (T_end / T_start) ** (1.0 / num_iters)
        else:
            alpha = T_end / T_start

        for it in range(num_iters):
            if time_left() < 3.0:
                break

            v = np.random.randint(0, n)
            same = (labels == labels[v])
            same_count = adj[v, same].sum()
            diff_count = degrees[v] - same_count
            delta = diff_count - same_count

            if delta >= 0:
                labels[v] = 1 - labels[v]
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            else:
                if T > 0.005:
                    # Accept with probability exp(delta/T)
                    if delta > -50 * T:  # Avoid underflow
                        if np.random.random() < np.exp(delta / T):
                            labels[v] = 1 - labels[v]
                            current_cut += delta

            T = max(T * alpha, T_end)

        return best_labels, best_cut

    def multi_start(num_starts, use_sa=False):
        """Multiple random restarts with local search."""
        best_labels = None
        best_cut = -1
        for s in range(num_starts):
            if time_left() < 3.0:
                break
            labels = np.random.randint(0, 2, size=n)
            labels, cut = local_search_sequential(labels, max_rounds=30)
            if use_sa and time_left() > 10.0:
                labels, cut = simulated_annealing(labels, max_iters=30000, T_start=3.0, T_end=0.01)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
        return best_labels, best_cut

    # Strategy:
    # 1. Spectral partition + local search
    # 2. Multiple random restarts with local search
    # 3. Simulated annealing from best solution
    # 4. Final polish

    # Step 1: Spectral
    labels_spec = spectral_partition()
    best_labels, best_cut = local_search_sequential(labels_spec, max_rounds=50)

    # Step 2: Random restarts
    if time_left() > 20.0:
        labels_rs, cut_rs = multi_start(15, use_sa=False)
        if cut_rs > best_cut:
            best_cut = cut_rs
            best_labels = labels_rs

    # Step 3: Simulated annealing from best
    if time_left() > 25.0:
        labels_sa, cut_sa = simulated_annealing(best_labels, max_iters=150000, T_start=5.0, T_end=0.01)
        if cut_sa > best_cut:
            best_cut = cut_sa
            best_labels = labels_sa

    # Step 4: More random restarts with SA if time permits
    if time_left() > 30.0:
        labels_rs2, cut_rs2 = multi_start(10, use_sa=True)
        if cut_rs2 > best_cut:
            best_cut = cut_rs2
            best_labels = labels_rs2

    # Step 5: Final polish
    if time_left() > 5.0:
        for _ in range(5):
            if time_left() < 2.0:
                break
            labels_polish, cut_polish = local_search_sequential(best_labels, max_rounds=30)
            if cut_polish > best_cut:
                best_cut = cut_polish
                best_labels = labels_polish
            else:
                break

    return best_labels.astype(int)
# EVOLVE-BLOCK-END
