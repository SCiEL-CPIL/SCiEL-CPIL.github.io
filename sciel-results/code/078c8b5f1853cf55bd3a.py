import numpy as np
import time
import random

def partition(adjacency):
    """
    Assigns 0/1 labels to vertices to maximize the number of edges crossing the partition.
    Uses a combination of:
      1. Spectral initialization (adjacency eigenvector)
      2. Fast local search (greedy hill-climbing with cross-array updates)
      3. Pairwise flip local search (escapes 2-vertex local optima)
      4. Simulated annealing / quantum-inspired annealing
      5. Multi-start with random restarts
    Deadline guard ensures return well before 150s.
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    # Ensure symmetric and no self loops for safety
    adj = (adj + adj.T) // 2
    np.fill_diagonal(adj, 0)

    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin before 150s kill

    def time_left():
        return DEADLINE - time.time()

    # Precompute
    degrees = adj.sum(axis=1).astype(np.int32)
    E = int(degrees.sum()) // 2

    # Build neighbor list for fast iteration
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        idxs = np.nonzero(adj[i])[0]
        neighbors[i] = idxs.tolist()

    def init_cross(labels):
        """Vectorized initialization of cross array."""
        diff_matrix = (labels[None, :] != labels[:, None])
        return np.sum(adj * diff_matrix, axis=1).astype(np.int32)

    def cut_size(labels):
        """Fast cut size: count edges crossing the cut."""
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2

    def run_local_search(labels, max_rounds=500):
        """Greedy hill-climbing: flip any vertex that improves cut.
        Uses cross-array for O(degree) updates per flip."""
        labels = labels.copy().astype(np.int8)
        cross = init_cross(labels)

        best_cut = int(0.5 * np.sum(cross))
        best_labels = labels.copy()

        for rnd in range(max_rounds):
            if time_left() < 2.0:
                break

            # gains = degrees - 2 * cross
            gains = degrees - 2 * cross

            # Find vertex with max gain
            best_v = np.argmax(gains)
            max_gain = gains[best_v]

            if max_gain <= 0:
                break

            # Flip best_v
            v = best_v
            labels[v] = 1 - labels[v]

            # Update cross for v and neighbors
            old_cross_v = cross[v]
            cross[v] = degrees[v] - old_cross_v

            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    cross[u] -= 1
                else:
                    cross[u] += 1

            current_cut = int(0.5 * np.sum(cross))
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()

        return best_labels, best_cut

    def run_pairwise_local_search(labels, max_rounds=200):
        """Pairwise flip local search: try flipping pairs of vertices.
        Escapes 2-vertex local optima that single-flip cannot."""
        labels = labels.copy().astype(np.int8)
        cross = init_cross(labels)

        best_cut = int(0.5 * np.sum(cross))
        best_labels = labels.copy()

        for rnd in range(max_rounds):
            if time_left() < 3.0:
                break

            improved = False
            # Try random pairs of vertices
            num_pairs = min(200, n * (n - 1) // 2)
            for _ in range(num_pairs):
                if time_left() < 2.0:
                    break

                v1, v2 = np.random.choice(n, size=2, replace=False)
                if v1 == v2:
                    continue

                # Compute gain of flipping both v1 and v2
                # Gain = (degrees[v1] - 2*cross[v1]) + (degrees[v2] - 2*cross[v2])
                # But if v1 and v2 are adjacent, the edge between them contributes
                # differently: if they have same label, flipping both keeps the edge
                # internal; if different, flipping both keeps it crossing.
                # Actually, let's compute the exact gain:
                # Before: cross[v1] counts neighbors of v1 with different label
                # After flipping v1: cross[v1] becomes degrees[v1] - cross[v1]
                # But we also flip v2, so we need to account for the interaction.

                # Simpler: compute the change in cut size directly.
                # Current cut contribution from edges incident to v1 or v2:
                # Let's just compute the full gain by checking the effect.

                # Gain from flipping v1 alone: degrees[v1] - 2*cross[v1]
                # Gain from flipping v2 alone: degrees[v2] - 2*cross[v2]
                # If v1 and v2 are adjacent:
                #   If labels[v1] == labels[v2]: the edge (v1,v2) is internal.
                #     After flipping both, it's still internal. No change from this edge.
                #   If labels[v1] != labels[v2]: the edge (v1,v2) is crossing.
                #     After flipping both, it's still crossing. No change from this edge.
                # So the pairwise gain is just the sum of individual gains.

                gain1 = degrees[v1] - 2 * cross[v1]
                gain2 = degrees[v2] - 2 * cross[v2]
                total_gain = gain1 + gain2

                if total_gain > 0:
                    # Flip both
                    labels[v1] = 1 - labels[v1]
                    labels[v2] = 1 - labels[v2]

                    # Update cross for v1, v2, and their neighbors
                    # First update v1's neighbors (excluding v2 if adjacent)
                    old_cross_v1 = cross[v1]
                    old_cross_v2 = cross[v2]

                    # Update cross for v1
                    for u in neighbors[v1]:
                        if u == v2:
                            continue
                        if labels[u] == labels[v1]:
                            cross[u] -= 1
                        else:
                            cross[u] += 1

                    # Update cross for v2
                    for u in neighbors[v2]:
                        if u == v1:
                            continue
                        if labels[u] == labels[v2]:
                            cross[u] -= 1
                        else:
                            cross[u] += 1

                    # Update cross[v1] and cross[v2]
                    # cross[v1] = number of neighbors of v1 with different label
                    cross[v1] = 0
                    for u in neighbors[v1]:
                        if labels[u] != labels[v1]:
                            cross[v1] += 1

                    cross[v2] = 0
                    for u in neighbors[v2]:
                        if labels[u] != labels[v2]:
                            cross[v2] += 1

                    current_cut = int(0.5 * np.sum(cross))
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_labels = labels.copy()
                    improved = True

            if not improved:
                break

        return best_labels, best_cut

    def run_simulated_annealing(labels, max_iters=50000, T_start=10.0, T_end=0.1, alpha=0.99995):
        """Simulated annealing to escape local optima."""
        labels = labels.copy().astype(np.int8)
        cross = init_cross(labels)

        current_cut = int(0.5 * np.sum(cross))
        best_cut = current_cut
        best_labels = labels.copy()

        T = T_start
        for it in range(max_iters):
            if time_left() < 3.0:
                break

            v = random.randint(0, n - 1)

            # Gain = degrees[v] - 2 * cross[v]
            gain = degrees[v] - 2 * cross[v]

            # delta cut = gain / 2
            delta = gain / 2.0

            if delta > 0 or (T > 0.01 and random.random() < np.exp(delta / T)):
                # Flip v
                labels[v] = 1 - labels[v]

                # Update cross
                old_cross_v = cross[v]
                cross[v] = degrees[v] - old_cross_v

                for u in neighbors[v]:
                    if labels[u] == labels[v]:
                        cross[u] -= 1
                    else:
                        cross[u] += 1

                current_cut += int(delta)

                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()

            T = max(T * alpha, T_end)

        return best_labels, best_cut

    def run_quantum_annealing(labels, max_iters=80000, s_start=1.0, s_end=0.0, cooling=0.99998):
        """Quantum-inspired annealing: interpolate between transverse field and problem Hamiltonian.
        Uses vectorized gain computation for speed."""
        labels = labels.copy().astype(np.int8)
        cross = init_cross(labels)

        current_cut = int(0.5 * np.sum(cross))
        best_cut = current_cut
        best_labels = labels.copy()

        s = s_start  # transverse field strength (1.0 = pure quantum, 0.0 = pure classical)

        for it in range(max_iters):
            if time_left() < 3.0:
                break

            # Compute gains for ALL vertices at once (vectorized)
            # gain[v] = degrees[v] - 2 * cross[v]
            gains = degrees - 2 * cross

            # Quantum tunneling: with probability proportional to s,
            # we accept worse moves more readily
            T_eff = s * 5.0 + 0.1

            v = random.randint(0, n - 1)
            delta = gains[v] / 2.0

            if delta > 0 or (T_eff > 0.01 and random.random() < np.exp(delta / T_eff)):
                # Flip v
                labels[v] = 1 - labels[v]

                # Update cross
                old_cross_v = cross[v]
                cross[v] = degrees[v] - old_cross_v

                for u in neighbors[v]:
                    if labels[u] == labels[v]:
                        cross[u] -= 1
                    else:
                        cross[u] += 1

                current_cut += int(delta)

                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()

            s = max(s * cooling, s_end)

        return best_labels, best_cut

    # Strategy
    best_labels = None
    best_cut = -1

    # 1. Spectral Initialization
    try:
        vals, vecs = np.linalg.eigh(adj.astype(np.float64))
        eigvec = vecs[:, -1]
        labels_spec = (eigvec > 0).astype(np.int8)
    except:
        labels_spec = np.arange(n) % 2

    l, c = run_local_search(labels_spec)
    if c > best_cut:
        best_cut = c
        best_labels = l

    # 2. Pairwise local search from spectral init
    l, c = run_pairwise_local_search(labels_spec, max_rounds=100)
    if c > best_cut:
        best_cut = c
        best_labels = l

    # 3. Alternating start
    labels_alt = np.arange(n) % 2
    l, c = run_local_search(labels_alt)
    if c > best_cut:
        best_cut = c
        best_labels = l

    l, c = run_pairwise_local_search(labels_alt, max_rounds=100)
    if c > best_cut:
        best_cut = c
        best_labels = l

    # 4. Random starts with Local Search
    num_restarts = 150
    for _ in range(num_restarts):
        if time_left() < 5.0:
            break
        labels_rand = np.random.randint(0, 2, size=n, dtype=np.int8)
        l, c = run_local_search(labels_rand, max_rounds=100)
        if c > best_cut:
            best_cut = c
            best_labels = l

    # 5. Simulated Annealing from best
    if time_left() > 15.0:
        l, c = run_simulated_annealing(best_labels)
        if c > best_cut:
            best_cut = c
            best_labels = l

    # 6. Quantum Annealing from best
    if time_left() > 15.0:
        l, c = run_quantum_annealing(best_labels)
        if c > best_cut:
            best_cut = c
            best_labels = l

    # 7. More SA restarts
    if time_left() > 10.0:
        for _ in range(15):
            if time_left() < 3.0:
                break
            labels_rand = np.random.randint(0, 2, size=n, dtype=np.int8)
            l, c = run_simulated_annealing(labels_rand, max_iters=10000, T_start=5.0, T_end=0.1)
            if c > best_cut:
                best_cut = c
                best_labels = l

    # 8. Pairwise local search from best
    if time_left() > 5.0:
        l, c = run_pairwise_local_search(best_labels, max_rounds=200)
        if c > best_cut:
            best_cut = c
            best_labels = l

    # 9. Final polish
    if time_left() > 3.0:
        l, c = run_local_search(best_labels, max_rounds=500)
        if c > best_cut:
            best_cut = c
            best_labels = l

    return best_labels.astype(int)
