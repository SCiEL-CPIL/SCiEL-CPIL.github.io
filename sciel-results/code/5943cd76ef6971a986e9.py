"""Maximum Cut via multi-start spectral + aggressive local search with tabu and 2-swap."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s limit

    if n == 0:
        return np.array([], dtype=np.int8)

    adj = np.array(adjacency, dtype=np.int32)
    total_edges = int(np.sum(adj)) // 2

    # Precompute neighbor lists and degrees
    neighbors = [np.where(adj[i] > 0)[0] for i in range(n)]
    degrees = np.array([len(neighbors[i]) for i in range(n)], dtype=np.int32)

    def compute_cut(labels):
        """Compute the cut size for a given labeling."""
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2

    def compute_same_count(labels):
        """Compute for each vertex the number of same-side neighbors."""
        same_count = np.zeros(n, dtype=np.int32)
        for v in range(n):
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[v] += 1
        return same_count

    def flip_greedy(labels, max_passes=100):
        """Fast 1-flip greedy using incremental cut tracking."""
        labels = labels.copy()
        same_count = compute_same_count(labels)

        for _ in range(max_passes):
            if time.time() > deadline:
                break
            changed = False
            # Randomize order for better exploration
            order = list(range(n))
            random.shuffle(order)
            for v in order:
                if time.time() > deadline:
                    break
                deg = degrees[v]
                gain = 2 * same_count[v] - deg
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    changed = True
                    old_same = same_count[v]
                    same_count[v] = deg - old_same
                    for u in neighbors[v]:
                        if labels[u] == labels[v]:
                            same_count[u] += 1
                        else:
                            same_count[u] -= 1
            if not changed:
                break
        return labels

    def tabu_search(labels, max_iters=5000, tabu_tenure=20):
        """Tabu search with 1-flip moves to escape local optima."""
        labels = labels.copy()
        same_count = compute_same_count(labels)
        current_cut = compute_cut(labels)
        best_cut = current_cut
        best_labels = labels.copy()

        # Tabu list: store (vertex, time) for last flip
        tabu = {}
        best_gain = -np.inf
        best_vertex = -1

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Find best non-tabu flip
            best_gain = -np.inf
            best_vertex = -1
            best_aspiration = False

            # Sample a subset of vertices for speed
            if n <= 200:
                candidates = list(range(n))
            else:
                candidates = random.sample(range(n), min(n, 50))

            for v in candidates:
                if v in tabu and tabu[v] > it - tabu_tenure:
                    # Check aspiration: if this move improves best, allow it
                    gain = 2 * same_count[v] - degrees[v]
                    if gain > 0 and (current_cut + gain) > best_cut:
                        best_aspiration = True
                        if gain > best_gain:
                            best_gain = gain
                            best_vertex = v
                else:
                    gain = 2 * same_count[v] - degrees[v]
                    if gain > best_gain:
                        best_gain = gain
                        best_vertex = v

            if best_vertex == -1:
                break

            # Perform the flip
            v = best_vertex
            if v in tabu:
                del tabu[v]
            else:
                tabu[v] = it

            old_same = same_count[v]
            labels[v] = 1 - labels[v]
            same_count[v] = degrees[v] - old_same
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[u] += 1
                else:
                    same_count[u] -= 1

            current_cut += best_gain
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()

        return best_labels

    def two_swap_search(labels, max_passes=15):
        """2-swap local search with incremental updates."""
        labels = labels.copy()
        same_count = compute_same_count(labels)

        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            order = list(range(n))
            random.shuffle(order)

            for i in order:
                if time.time() > deadline:
                    break
                li = labels[i]
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    if labels[i] == labels[j]:
                        continue

                    lj = labels[j]
                    e_ij = adj[i, j]

                    # Gain calculation for swapping i and j
                    new_same_i = degrees[i] - same_count[i] - e_ij
                    new_same_j = degrees[j] - same_count[j] - e_ij
                    delta_same = (new_same_i + new_same_j) - (same_count[i] + same_count[j])

                    if delta_same > 0:
                        # Perform swap
                        labels[i] = lj
                        labels[j] = li
                        improved = True

                        old_same_i = same_count[i]
                        old_same_j = same_count[j]
                        same_count[i] = new_same_i
                        same_count[j] = new_same_j

                        # Update neighbors of i (excluding j)
                        for u in neighbors[i]:
                            if u == j:
                                continue
                            if labels[u] == labels[i]:
                                same_count[u] += 1
                            else:
                                same_count[u] -= 1

                        # Update neighbors of j (excluding i)
                        for u in neighbors[j]:
                            if u == i:
                                continue
                            if labels[u] == labels[j]:
                                same_count[u] += 1
                            else:
                                same_count[u] -= 1

            if not improved:
                break

        return labels

    def spectral_partition(k=32, num_random=5):
        """Spectral partitioning with multiple random hyperplanes."""
        try:
            D = np.diag(degrees.astype(np.float64))
            L = D - adj.astype(np.float64)
            eigenvalues, eigenvectors = np.linalg.eigh(L)

            k = min(k, n - 1)
            embedding = eigenvectors[:, 1:k + 1]

            # Normalize rows
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embedding = embedding / norms

            best_labels = None
            best_cut = -1

            for _ in range(num_random):
                random_vec = np.random.randn(k)
                random_vec /= np.linalg.norm(random_vec)
                projection = embedding @ random_vec
                labels = (projection > 0).astype(np.int8)
                cut = compute_cut(labels)
                if cut > best_cut:
                    best_cut = cut
                    best_labels = labels

            if best_labels is not None:
                return best_labels
        except Exception:
            pass
        return np.random.randint(0, 2, size=n, dtype=np.int8)

    def fiedler_partition():
        """Partition using the Fiedler vector (2nd eigenvector)."""
        try:
            D = np.diag(degrees.astype(np.float64))
            L = D - adj.astype(np.float64)
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            fiedler = eigenvectors[:, 1]
            return (fiedler > 0).astype(np.int8)
        except Exception:
            return np.random.randint(0, 2, size=n, dtype=np.int8)

    def greedy_partition():
        """Greedy partition: assign vertices one by one to maximize cut."""
        labels = np.zeros(n, dtype=np.int8)
        # Process vertices in order of decreasing degree
        order = np.argsort(-degrees)
        for v in order:
            # Count neighbors in each partition
            count_0 = 0
            count_1 = 0
            for u in neighbors[v]:
                if labels[u] == 0:
                    count_0 += 1
                else:
                    count_1 += 1
            # Assign to partition with fewer same-side neighbors
            if count_0 < count_1:
                labels[v] = 1
            elif count_1 < count_0:
                labels[v] = 0
            else:
                labels[v] = random.randint(0, 1)
        return labels

    best_cut = -1
    best_labels = None

    def try_solution(labels):
        nonlocal best_cut, best_labels
        # 1-flip greedy
        labels1 = flip_greedy(labels, max_passes=50)
        cut1 = compute_cut(labels1)
        if cut1 > best_cut:
            best_cut = cut1
            best_labels = labels1.copy()

        # Tabu search
        if time.time() < deadline:
            labels2 = tabu_search(labels1, max_iters=3000, tabu_tenure=15)
            cut2 = compute_cut(labels2)
            if cut2 > best_cut:
                best_cut = cut2
                best_labels = labels2.copy()

        # 2-swap
        if time.time() < deadline:
            labels3 = two_swap_search(labels2, max_passes=10)
            cut3 = compute_cut(labels3)
            if cut3 > best_cut:
                best_cut = cut3
                best_labels = labels3.copy()

        # Another round of 1-flip + tabu on the result
        if time.time() < deadline:
            labels4 = flip_greedy(labels3, max_passes=30)
            labels5 = tabu_search(labels4, max_iters=2000, tabu_tenure=12)
            labels6 = two_swap_search(labels5, max_passes=8)
            cut6 = compute_cut(labels6)
            if cut6 > best_cut:
                best_cut = cut6
                best_labels = labels6.copy()

    # Generate diverse initial solutions
    init_solutions = []

    # 1. Spectral with multiple ranks
    init_solutions.append(spectral_partition(k=32, num_random=5))
    init_solutions.append(spectral_partition(k=16, num_random=5))
    init_solutions.append(spectral_partition(k=8, num_random=5))

    # 2. Fiedler vector
    init_solutions.append(fiedler_partition())

    # 3. Greedy
    init_solutions.append(greedy_partition())

    # 4. Random
    init_solutions.append(np.random.randint(0, 2, size=n, dtype=np.int8))

    # 5. Alternating
    init_solutions.append(np.arange(n) % 2)

    # Process initial solutions
    for init in init_solutions:
        if time.time() > deadline:
            break
        try_solution(init)

    # Additional random restarts
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() > deadline:
            break

        # Mix of random and spectral
        if r % 3 == 0:
            labels = np.random.randint(0, 2, size=n, dtype=np.int8)
        elif r % 3 == 1:
            labels = spectral_partition(k=16, num_random=1)
        else:
            labels = greedy_partition()

        try_solution(labels)

    # Final polish: multiple rounds of local search on best solution
    if best_labels is not None and time.time() < deadline:
        for _ in range(3):
            if time.time() > deadline:
                break
            labels = flip_greedy(best_labels, max_passes=40)
            labels = tabu_search(labels, max_iters=2000, tabu_tenure=10)
            labels = two_swap_search(labels, max_passes=10)
            cut = compute_cut(labels)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
