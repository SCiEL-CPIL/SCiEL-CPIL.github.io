"""Maximum Cut via multi-restart local search with spectral initialization and efficient gain computation."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses multi-restart local search with spectral initialization and efficient gain computation.
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)

    if n <= 1:
        return np.zeros(n, dtype=np.int8)

    # Precompute degrees
    degrees = adj.sum(axis=1).astype(np.int32)

    def compute_same(labels):
        """Compute for each vertex the number of neighbors in the same group."""
        in_group0 = (labels == 0).astype(np.int32)
        in_group1 = (labels == 1).astype(np.int32)
        same = np.zeros(n, dtype=np.int32)
        same[labels == 0] = (adj @ in_group0)[labels == 0]
        same[labels == 1] = (adj @ in_group1)[labels == 1]
        return same

    def compute_cut(labels):
        """Compute the total cut size."""
        in_group0 = (labels == 0).astype(np.int32)
        in_group1 = (labels == 1).astype(np.int32)
        return int(in_group0 @ adj @ in_group1)

    def local_search(labels, same, deadline):
        """
        Run local search until no improvement or deadline reached.
        Uses a priority queue of positive-gain flips for efficiency.
        """
        import heapq

        cut = compute_cut(labels)

        # Build initial priority queue of all vertices with positive gain
        # gain[v] = 2*same[v] - degrees[v]
        heap = []
        for v in range(n):
            gain = 2 * same[v] - degrees[v]
            if gain > 0:
                heapq.heappush(heap, (-gain, v))

        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Process the best available flip
            while heap:
                neg_gain, v = heapq.heappop(heap)
                gain = -neg_gain

                # Verify gain is still valid (may have changed due to other flips)
                current_gain = 2 * same[v] - degrees[v]
                if current_gain != gain or current_gain <= 0:
                    continue

                # Flip vertex v
                labels[v] = 1 - labels[v]
                old_group = 1 - labels[v]
                new_group = labels[v]

                # Update same[v]: neighbors in new_group
                if new_group == 0:
                    same[v] = int(adj[v, labels == 0].sum())
                else:
                    same[v] = int(adj[v, labels == 1].sum())

                # Update neighbors of v
                neighbors = np.where(adj[v] == 1)[0]
                for u in neighbors:
                    if u == v:
                        continue
                    if labels[u] == old_group:
                        same[u] -= 1
                    elif labels[u] == new_group:
                        same[u] += 1

                    # Check if u now has positive gain
                    u_gain = 2 * same[u] - degrees[u]
                    if u_gain > 0:
                        heapq.heappush(heap, (-u_gain, u))

                cut += current_gain
                improved = True
                break

        return labels, cut

    def simulated_annealing(labels, same, deadline):
        """
        Run simulated annealing to escape local optima.
        """
        cut = compute_cut(labels)
        best_labels = labels.copy()
        best_cut = cut

        # Temperature schedule
        t0 = 50.0
        cooling = 0.999
        temp = t0

        while time.time() < deadline and temp > 0.01:
            temp *= cooling

            # Pick a random vertex to flip
            v = np.random.randint(0, n)
            old_label = labels[v]
            labels[v] = 1 - old_label

            # Compute new cut
            new_cut = compute_cut(labels)
            delta = new_cut - cut

            if delta > 0:
                # Accept
                cut = new_cut
                if cut > best_cut:
                    best_cut = cut
                    best_labels = labels.copy()
            else:
                # Accept with probability exp(delta/temp)
                if np.random.rand() < np.exp(delta / max(temp, 1e-10)):
                    cut = new_cut
                else:
                    # Reject: revert
                    labels[v] = old_label

        return best_labels, best_cut

    def spectral_init():
        """
        Generate initial labels using the Fiedler vector of the graph Laplacian.
        """
        try:
            # Compute graph Laplacian L = D - A
            L = np.diag(degrees.astype(np.float64)) - adj.astype(np.float64)
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
            # Sort eigenvalues to find the second smallest
            idx = np.argsort(eigenvalues)
            fiedler = eigenvectors[:, idx[1]]
            # Use the sign of the Fiedler vector to create initial labels
            # Center the Fiedler vector first
            fiedler_centered = fiedler - np.mean(fiedler)
            labels = (fiedler_centered > 0).astype(np.int8)
            return labels
        except Exception:
            # Fallback to random initialization if spectral method fails
            return np.random.randint(0, 2, size=n, dtype=np.int8)

    # Deadline: leave 20 seconds margin for safety
    start_time = time.time()
    deadline = start_time + 130.0

    best_labels = None
    best_cut = -1

    # Use a fixed seed for reproducibility
    rng = np.random.default_rng(42)
    np.random.seed(42)

    def greedy_init():
        """Greedy sequential initialization."""
        labels = np.zeros(n, dtype=np.int8)
        order = np.argsort(-degrees)
        for v in order:
            neighbors = np.where(adj[v] == 1)[0]
            if len(neighbors) == 0:
                continue
            count_0 = np.sum(labels[neighbors] == 0)
            count_1 = np.sum(labels[neighbors] == 1)
            if count_1 > count_0:
                labels[v] = 1
            else:
                labels[v] = 0
        return labels

    def random_init():
        return rng.integers(0, 2, size=n, dtype=np.int8)

    # Generate initializations
    initializations = []

    # Spectral initialization (highest priority)
    initializations.append(spectral_init())

    # Alternating
    initializations.append(np.arange(n) % 2)

    # Greedy
    initializations.append(greedy_init())

    # Random restarts
    for _ in range(10):
        initializations.append(random_init())

    # All zeros with one flip
    labels_zero = np.zeros(n, dtype=np.int8)
    labels_zero[0] = 1
    initializations.append(labels_zero.copy())

    # All ones with one flip
    labels_one = np.ones(n, dtype=np.int8)
    labels_one[0] = 0
    initializations.append(labels_one.copy())

    # Process each initialization
    for init_labels in initializations:
        if time.time() > deadline:
            break

        labels = init_labels.copy()
        same = compute_same(labels)

        # Run local search
        labels, cut = local_search(labels, same, deadline)

        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

        # If we still have time, run simulated annealing from the local optimum
        if time.time() < deadline - 30:
            sa_labels, sa_cut = simulated_annealing(labels, compute_same(labels), deadline)
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()

    if best_labels is None:
        best_labels = np.zeros(n, dtype=np.int8)

    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
