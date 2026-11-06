"""Maximum Cut via multi-restart local search with efficient gain computation."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses multi-restart local search with efficient gain computation.
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)

    if n <= 1:
        return np.zeros(n, dtype=np.int8)

    # Precompute degrees
    degrees = adj.sum(axis=1).astype(np.int32)
    # Precompute the full adjacency as int8 for fast operations
    # For a vertex v in group 0, cut contribution = sum of adj[v] where neighbor is in group 1
    # gain of flipping v: if v is in group 0, flipping to 1 changes cut by
    #   (neighbors in group 1) - (neighbors in group 0)
    #   = (deg[v] - same_count) - same_count = deg[v] - 2*same_count
    # where same_count = number of neighbors in same group as v

    # We'll track for each vertex the number of neighbors in the same group
    # same[v] = sum of adj[v] where labels[neighbor] == labels[v]

    def compute_same(labels):
        """Compute for each vertex the number of neighbors in the same group."""
        # same[v] = sum over u where adj[v,u]=1 and labels[u]==labels[v]
        # = sum over u where adj[v,u]=1 and labels[u]==labels[v]
        # For v in group 0: same[v] = sum of adj[v] * (labels == 0)
        # For v in group 1: same[v] = sum of adj[v] * (labels == 1)
        # We can compute: same = adj @ (labels == labels[:,None]) ... but that's O(n^2)
        # Better: for group 0 vertices, same = (adj @ (labels==0))
        # for group 1 vertices, same = (adj @ (labels==1))
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
        # Cut edges = edges between group 0 and group 1
        # = sum over v in group 0 of (adj[v] @ in_group1)
        # = in_group0 @ adj @ in_group1
        return int(in_group0 @ adj @ in_group1)

    def local_search(labels, same, deadline):
        """Run local search until no improvement or deadline reached."""
        cut = compute_cut(labels)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Try flipping each vertex
            for v in range(n):
                if time.time() > deadline:
                    break
                # Gain of flipping v:
                # If v is in group g, neighbors in same group = same[v]
                # After flip, v is in group 1-g
                # New same count for v would be: degrees[v] - same[v]
                # Change in cut: the edges from v to same-group neighbors become cut,
                # and edges from v to other-group neighbors become non-cut.
                # Current contribution of v to cut: degrees[v] - same[v]
                # New contribution: same[v]
                # Gain = same[v] - (degrees[v] - same[v]) = 2*same[v] - degrees[v]
                gain = 2 * same[v] - degrees[v]
                if gain > 0:
                    # Flip vertex v
                    labels[v] = 1 - labels[v]
                    # Update same for v and its neighbors
                    old_group = 1 - labels[v]  # the group v was in
                    new_group = labels[v]
                    # v's new same count: neighbors in new_group
                    if new_group == 0:
                        same[v] = (adj[v] @ in_group0_temp) if False else int(adj[v, labels == 0].sum())
                    else:
                        same[v] = int(adj[v, labels == 1].sum())
                    # Update neighbors of v
                    neighbors = np.where(adj[v] == 1)[0]
                    for u in neighbors:
                        if u == v:
                            continue
                        # u's same count changes: 
                        # If u was in same group as v before (old_group), then v was contributing to same[u]
                        # Now v is in new_group, so:
                        if labels[u] == old_group:
                            same[u] -= 1  # v left u's group
                        elif labels[u] == new_group:
                            same[u] += 1  # v entered u's group
                    cut += gain
                    improved = True
        return labels, cut

    # Deadline: leave 10 seconds margin
    start_time = time.time()
    deadline = start_time + 110.0

    best_labels = None
    best_cut = -1

    # Multiple restarts with different initializations
    rng = np.random.default_rng(42)

    # Initializations:
    # 1. Alternating
    # 2. Random
    # 3. All zeros then flip one (to break symmetry)
    # 4. Greedy sequential

    def greedy_init():
        """Greedy sequential initialization."""
        labels = np.zeros(n, dtype=np.int8)
        # Process vertices in order of degree (descending)
        order = np.argsort(-degrees)
        for v in order:
            # Count neighbors in each group
            neighbors = np.where(adj[v] == 1)[0]
            if len(neighbors) == 0:
                continue
            count_0 = np.sum(labels[neighbors] == 0)
            count_1 = np.sum(labels[neighbors] == 1)
            if count_1 > count_0:
                labels[v] = 1
            elif count_0 > count_1:
                labels[v] = 0
            else:
                labels[v] = 0
        return labels

    def random_init():
        return rng.integers(0, 2, size=n, dtype=np.int8)

    # Generate a set of initializations
    initializations = []
    
    # Alternating
    initializations.append(np.arange(n) % 2)
    
    # Greedy
    initializations.append(greedy_init())
    
    # Random restarts
    for _ in range(20):
        initializations.append(random_init())
    
    # All zeros with one flip
    labels_zero = np.zeros(n, dtype=np.int8)
    labels_zero[0] = 1
    initializations.append(labels_zero.copy())
    
    # All ones with one flip
    labels_one = np.ones(n, dtype=np.int8)
    labels_one[0] = 0
    initializations.append(labels_one.copy())

    for init_labels in initializations:
        if time.time() > deadline:
            break
        labels = init_labels.copy()
        same = compute_same(labels)
        labels, cut = local_search(labels, same, deadline)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    if best_labels is None:
        best_labels = np.zeros(n, dtype=np.int8)

    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
