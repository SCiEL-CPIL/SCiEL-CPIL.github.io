"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    adj = np.array(adjacency, dtype=np.int8)
    best_labels = None
    best_cut = -1

    def compute_cut(labels):
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2

    def greedy_1flip(labels, max_passes=10):
        labels = labels.copy()
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            changed = False
            for v in range(n):
                if time.time() > deadline:
                    break
                same = int(adj[v][labels == labels[v]].sum())
                other = int(adj[v][labels != labels[v]].sum())
                if same > other:
                    labels[v] = 1 - labels[v]
                    changed = True
            if not changed:
                break
        return labels

    def fast_2swap(labels, max_passes=100):
        """2-swap local search with incremental O(n) updates per swap."""
        labels = labels.copy()
        sum_same = np.zeros(n, dtype=np.int32)
        sum_other = np.zeros(n, dtype=np.int32)
        for v in range(n):
            sum_same[v] = int(adj[v][labels == labels[v]].sum())
            sum_other[v] = int(adj[v][labels != labels[v]].sum())

        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            order = list(range(n))
            random.shuffle(order)
            for i_idx in range(n):
                if time.time() > deadline:
                    break
                i = order[i_idx]
                for j_idx in range(i_idx + 1, n):
                    if time.time() > deadline:
                        break
                    j = order[j_idx]
                    if labels[i] == labels[j]:
                        continue

                    a = int(adj[i, j])
                    si = int(sum_same[i])
                    oi = int(sum_other[i])
                    sj = int(sum_same[j])
                    oj = int(sum_other[j])

                    gain = si + sj - oi - oj + 2 * a

                    if gain > 0:
                        li = labels[i]
                        lj = labels[j]
                        labels[i], labels[j] = labels[j], labels[i]

                        # Update sum_same/sum_other for i and j
                        sum_same[i], sum_other[i] = oi, si
                        sum_same[j], sum_other[j] = oj, sj

                        # Update all other nodes k in O(n)
                        for k in range(n):
                            if k == i or k == j:
                                continue
                            if labels[k] == li:
                                # k was same-side with i (now other), was other-side with j (now same)
                                delta = int(adj[k, j]) - int(adj[k, i])
                                sum_same[k] += delta
                                sum_other[k] -= delta
                            else:
                                # k was other-side with i (now same), was same-side with j (now other)
                                delta = int(adj[k, i]) - int(adj[k, j])
                                sum_same[k] += delta
                                sum_other[k] -= delta

                        improved = True

        return labels

    def try_labels(labels):
        nonlocal best_labels, best_cut
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # --- GW-inspired Initialization ---
    def goemans_williamson_init(num_hyperplanes=10):
        """
        Approximate GW SDP relaxation using spectral embedding.
        For Max-Cut, the SDP relaxation maximizes sum w_ij (1 - y_i . y_j) / 2
        subject to y_i . y_i = 1.
        The spectral embedding using the Laplacian L = D - A provides vectors
        where the angle between vectors correlates with the cut structure.
        Specifically, using the Fiedler vector and subsequent eigenvectors
        provides a low-dimensional representation where a hyperplane cut
        approximates the SDP rounding.
        """
        if time.time() > deadline:
            return None

        # Compute Laplacian L = D - A
        degree = adj.sum(axis=1).astype(np.float64)
        L = np.diag(degree) - adj.astype(np.float64)

        try:
            # Eigendecomposition of L
            eigvals, eigvecs = np.linalg.eigh(L)
        except np.linalg.LinAlgError:
            return None

        # The smallest eigenvalue is 0 (trivial). The next ones (Fiedler) are most useful for partitioning.
        # We take a few eigenvectors corresponding to the smallest non-zero eigenvalues.
        # For a connected graph, eigvals[0] is 0.
        # We want to use eigenvectors that capture the "cut" structure.
        # Typically, the Fiedler vector (index 1) is the best single vector.
        # Using more vectors allows for more complex hyperplane cuts.
        
        # Select k eigenvectors. Let's use up to 5 or 10, depending on n.
        k = min(10, n - 1)
        
        # Extract eigenvectors corresponding to the smallest non-zero eigenvalues
        # Skip the first one (index 0) which is the all-ones vector (or constant)
        V = eigvecs[:, 1:k+1]
        
        # Normalize rows to unit length to mimic SDP constraint y_i . y_i = 1
        # This makes the hyperplane rounding more faithful to the GW algorithm
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        V_normalized = V / norms
        
        # Generate random hyperplanes
        # A hyperplane is defined by a random unit vector u in R^k
        # Label i is 0 if u . V_i >= 0, else 1
        
        inits = []
        for _ in range(num_hyperplanes):
            # Random unit vector in R^k
            u = np.random.randn(k)
            u /= np.linalg.norm(u)
            
            # Project vertices onto hyperplane
            projections = V_normalized @ u
            
            # Round to 0/1
            labels = (projections < 0).astype(np.int8)
            
            # Ensure we don't have trivial solutions (all 0 or all 1)
            if np.all(labels == 0) or np.all(labels == 1):
                continue
                
            inits.append(labels)
            
        return inits

    # Strategy 1: GW-based initializations
    gw_inits = goemans_williamson_init(num_hyperplanes=20)
    if gw_inits:
        for labels in gw_inits:
            if time.time() > deadline:
                break
            # Polish with local search
            polished = greedy_1flip(labels, max_passes=5)
            polished = fast_2swap(polished, max_passes=50)
            try_labels(polished)

    # Strategy 2: Alternating start (backup)
    if time.time() < deadline:
        labels = np.arange(n) % 2
        labels = greedy_1flip(labels, max_passes=10)
        labels = fast_2swap(labels, max_passes=80)
        try_labels(labels)

    # Strategy 3: Multiple random restarts with varied initialization
    for restart in range(20):
        if time.time() > deadline:
            break

        # Vary initialization strategy
        if restart % 3 == 0:
            labels = np.random.randint(0, 2, size=n)
        elif restart % 3 == 1:
            # Biased random: start with slight imbalance
            bias = 0.4 + 0.2 * random.random()
            labels = (np.random.random(n) < bias).astype(np.int8)
        else:
            # Perturbed alternating
            labels = np.arange(n) % 2
            flip_mask = np.random.random(n) < 0.3
            labels[flip_mask] = 1 - labels[flip_mask]

        labels = greedy_1flip(labels, max_passes=5)
        labels = fast_2swap(labels, max_passes=40)
        try_labels(labels)

    # Final polish on best found
    if best_labels is not None and time.time() < deadline:
        final_labels = best_labels.copy()
        final_labels = fast_2swap(final_labels, max_passes=100)
        try_labels(final_labels)

        # One more 1-flip polish
        if time.time() < deadline:
            final_labels = greedy_1flip(final_labels, max_passes=5)
            final_labels = fast_2swap(final_labels, max_passes=50)
            try_labels(final_labels)

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels
# EVOLVE-BLOCK-END
