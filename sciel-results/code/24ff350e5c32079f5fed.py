"""Combined Maximum Cut solver: spectral multi-start + batched vectorized local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.asarray(adjacency, dtype=np.int8)
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    total_edges = adj.sum() // 2
    degs = adj.sum(axis=1)

    def cut_size(labels):
        """Compute cut size from binary labels using quadratic form."""
        s = 2 * labels - 1
        val = s @ (adj @ s)
        return total_edges - val // 2

    def compute_gains(labels):
        """Vectorized computation of flip gains for all vertices."""
        s = 2 * labels - 1
        # Number of edges from v to vertices on the same side
        same_mask = (s[:, None] == s[None, :]).astype(np.int8)
        same_counts = (adj * same_mask).sum(axis=1)
        # Gain = 2 * same_side_edges - degree
        gains = 2 * same_counts - degs
        return gains

    def batch_local_search(labels, time_limit=None, max_iters=100000):
        """
        Batched vectorized local search: in each iteration, evaluate all gains,
        select the best non-conflicting set of positive-gain flips, apply them
        simultaneously, and repeat until no positive gains remain.
        """
        labels = labels.copy()
        iters = 0
        while iters < max_iters:
            if time_limit is not None and time.time() > time_limit:
                break
            iters += 1
            
            gains = compute_gains(labels)
            positive = np.where(gains > 0)[0]
            if len(positive) == 0:
                break
            
            # Select best non-conflicting flips:
            # Sort by gain descending, greedily pick vertices that don't conflict
            # (i.e., are not neighbors of already-picked vertices)
            order = positive[np.argsort(-gains[positive])]
            selected = []
            blocked = np.zeros(n, dtype=bool)
            
            for v in order:
                if blocked[v]:
                    continue
                selected.append(v)
                # Block neighbors of v (they conflict)
                blocked[v] = True
                neighbors = np.nonzero(adj[v])[0]
                blocked[neighbors] = True
            
            if not selected:
                break
            
            # Apply all selected flips simultaneously
            labels[selected] = 1 - labels[selected]
        
        return labels

    def spectral_init(adj, n):
        """Spectral initialization using Fiedler vector (second eigenvector of Laplacian)."""
        deg = adj.sum(axis=1).astype(float)
        L = np.diag(deg) - adj.astype(float)
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            fiedler = eigenvectors[:, 1]
            labels = (fiedler >= 0).astype(int)
            if np.all(labels == labels[0]):
                labels = np.random.randint(0, 2, size=n)
            return labels
        except:
            return np.random.randint(0, 2, size=n)

    def spectral_kmeans_init(adj, n, k=3):
        """
        Generate diverse initial partitions using spectral clustering via
        k-means on the top-k Laplacian eigenvectors. Returns a list of
        different 0/1 labelings by projecting the k-means assignments onto
        a binary split.
        """
        deg = adj.sum(axis=1).astype(float)
        L = np.diag(deg) - adj.astype(float)
        
        inits = []
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            # Use the first k eigenvectors (smallest eigenvalues)
            X = eigenvectors[:, :k]
            
            # Run a simple k-means-like clustering
            # Initialize cluster centers randomly from data points
            rng = np.random.default_rng(42)
            centers = X[rng.choice(n, k, replace=False)].copy()
            
            for _ in range(50):  # k-means iterations
                # Assign each point to nearest center
                dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
                assignments = np.argmin(dists, axis=1)
                
                # Update centers
                new_centers = np.zeros_like(centers)
                for j in range(k):
                    mask = assignments == j
                    if np.any(mask):
                        new_centers[j] = X[mask].mean(axis=0)
                    else:
                        new_centers[j] = centers[j]
                
                if np.allclose(centers, new_centers):
                    break
                centers = new_centers
            
            # Create binary splits from k-means assignments:
            # For each pair of clusters, create a 0/1 split
            for i in range(k):
                for j in range(i + 1, k):
                    labels = np.zeros(n, dtype=int)
                    labels[assignments == i] = 1
                    labels[assignments == j] = 1
                    # If one side is empty or too small, skip
                    if 0.1 * n < labels.sum() < 0.9 * n:
                        inits.append(labels)
            
            # Also use the Fiedler vector directly
            fiedler = eigenvectors[:, 1]
            labels = (fiedler >= 0).astype(int)
            if not np.all(labels == labels[0]):
                inits.append(labels)
                
        except:
            pass
        
        if not inits:
            inits.append(np.random.randint(0, 2, size=n))
        
        return inits

    def greedy_init(adj, n):
        """Greedy initialization: add vertices one by one to the side that maximizes cut."""
        labels = np.zeros(n, dtype=int)
        order = np.arange(n)
        np.random.shuffle(order)
        
        for v in order:
            left_count = (adj[v] * (labels == 0)).sum()
            right_count = (adj[v] * (labels == 1)).sum()
            if right_count > left_count:
                labels[v] = 0
            else:
                labels[v] = 1
        return labels

    def sa_from_start(labels, adj, n, t_start, t_end, iters, time_limit=None):
        """Simulated annealing for escaping local optima."""
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        t = t_start
        cooling = (t_end / t_start) ** (1.0 / max(iters, 1))
        
        for it in range(iters):
            if time_limit is not None and time.time() > time_limit:
                break
            t *= cooling
            
            v = random.randint(0, n - 1)
            s = 2 * labels - 1
            same_side = (adj[v] * (s == s[v])).sum()
            gain = 2 * same_side - degs[v]
            
            if gain > 0 or (t > 0.001 and random.random() < np.exp(gain / t)):
                labels[v] = 1 - labels[v]
                current_cut += gain
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
        
        return best_labels, best_cut

    def pair_flip_search(labels, time_limit=None, max_iters=30000):
        """Try flipping pairs of vertices to escape local optima."""
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        iters = 0
        while iters < max_iters:
            if time_limit is not None and time.time() > time_limit:
                break
            iters += 1
            
            v1 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if v1 == v2:
                continue
            
            new_labels = labels.copy()
            new_labels[v1] = 1 - new_labels[v1]
            new_labels[v2] = 1 - new_labels[v2]
            new_cut = cut_size(new_labels)
            
            if new_cut > best_cut:
                best_cut = new_cut
                best_labels = new_labels.copy()
                labels = new_labels
        
        return best_labels, best_cut

    best_labels = None
    best_cut = -1

    # Phase 1: Generate diverse initial partitions
    starts = []
    
    # Spectral k-means based inits (diverse)
    starts.extend(spectral_kmeans_init(adj, n, k=3))
    
    # Standard spectral init
    starts.append(spectral_init(adj, n))
    
    # Alternating
    starts.append(np.arange(n) % 2)
    
    # Greedy
    starts.append(greedy_init(adj, n))
    
    # Random starts
    for _ in range(6):
        starts.append(np.random.randint(0, 2, size=n))

    # Phase 2: Refine each start with batched local search
    for i, init_labels in enumerate(starts):
        if time.time() > deadline:
            break
        
        ls_labels = batch_local_search(init_labels, time_limit=deadline)
        c = cut_size(ls_labels)
        
        if c > best_cut:
            best_cut = c
            best_labels = ls_labels.copy()

    # Phase 3: Simulated annealing from best solution
    if time.time() < deadline - 25:
        sa_time_limit = deadline - 25
        sa_labels, sa_cut = sa_from_start(
            best_labels, adj, n,
            t_start=3.0, t_end=0.01,
            iters=400000,
            time_limit=sa_time_limit
        )
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
        
        # Polish after SA
        if time.time() < deadline - 10:
            ls2 = batch_local_search(best_labels, time_limit=deadline - 10)
            c2 = cut_size(ls2)
            if c2 > best_cut:
                best_cut = c2
                best_labels = ls2.copy()

    # Phase 4: Pair flip search to escape local optima
    if time.time() < deadline - 20:
        pf_labels, pf_cut = pair_flip_search(
            best_labels,
            time_limit=deadline - 20,
            max_iters=50000
        )
        if pf_cut > best_cut:
            best_cut = pf_cut
            best_labels = pf_labels.copy()
        
        # Polish after pair flips
        if time.time() < deadline - 10:
            ls3 = batch_local_search(best_labels, time_limit=deadline - 10)
            c3 = cut_size(ls3)
            if c3 > best_cut:
                best_cut = c3
                best_labels = ls3.copy()

    # Phase 5: Second SA pass with different parameters
    if time.time() < deadline - 15:
        sa2_labels, sa2_cut = sa_from_start(
            best_labels, adj, n,
            t_start=1.5, t_end=0.005,
            iters=300000,
            time_limit=deadline - 15
        )
        if sa2_cut > best_cut:
            best_cut = sa2_cut
            best_labels = sa2_labels.copy()
        
        # Final polish
        if time.time() < deadline:
            final = batch_local_search(best_labels, time_limit=deadline)
            final_cut = cut_size(final)
            if final_cut > best_cut:
                best_labels = final

    # Final safety check
    if time.time() > deadline - 5 and best_labels is not None:
        return best_labels

    if best_labels is None:
        best_labels = np.random.randint(0, 2, size=n)
    
    return best_labels
# EVOLVE-BLOCK-END
