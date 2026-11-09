"""Maximum Cut via Fiedler vector bisection + fast greedy + precomputed 2-swap local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    adj = np.array(adjacency, dtype=np.int32)
    total_edges = int(np.sum(adj)) // 2

    if n == 0:
        return np.array([], dtype=np.int8)

    # Precompute neighbor lists and degrees
    neighbors = [np.where(adj[i] > 0)[0] for i in range(n)]
    degs = np.array([len(neighbors[i]) for i in range(n)], dtype=np.int32)

    def compute_cut(labels):
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2

    def compute_same_count(labels):
        """Compute same_count[v] = number of neighbors of v in same partition as v."""
        same_count = np.zeros(n, dtype=np.int32)
        for v in range(n):
            lv = labels[v]
            for u in neighbors[v]:
                if labels[u] == lv:
                    same_count[v] += 1
        return same_count

    def flip_greedy(labels, max_passes=50):
        """Fast 1-flip greedy using incremental cut tracking."""
        labels = labels.copy()
        same_count = compute_same_count(labels)

        for _ in range(max_passes):
            if time.time() > deadline:
                break
            changed = False
            for v in range(n):
                if time.time() > deadline:
                    break
                gain = 2 * same_count[v] - degs[v]
                if gain > 0:
                    old_label = labels[v]
                    labels[v] = 1 - old_label
                    changed = True
                    new_same_v = degs[v] - same_count[v]
                    same_count[v] = new_same_v
                    for u in neighbors[v]:
                        if labels[u] == labels[v]:
                            same_count[u] += 1
                        else:
                            same_count[u] -= 1
            if not changed:
                break
        return labels

    def two_swap_search(labels, max_passes=20):
        """2-swap local search with precomputed same_count for O(1) gain checks."""
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
                    lj = labels[j]
                    if li == lj:
                        continue
                    
                    # Gain = 2*S[i] + 2*S[j] - deg[i] - deg[j] - 2*adj[i,j]
                    gain = 2 * same_count[i] + 2 * same_count[j] - degs[i] - degs[j] - 2 * adj[i, j]
                    
                    if gain > 0:
                        # Perform swap
                        labels[i] = lj
                        labels[j] = li
                        improved = True
                        
                        # Update same_count
                        # i moves from partition li to lj
                        # j moves from partition lj to li
                        # New same_count[i] = degs[i] - same_count[i]
                        # New same_count[j] = degs[j] - same_count[j]
                        new_si = degs[i] - same_count[i]
                        new_sj = degs[j] - same_count[j]
                        
                        # Update neighbors of i
                        for u in neighbors[i]:
                            if u == j:
                                continue
                            if labels[u] == lj:  # u is in i's new partition
                                same_count[u] += 1
                            elif labels[u] == li:  # u is in i's old partition
                                same_count[u] -= 1
                        
                        # Update neighbors of j
                        for u in neighbors[j]:
                            if u == i:
                                continue
                            if labels[u] == li:  # u is in j's new partition
                                same_count[u] += 1
                            elif labels[u] == lj:  # u is in j's old partition
                                same_count[u] -= 1
                        
                        same_count[i] = new_si
                        same_count[j] = new_sj
            
            if not improved:
                break
        
        return labels

    # Strategy 1: Fiedler vector (second-smallest eigenvector of Laplacian)
    try:
        D = np.diag(degs)
        L = D - adj
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Fiedler vector: second-smallest eigenvalue (index 1, since 0 is smallest)
        fiedler = eigenvectors[:, 1]
        labels_fiedler = (fiedler > 0).astype(np.int8)
        labels_fiedler_neg = (fiedler < 0).astype(np.int8)
    except Exception:
        labels_fiedler = np.arange(n) % 2
        labels_fiedler_neg = labels_fiedler.copy()

    # Also try principal eigenvector of adjacency as fallback
    try:
        eigenvalues_a, eigenvectors_a = np.linalg.eigh(adj)
        idx = np.argmax(eigenvalues_a)
        v = eigenvectors_a[:, idx]
        labels_principal = (v > 0).astype(np.int8)
        labels_principal_neg = (v < 0).astype(np.int8)
    except Exception:
        labels_principal = np.arange(n) % 2
        labels_principal_neg = labels_principal.copy()

    best_cut = -1
    best_labels = None

    def try_candidate(labels, do_flip=True, do_2swap=True, max_flip=50, max_swap=20):
        nonlocal best_cut, best_labels
        if time.time() > deadline:
            return
        lab = labels.copy()
        if do_flip:
            lab = flip_greedy(lab, max_passes=max_flip)
        cut = compute_cut(lab)
        if cut > best_cut:
            best_cut = cut
            best_labels = lab.copy()
        if do_2swap and time.time() < deadline:
            lab2 = two_swap_search(lab, max_passes=max_swap)
            cut2 = compute_cut(lab2)
            if cut2 > best_cut:
                best_cut = cut2
                best_labels = lab2.copy()

    # Try Fiedler vector inits
    try_candidate(labels_fiedler)
    try_candidate(labels_fiedler_neg)

    # Try principal eigenvector inits
    try_candidate(labels_principal)
    try_candidate(labels_principal_neg)

    # Random restarts
    num_random = 40
    for r in range(num_random):
        if time.time() > deadline:
            break
        labels = np.random.randint(0, 2, size=n, dtype=np.int8)
        try_candidate(labels, do_flip=True, do_2swap=True, max_flip=30, max_swap=15)

    # Try alternating bit pattern
    if time.time() < deadline:
        labels_alt = np.arange(n) % 2
        try_candidate(labels_alt, do_flip=True, do_2swap=True, max_flip=50, max_swap=20)

    # Final polish: additional 2-swap passes on the best solution
    if best_labels is not None and time.time() < deadline:
        for _ in range(3):
            if time.time() > deadline:
                break
            polished = two_swap_search(best_labels, max_passes=30)
            cut_p = compute_cut(polished)
            if cut_p > best_cut:
                best_cut = cut_p
                best_labels = polished.copy()
            else:
                break

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
