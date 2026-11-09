"""Maximum Cut via spectral init + vectorized 2-swap + simulated annealing + Cerny tabu."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

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
        same_mask = (labels[:, None] == labels[None, :]).astype(np.int32)
        same_count = (adj * same_mask).sum(axis=1).astype(np.int32)
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
        """Vectorized 2-swap local search: precompute all pair gains per pass."""
        labels = labels.copy()
        same_count = compute_same_count(labels)
        
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            
            # Precompute gain matrix for all pairs (i,j) where labels differ
            # gain[i,j] = 2*S[i] + 2*S[j] - deg[i] - deg[j] - 2*adj[i,j]
            # Vectorized:
            S = same_count.astype(np.float64)
            d = degs.astype(np.float64)
            
            # gain[i,j] for i < j, only where labels[i] != labels[j]
            # Build gain matrix
            gain_mat = 2.0 * S[:, None] + 2.0 * S[None, :] - d[:, None] - d[None, :] - 2.0 * adj
            
            # Only consider pairs where labels differ
            diff_mask = (labels[:, None] != labels[None, :])
            # Only upper triangle
            tri_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            valid_mask = diff_mask & tri_mask
            
            # Find best valid pair
            if not np.any(valid_mask):
                break
            
            # Set invalid entries to -inf
            gain_mat_valid = np.where(valid_mask, gain_mat, -1e9)
            max_gain = np.max(gain_mat_valid)
            
            if max_gain <= 0:
                break
            
            # Find the pair
            idx = np.unravel_index(np.argmax(gain_mat_valid), gain_mat.shape)
            i, j = idx[0], idx[1]
            
            # Perform swap
            li, lj = labels[i], labels[j]
            labels[i] = lj
            labels[j] = li
            
            # Update same_count
            new_si = degs[i] - same_count[i]
            new_sj = degs[j] - same_count[j]
            
            # Update neighbors of i
            for u in neighbors[i]:
                if u == j:
                    continue
                if labels[u] == lj:
                    same_count[u] += 1
                elif labels[u] == li:
                    same_count[u] -= 1
            
            # Update neighbors of j
            for u in neighbors[j]:
                if u == i:
                    continue
                if labels[u] == li:
                    same_count[u] += 1
                elif labels[u] == lj:
                    same_count[u] -= 1
            
            same_count[i] = new_si
            same_count[j] = new_sj
        
        return labels

    def simulated_annealing(labels, max_iterations=2000, T0=10.0, T_end=0.1, cooling=0.995):
        """Simulated annealing that accepts worsening moves probabilistically."""
        labels = labels.copy()
        same_count = compute_same_count(labels)
        current_cut = compute_cut(labels)
        best_cut = current_cut
        best_labels = labels.copy()
        
        T = T0
        rng = np.random.RandomState(42)
        
        for it in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Pick random vertex to flip
            v = rng.randint(n)
            gain = 2 * same_count[v] - degs[v]
            
            # Accept if improving, or with probability exp(gain/T) if worsening
            if gain >= 0 or rng.random() < np.exp(gain / max(T, 1e-10)):
                old_label = labels[v]
                labels[v] = 1 - old_label
                
                # Update same_count
                new_same_v = degs[v] - same_count[v]
                same_count[v] = new_same_v
                for u in neighbors[v]:
                    if labels[u] == labels[v]:
                        same_count[u] += 1
                    else:
                        same_count[u] -= 1
                
                current_cut += gain // 2
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            
            T *= cooling
        
        # Final greedy polish
        best_labels = flip_greedy(best_labels, max_passes=30)
        return best_labels

    def cerny_tabu_search(labels, max_iterations=500, tabu_tenure=10):
        """Cerny-style tabu search that accepts worsening moves to escape local optima."""
        labels = labels.copy()
        same_count = compute_same_count(labels)
        current_cut = compute_cut(labels)
        best_cut = current_cut
        best_labels = labels.copy()
        
        tabu_list = {}
        tabu_tenure = max(5, n // 10)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            best_gain = -10**9
            best_move = None
            best_gain_value = -10**9
            
            # Use full set for n<=100, sample for larger
            if n <= 100:
                candidates = list(range(n))
            else:
                sample_size = min(n, max(20, n // 2))
                candidates = random.sample(range(n), sample_size)
            
            for v in candidates:
                gain = 2 * same_count[v] - degs[v]
                
                is_tabu = v in tabu_list and (iteration - tabu_list[v]) < tabu_tenure
                
                if is_tabu and gain <= best_gain_value:
                    continue
                
                if gain > best_gain:
                    best_gain = gain
                    best_move = v
                    best_gain_value = gain
            
            if best_move is None:
                break
            
            v = best_move
            old_label = labels[v]
            labels[v] = 1 - old_label
            
            tabu_list[v] = iteration
            
            new_same_v = degs[v] - same_count[v]
            same_count[v] = new_same_v
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[u] += 1
                else:
                    same_count[u] -= 1
            
            current_cut += best_gain // 2
            
            if iteration % 50 == 0:
                current_cut = compute_cut(labels)
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        
        best_labels = flip_greedy(best_labels, max_passes=30)
        return best_labels

    # Strategy 1: Fiedler vector
    try:
        D = np.diag(degs)
        L = D - adj
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        fiedler = eigenvectors[:, 1]
        labels_fiedler = (fiedler > 0).astype(np.int8)
        labels_fiedler_neg = (fiedler < 0).astype(np.int8)
    except Exception:
        labels_fiedler = np.arange(n) % 2
        labels_fiedler_neg = labels_fiedler.copy()

    # Principal eigenvector
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

    def try_candidate(labels, do_flip=True, do_2swap=True, do_sa=False, do_tabu=False, 
                      max_flip=50, max_swap=20, max_sa=1000, max_tabu=200):
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
        if do_sa and time.time() < deadline:
            lab3 = simulated_annealing(lab, max_iterations=max_sa)
            cut3 = compute_cut(lab3)
            if cut3 > best_cut:
                best_cut = cut3
                best_labels = lab3.copy()
        if do_tabu and time.time() < deadline:
            lab4 = cerny_tabu_search(lab, max_iterations=max_tabu)
            cut4 = compute_cut(lab4)
            if cut4 > best_cut:
                best_cut = cut4
                best_labels = lab4.copy()

    # Try spectral inits
    try_candidate(labels_fiedler)
    try_candidate(labels_fiedler_neg)
    try_candidate(labels_principal)
    try_candidate(labels_principal_neg)

    # Random restarts with mixed strategies
    num_random = 25
    for r in range(num_random):
        if time.time() > deadline:
            break
        labels = np.random.randint(0, 2, size=n, dtype=np.int8)
        # Vary strategy: use SA or tabu for different restarts
        use_sa = (r % 3 == 0)
        use_tabu = (r % 3 == 1)
        try_candidate(labels, do_flip=True, do_2swap=True, do_sa=use_sa, do_tabu=use_tabu,
                      max_flip=30, max_swap=15, max_sa=800, max_tabu=150)

    # Alternating pattern
    if time.time() < deadline:
        labels_alt = np.arange(n) % 2
        try_candidate(labels_alt, do_flip=True, do_2swap=True, do_sa=False, do_tabu=False,
                      max_flip=50, max_swap=20)

    # Final aggressive polish on best solution
    if best_labels is not None and time.time() < deadline:
        # Multiple 2-swap passes
        for _ in range(10):
            if time.time() > deadline:
                break
            polished = two_swap_search(best_labels, max_passes=30)
            cut_p = compute_cut(polished)
            if cut_p > best_cut:
                best_cut = cut_p
                best_labels = polished.copy()
            else:
                break
        
        # SA polish if time permits
        if time.time() < deadline - 20:
            sa_result = simulated_annealing(best_labels, max_iterations=1500, T0=5.0, T_end=0.05, cooling=0.997)
            cut_sa = compute_cut(sa_result)
            if cut_sa > best_cut:
                best_cut = cut_sa
                best_labels = sa_result.copy()
        
        # Tabu polish if time permits
        if time.time() < deadline - 15:
            tabu_result = cerny_tabu_search(best_labels, max_iterations=200)
            cut_tabu = compute_cut(tabu_result)
            if cut_tabu > best_cut:
                best_cut = cut_tabu
                best_labels = tabu_result.copy()
        
        # Final 2-swap + flip
        if time.time() < deadline:
            polished = two_swap_search(best_labels, max_passes=20)
            cut_p = compute_cut(polished)
            if cut_p > best_cut:
                best_cut = cut_p
                best_labels = polished.copy()
            polished = flip_greedy(best_labels, max_passes=20)
            cut_p = compute_cut(polished)
            if cut_p > best_cut:
                best_cut = cut_p
                best_labels = polished.copy()

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
