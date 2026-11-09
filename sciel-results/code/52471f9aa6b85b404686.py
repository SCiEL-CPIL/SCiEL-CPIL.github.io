"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin before 150s kill

    adj = np.array(adjacency, dtype=np.int8)
    best_labels = None
    best_cut = -1

    # Precompute neighbor lists for fast access
    neighbors = [np.where(adj[i] == 1)[0] for i in range(n)]

    def compute_cut(labels):
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2

    def fast_2swap_local_search(labels, max_passes=30):
        """
        2-swap local search: try swapping labels of two vertices.
        A swap is beneficial if the total change in cut size is positive.
        """
        labels = labels.copy()
        current_cut = compute_cut(labels)
        
        # Precompute for each vertex: sum_same and sum_opp
        # For efficiency in 2-swap, we need to be careful with updates.
        # Let's use a simpler approach for 2-swap: 
        # Delta for swapping u and v:
        # Let su = sum_same[u], ou = sum_opp[u]
        # Let sv = sum_same[v], ov = sum_opp[v]
        # Edge (u,v) exists? Let e_uv = adj[u,v]
        
        # If we swap u and v:
        # u's neighbors: 
        #   Those in same side as u (excluding v if e_uv=1 and v was same? No, v's label changes too)
        # This is complex. Let's stick to 1-flip for now but make it faster.
        # The inspiration mentions 2-swap, but 1-flip with precomputed neighbors might be enough.
        
        # Let's implement a fast 1-flip local search using precomputed neighbors.
        # We maintain sum_same and sum_opp arrays.
        sum_same = np.zeros(n, dtype=np.int32)
        sum_opp = np.zeros(n, dtype=np.int32)
        
        for v in range(n):
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    sum_same[v] += 1
                else:
                    sum_opp[v] += 1
        
        for _ in range(max_passes):
            if time.time() > deadline - 1.0:
                break
            changed = False
            # Iterate in random order to avoid bias
            order = np.random.permutation(n)
            for v in order:
                delta = sum_same[v] - sum_opp[v]
                if delta > 0:
                    # Flip v
                    labels[v] = 1 - labels[v]
                    current_cut += delta
                    sum_same[v], sum_opp[v] = sum_opp[v], sum_same[v]
                    
                    for u in neighbors[v]:
                        if labels[u] == labels[v]:
                            # u is now same side. Before flip, u was opp side relative to v.
                            # So for u: v moves from opp to same.
                            sum_same[u] += 1
                            sum_opp[u] -= 1
                        else:
                            # u is now opp side. Before flip, u was same side relative to v.
                            # So for u: v moves from same to opp.
                            sum_same[u] -= 1
                            sum_opp[u] += 1
                    changed = True
            if not changed:
                break
        
        return labels, current_cut

    def fast_1flip_local_search(labels, max_passes=50):
        """
        Fast 1-flip local search.
        """
        labels, cut = fast_2swap_local_search(labels, max_passes)
        return labels, cut

    def simulated_annealing(labels, T_initial, T_final, steps):
        """
        Simulated annealing for Max-Cut with incremental updates.
        """
        labels = labels.copy()
        current_cut = compute_cut(labels)
        best_local_cut = current_cut
        best_local_labels = labels.copy()
        
        # Initialize sum_same and sum_opp
        sum_same = np.zeros(n, dtype=np.int32)
        sum_opp = np.zeros(n, dtype=np.int32)
        
        for v in range(n):
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    sum_same[v] += 1
                else:
                    sum_opp[v] += 1
        
        T = T_initial
        if T_initial > 0 and T_final > 0:
            cf = (T_final / T_initial) ** (1.0 / steps)
        else:
            cf = 0.999

        for step in range(steps):
            if time.time() > deadline - 2.0:
                break
            
            # Pick random vertex
            v = random.randint(0, n - 1)
            
            delta = sum_same[v] - sum_opp[v]
            
            accept = False
            if delta > 0:
                accept = True
            else:
                if T > 0:
                    ratio = np.exp(delta / T)
                    if random.random() < ratio:
                        accept = True
            
            if accept:
                labels[v] = 1 - labels[v]
                current_cut += delta
                sum_same[v], sum_opp[v] = sum_opp[v], sum_same[v]
                
                for u in neighbors[v]:
                    if labels[u] == labels[v]:
                        sum_same[u] += 1
                        sum_opp[u] -= 1
                    else:
                        sum_same[u] -= 1
                        sum_opp[u] += 1
                
                if current_cut > best_local_cut:
                    best_local_cut = current_cut
                    best_local_labels = labels.copy()
            
            T *= cf
            
        return best_local_labels, best_local_cut

    def spectral_init():
        """Spectral initialization using Laplacian eigenvectors."""
        degree = adj.sum(axis=1).astype(np.float64)
        D = np.diag(degree)
        L = D - adj.astype(np.float64)
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            k = min(5, n - 1)
            if k >= 2:
                Y = eigvecs[:, 1:k+1]
                norms = np.linalg.norm(Y, axis=1, keepdims=True)
                norms[norms < 1e-10] = 1.0
                Y = Y / norms
                u = np.random.randn(k)
                u = u / np.linalg.norm(u)
                proj = Y @ u
                labels = (proj > 0).astype(np.int8)
            else:
                labels = np.random.randint(0, 2, size=n)
        except:
            labels = np.random.randint(0, 2, size=n)
        return labels

    def try_labels(labels):
        nonlocal best_labels, best_cut
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Strategy: Multiple SA restarts with different initializations.
    
    # Initial run with alternating start
    if time.time() < deadline:
        labels = np.arange(n) % 2
        polished, cut = fast_1flip_local_search(labels, 10)
        try_labels(polished)

    # SA Restarts
    num_restarts = 25
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:  # Leave 15s for final polish
            break
        
        # Initialization
        if r % 3 == 0:
            labels = np.random.randint(0, 2, size=n)
        elif r % 3 == 1:
            labels = np.arange(n) % 2
            flip_mask = np.random.random(n) < 0.3
            labels[flip_mask] = 1 - labels[flip_mask]
        else:
            labels = spectral_init()
        
        # SA Parameters
        T_initial = n / 5.0
        T_final = 0.1
        steps = 3000
        
        sa_labels, sa_cut = simulated_annealing(labels, T_initial, T_final, steps)
        try_labels(sa_labels)
        
        # Local search polish on the best found in SA
        if time.time() < deadline - 10.0:
            polished, cut = fast_1flip_local_search(sa_labels, 20)
            try_labels(polished)

    # Final polish on best found
    if best_labels is not None and time.time() < deadline - 1.0:
        final_labels, final_cut = fast_1flip_local_search(best_labels, 50)
        try_labels(final_labels)

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels
# EVOLVE-BLOCK-END
