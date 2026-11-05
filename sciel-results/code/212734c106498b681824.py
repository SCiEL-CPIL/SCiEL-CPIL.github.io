"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    # Deadline guard: aim to finish well under 150s.
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Precompute adjacency as a boolean numpy array for fast operations
    A = np.asarray(adjacency, dtype=bool)
    degrees = A.sum(axis=1).astype(np.int64)
    
    # Helper to compute cut size efficiently using gain array
    def cut_size_from_gains(gains):
        # Sum of positive gains / 2 gives the cut size
        # Actually, let's compute directly for safety
        # For each vertex, count edges to different partition
        # cut = sum_v (A[v] & (labels != labels[v])).sum() / 2
        # But we don't have labels here. Let's use a different approach.
        # The cut size = sum over v of (number of neighbors in opposite partition) / 2
        # If we maintain gains where gains[v] = (edges_to_diff - edges_to_same),
        # then cut = sum_v (degree[v] + gains[v]) / 4
        # Because edges_to_diff = (degree + gains) / 2, and cut = sum(edges_to_diff) / 2
        return np.sum(degrees + gains) // 4
    
    # Initialize gains: gains[v] = (edges_to_diff - edges_to_same) for current labeling
    def compute_gains(labels):
        gains = np.zeros(n, dtype=np.int64)
        for v in range(n):
            same = (labels == labels[v])
            diff = (labels != labels[v])
            edges_to_same = A[v, same].sum()
            edges_to_diff = A[v, diff].sum()
            gains[v] = edges_to_diff - edges_to_same
        return gains
    
    # Local search: flip vertices that improve the cut
    # Returns updated labels, gains, and whether improvement was made
    def local_search(labels, gains, max_iters=1000):
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iters += 1
            # Find vertex with maximum positive gain
            max_gain_idx = np.argmax(gains)
            max_gain = gains[max_gain_idx]
            if max_gain <= 0:
                break
            
            v = max_gain_idx
            # Flip v
            labels[v] = 1 - labels[v]
            
            # Update gains for v and its neighbors
            # v's gain becomes -gains[v]
            gains[v] = -gains[v]
            
            # For each neighbor u of v:
            #   If labels[u] == labels[v] (new label of v):
            #      They were DIFF before, now SAME. gains[u] decreases by 2.
            #   Else:
            #      They were SAME before, now DIFF. gains[u] increases by 2.
            neighbors = np.where(A[v])[0]
            for u in neighbors:
                if labels[u] == labels[v]:
                    gains[u] -= 2
                else:
                    gains[u] += 2
            
            improved = True
        
        return labels, gains
    
    # Simulated annealing step
    def simulated_annealing(labels, gains, temperature):
        if time.time() > deadline:
            return labels, gains, False
        
        # Pick a random vertex to flip
        v = random.randint(0, n - 1)
        gain = gains[v]
        
        # Accept with probability based on gain and temperature
        if gain > 0:
            accept = True
        elif temperature > 0:
            # gain is negative, accept with probability exp(gain / temperature)
            # gain is in units of 2 per edge, so we scale
            prob = np.exp(gain / (2.0 * temperature))
            accept = random.random() < prob
        else:
            accept = False
        
        if not accept:
            return labels, gains, False
        
        # Flip v
        old_label = labels[v]
        labels[v] = 1 - labels[v]
        
        # Update gains
        gains[v] = -gains[v]
        neighbors = np.where(A[v])[0]
        for u in neighbors:
            if labels[u] == labels[v]:
                gains[u] -= 2
            else:
                gains[u] += 2
        
        return labels, gains, True
    
    best_labels = None
    best_cut = -1
    
    # Multiple restarts with different initializations
    num_restarts = 100  # Will be limited by time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Alternating
            labels = np.arange(n) % 2
        elif restart == 1:
            # Random
            labels = np.random.randint(0, 2, n)
        elif restart == 2:
            # Greedy initialization: add vertices one by one, assign to side that maximizes cut
            labels = np.zeros(n, dtype=np.int8)
            order = np.random.permutation(n)
            for v in order:
                # Count edges to each side
                same_count = A[v, labels == 0].sum() if labels[v] == 1 else A[v, labels == 1].sum()
                diff_count = A[v, labels == 1].sum() if labels[v] == 0 else A[v, labels == 0].sum()
                # Actually, for new vertex, we want to maximize edges to opposite side
                # If we put v in side 0, edges to side 1 contribute to cut
                # If we put v in side 1, edges to side 0 contribute to cut
                edges_to_0 = A[v, labels == 0].sum()
                edges_to_1 = A[v, labels == 1].sum()
                # If we put v in side 0, cut contribution from v's edges = edges_to_1
                # If we put v in side 1, cut contribution from v's edges = edges_to_0
                if edges_to_1 >= edges_to_0:
                    labels[v] = 0
                else:
                    labels[v] = 1
        elif restart == 3:
            # Spectral initialization: use eigenvector of Laplacian
            try:
                D = np.diag(degrees)
                L = D - A.astype(np.float64)
                eigvals, eigvecs = np.linalg.eigh(L)
                # Use second smallest eigenvector (Fiedler vector)
                idx = np.argsort(eigvals)[1]
                vec = eigvecs[:, idx]
                median_val = np.median(vec)
                labels = (vec > median_val).astype(np.int8)
            except:
                labels = np.random.randint(0, 2, n)
        else:
            # Random
            labels = np.random.randint(0, 2, n)
        
        # Compute initial gains
        gains = compute_gains(labels)
        
        # Run local search to convergence
        labels, gains = local_search(labels, gains, max_iters=5000)
        
        # Compute cut size
        cur_cut = cut_size_from_gains(gains)
        
        if cur_cut > best_cut:
            best_cut = cur_cut
            best_labels = labels.copy()
        
        # If we have time, run simulated annealing from this point
        if time.time() < deadline - 5.0:
            sa_labels = labels.copy()
            sa_gains = gains.copy()
            sa_cut = cur_cut
            
            # Annealing schedule
            T_start = 5.0
            T_end = 0.1
            num_sa_iters = 2000
            
            for sa_iter in range(num_sa_iters):
                if time.time() > deadline - 3.0:
                    break
                
                # Linear cooling
                progress = sa_iter / num_sa_iters
                T = T_start * (T_end / T_start) ** progress
                
                sa_labels, sa_gains, accepted = simulated_annealing(sa_labels, sa_gains, T)
                
                if accepted:
                    # Compute new cut
                    new_cut = cut_size_from_gains(sa_gains)
                    if new_cut > sa_cut:
                        sa_cut = new_cut
                        if sa_cut > best_cut:
                            best_cut = sa_cut
                            best_labels = sa_labels.copy()
                
                # Periodically run local search
                if sa_iter % 100 == 0 and time.time() < deadline - 3.0:
                    sa_labels, sa_gains = local_search(sa_labels, sa_gains, max_iters=100)
                    new_cut = cut_size_from_gains(sa_gains)
                    if new_cut > sa_cut:
                        sa_cut = new_cut
                        if sa_cut > best_cut:
                            best_cut = sa_cut
                            best_labels = sa_labels.copy()
    
    if best_labels is None:
        # Fallback
        best_labels = np.arange(n) % 2
    
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
