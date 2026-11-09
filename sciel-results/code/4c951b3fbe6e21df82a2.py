import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Convert to numpy array for efficiency
    adj = np.array(adjacency, dtype=np.int32)
    # Ensure symmetry
    adj = (adj + adj.T) // 2
    np.fill_diagonal(adj, 0)
    
    # Precompute neighbor lists and degrees
    neighbors = [np.where(adj[i] > 0)[0] for i in range(n)]
    degrees = np.array([len(neighbors[i]) for i in range(n)], dtype=np.int32)
    
    def compute_cut(labels):
        """Compute cut size using vectorized operations."""
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2
        
    def init_same_count(labels):
        """Initialize the count of same-side neighbors for each vertex."""
        same_count = np.zeros(n, dtype=np.int32)
        for v in range(n):
            same_count[v] = np.sum(adj[v, :] * (labels == labels[v]))
        return same_count

    def flip_greedy(labels, same_count, max_passes=100):
        """Fast 1-flip greedy using incremental cut tracking."""
        labels = labels.copy()
        same_count = same_count.copy()
        
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            changed = False
            order = np.random.permutation(n)
            for v in order:
                if time.time() > deadline:
                    break
                deg = degrees[v]
                gain = 2 * same_count[v] - deg
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    changed = True
                    new_same_v = deg - same_count[v]
                    same_count[v] = new_same_v
                    
                    for u in neighbors[v]:
                        if labels[u] == labels[v]:
                            same_count[u] += 1
                        else:
                            same_count[u] -= 1
            if not changed:
                break
        return labels, same_count

    def cerny_tabu_search(labels, same_count, max_iter=2000, tabu_len=15):
        """Cerny tabu search with aspiration criterion."""
        labels = labels.copy()
        same_count = same_count.copy()
        current_cut = compute_cut(labels)
        best_cut = current_cut
        best_labels = labels.copy()
        
        tabu_until = np.zeros(n, dtype=np.int32)
        iteration = 0
        
        while iteration < max_iter:
            if time.time() > deadline:
                break
            
            best_gain = -999999
            best_v = -1
            best_gain_aspiration = -999999
            best_v_aspiration = -1
            
            for v in range(n):
                if tabu_until[v] > iteration:
                    continue
                deg = degrees[v]
                sc = same_count[v]
                gain = 2 * sc - deg
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            
            # Aspiration: if no non-tabu move improves, allow best tabu move if it improves best_cut
            if best_v == -1 or best_gain <= 0:
                for v in range(n):
                    deg = degrees[v]
                    sc = same_count[v]
                    gain = 2 * sc - deg
                    if current_cut + gain > best_cut:
                        if gain > best_gain_aspiration:
                            best_gain_aspiration = gain
                            best_v_aspiration = v
            
            if best_v != -1 and best_gain > 0:
                v = best_v
                gain = best_gain
            elif best_v_aspiration != -1:
                v = best_v_aspiration
                gain = best_gain_aspiration
            else:
                break
            
            # Perform flip
            labels[v] = 1 - labels[v]
            tabu_until[v] = iteration + tabu_len
            
            deg = degrees[v]
            old_sc = same_count[v]
            same_count[v] = deg - old_sc
            
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[u] += 1
                else:
                    same_count[u] -= 1
            
            current_cut += gain
            iteration += 1
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
                
        return best_labels, init_same_count(best_labels)

    def spectral_init():
        """Spectral initialization using Fiedler vector."""
        # Build Laplacian L = D - A
        D = np.diag(degrees.astype(np.float64))
        L = D - adj.astype(np.float64)
        
        # Compute eigenvectors
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
            # Sort by eigenvalue
            idx = np.argsort(eigvals)
            fiedler = eigvecs[:, idx[1]]
            
            # Threshold at mean
            threshold = np.mean(fiedler)
            labels = (fiedler > threshold).astype(np.int8)
        except:
            labels = np.random.randint(0, 2, n)
        
        return labels

    def multi_start_search(num_restarts=25):
        """Multi-start with diverse initializations."""
        best_labels_global = None
        best_cut_global = -1
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Diverse initialization
            if restart == 0:
                labels = spectral_init()
            elif restart == 1:
                # Random
                labels = np.random.randint(0, 2, n)
            elif restart == 2:
                # Random with slight bias
                labels = (np.random.rand(n) > 0.55).astype(np.int8)
            elif restart == 3:
                # Cluster-like init: split by degree
                sorted_by_deg = np.argsort(degrees)
                labels = np.zeros(n, dtype=np.int8)
                labels[sorted_by_deg[:n//2]] = 1
            else:
                # Random
                labels = np.random.randint(0, 2, n)
            
            # 1. Greedy Local Search
            same_count = init_same_count(labels)
            labels, same_count = flip_greedy(labels, same_count, max_passes=50)
            
            # 2. Cerny Tabu Search
            tabu_len = 10 + (restart % 5) * 2  # Vary tabu length: 10, 12, 14, 16, 18
            labels, same_count = cerny_tabu_search(labels, same_count, max_iter=1500, tabu_len=tabu_len)
            
            # 3. Greedy Local Search again
            labels, same_count = flip_greedy(labels, same_count, max_passes=50)
            
            # 4. Another tabu with different length
            tabu_len2 = 8 + (restart % 7) * 2
            labels, same_count = cerny_tabu_search(labels, same_count, max_iter=1000, tabu_len=tabu_len2)
            
            # 5. Final Greedy
            labels, same_count = flip_greedy(labels, same_count, max_passes=30)
            
            cut = compute_cut(labels)
            if cut > best_cut_global:
                best_cut_global = cut
                best_labels_global = labels.copy()
            
            # If time is running out, do a quick final polish on best found
            if time.time() > deadline - 20:
                break
        
        return best_labels_global

    best_labels = multi_start_search(num_restarts=25)
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    # Final polish if time remains
    if time.time() < deadline - 5:
        same_count = init_same_count(best_labels)
        best_labels, same_count = flip_greedy(best_labels, same_count, max_passes=30)
    
    return best_labels
