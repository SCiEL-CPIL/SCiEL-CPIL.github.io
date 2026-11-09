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
    # Ensure symmetric if not, though problem states unweighted graph
    # adj = (adj + adj.T) // 2 
    # For safety, just use as is, assuming it's a valid adjacency matrix
    
    # Precompute neighbor lists and degrees
    neighbors = [np.where(adj[i] > 0)[0] for i in range(n)]
    degrees = np.array([len(neighbors[i]) for i in range(n)], dtype=np.int32)
    
    def compute_cut(labels):
        """Compute cut size using vectorized operations."""
        # diff is True where labels differ
        # Note: labels are 0/1
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2
        
    def init_same_count(labels):
        """Initialize the count of same-side neighbors for each vertex."""
        same_count = np.zeros(n, dtype=np.int32)
        for v in range(n):
            # Count neighbors u of v such that labels[u] == labels[v]
            # We can do this vectorized per v, or loop
            # Since n is small, loop is fine, but let's try to be efficient
            # same_count[v] = sum(adj[v, :] * (labels == labels[v]))
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
            # Randomize order to break symmetry
            order = np.random.permutation(n)
            for v in order:
                if time.time() > deadline:
                    break
                deg = degrees[v]
                # Gain = 2 * same - deg
                # If gain > 0, flipping increases cut
                gain = 2 * same_count[v] - deg
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    changed = True
                    # Update same_count for v
                    # New same_count[v] = deg - old_same_count[v]
                    new_same_v = deg - same_count[v]
                    same_count[v] = new_same_v
                    
                    # Update same_count for neighbors
                    for u in neighbors[v]:
                        # Before flip: labels[u] compared to old labels[v]
                        # After flip: labels[u] compared to new labels[v]
                        # If labels[u] == old labels[v], it was same, now different -> same_count[u] -= 1
                        # If labels[u] != old labels[v], it was different, now same -> same_count[u] += 1
                        
                        # We can check current labels[u] vs new labels[v]
                        # If they are same now, and were different before, +1
                        # If they are different now, and were same before, -1
                        
                        # Simpler: 
                        # old_relation = (labels[u] == 1 - labels[v]) # since labels[v] just changed
                        # Actually, just check if labels[u] == labels[v] (new)
                        # If yes, and previously they were different (which means labels[u] != 1-labels[v]_old... wait)
                        
                        # Let's use the property:
                        # If labels[u] == labels[v] (new), then previously labels[u] != labels[v] (old).
                        # So same_count[u] should increase by 1.
                        # If labels[u] != labels[v] (new), then previously labels[u] == labels[v] (old).
                        # So same_count[u] should decrease by 1.
                        
                        if labels[u] == labels[v]:
                            same_count[u] += 1
                        else:
                            same_count[u] -= 1
            if not changed:
                break
        return labels, same_count

    def tabu_search(labels, same_count, max_iter=2000, tabu_len=10):
        """Cerny tabu search."""
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
            
            # Find best non-tabu move
            for v in range(n):
                if tabu_until[v] > iteration:
                    continue
                deg = degrees[v]
                sc = same_count[v]
                gain = 2 * sc - deg
                # Aspiration: if gain is very high, allow tabu? 
                # Standard Cerny: just pick best non-tabu.
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            
            if best_v == -1:
                # All moves are tabu, break or reset tabu
                break
            
            v = best_v
            # Perform flip
            labels[v] = 1 - labels[v]
            tabu_until[v] = iteration + tabu_len
            
            # Update same_count
            deg = degrees[v]
            old_sc = same_count[v]
            same_count[v] = deg - old_sc
            
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[u] += 1
                else:
                    same_count[u] -= 1
            
            current_cut += best_gain
            iteration += 1
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
                
        return best_labels, init_same_count(best_labels)

    def two_swap_search(labels, same_count, num_samples=2000):
        """Sampled 2-swap improvement."""
        labels = labels.copy()
        same_count = same_count.copy()
        current_cut = compute_cut(labels)
        best_cut = current_cut
        best_labels = labels.copy()
        
        # Sample pairs
        # n=100, total pairs ~5000. We can afford to check many.
        # Let's do a few passes of random sampling
        for _ in range(5):
            if time.time() > deadline:
                break
            for _ in range(num_samples // 5):
                if time.time() > deadline:
                    break
                i, j = np.random.choice(n, 2, replace=False)
                if labels[i] == labels[j]:
                    continue
                
                # Try flipping i and j
                labels[i] = 1 - labels[i]
                labels[j] = 1 - labels[j]
                
                new_cut = compute_cut(labels)
                
                if new_cut > best_cut:
                    best_cut = new_cut
                    best_labels = labels.copy()
                    same_count = init_same_count(labels)
                    current_cut = new_cut
                else:
                    # Revert
                    labels[i] = 1 - labels[i]
                    labels[j] = 1 - labels[j]
                    
        return best_labels, init_same_count(best_labels)

    best_labels_global = None
    best_cut_global = -1
    
    num_restarts = 30 # Increase restarts due to faster local search
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialization
        if restart % 2 == 0:
            labels = np.random.randint(0, 2, n)
        else:
            # Spectral-like init: use principal eigenvector of Laplacian?
            # Or just random with bias
            labels = np.random.randint(0, 2, n)
            
        # 1. Greedy Local Search
        same_count = init_same_count(labels)
        labels, same_count = flip_greedy(labels, same_count, max_passes=50)
        
        # 2. Tabu Search
        labels, same_count = tabu_search(labels, same_count, max_iter=1000, tabu_len=15)
        
        # 3. Greedy Local Search again
        labels, same_count = flip_greedy(labels, same_count, max_passes=50)
        
        # 4. 2-Swap Search
        labels, same_count = two_swap_search(labels, same_count, num_samples=2000)
        
        # 5. Final Greedy
        labels, same_count = flip_greedy(labels, same_count, max_passes=20)
        
        cut = compute_cut(labels)
        if cut > best_cut_global:
            best_cut_global = cut
            best_labels_global = labels.copy()

    if best_labels_global is None:
        best_labels_global = np.arange(n) % 2
        
    return best_labels_global
