import numpy as np
import time

def partition(adjacency):
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency, dtype=np.int8)
    n = adjacency.shape[0]
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Precompute neighbor lists for fast updates
    # adjacency is int8/bool. 
    # neighbors[i] = list of indices j where adjacency[i,j] == 1
    neighbors = [np.where(adjacency[i] == 1)[0] for i in range(n)]
    
    # We will work with labels in {0, 1}
    # Gain array G[i] = change in cut size if we flip i.
    # If label[i] == 0: G[i] = (# neighbors with 1) - (# neighbors with 0)
    # If label[i] == 1: G[i] = (# neighbors with 0) - (# neighbors with 1)
    
    def compute_gains(labels):
        """Compute initial gains for all vertices."""
        gains = np.zeros(n, dtype=np.int32)
        # count_0[i] = number of neighbors of i with label 0
        # count_1[i] = number of neighbors of i with label 1
        # We can compute this efficiently
        # mask_0 = (labels == 0)
        # mask_1 = (labels == 1)
        # count_0 = adjacency @ mask_0
        # count_1 = adjacency @ mask_1
        
        mask_0 = (labels == 0).astype(np.int8)
        mask_1 = (labels == 1).astype(np.int8)
        
        count_0 = adjacency @ mask_0
        count_1 = adjacency @ mask_1
        
        gains = np.where(labels == 0, count_1 - count_0, count_0 - count_1)
        return gains

    def cut_size(labels):
        # Efficient cut size calculation
        # Sum of edges between 0s and 1s
        # Using boolean indexing is fast for N=100
        mask_0 = (labels == 0)
        mask_1 = (labels == 1)
        # Sum adjacency entries where one is 0 and other is 1
        # This is symmetric, so we can just sum A[i,j] for i in 0, j in 1
        # Or simply: total_edges - (edges within 0) - (edges within 1)
        # But direct sum is fine.
        # A[np.ix_(mask_0, mask_1)] gets the submatrix
        return int(np.sum(adjacency[np.ix_(mask_0, mask_1)]))

    def improve(labels):
        """
        Local search with incremental gain updates.
        """
        current_labels = labels.copy()
        gains = compute_gains(current_labels)
        
        # We will repeatedly pick the vertex with max positive gain
        # To avoid $O(N)$ scan every time, we can just scan. N=100 is small.
        # But updating gains is the key.
        
        max_iters = 5000
        for _ in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            
            max_gain = np.max(gains)
            if max_gain <= 0:
                break # Local optimum
            
            # Pick vertex with max gain
            # If ties, pick randomly
            best_indices = np.where(gains == max_gain)[0]
            v = best_indices[np.random.randint(len(best_indices))]
            
            # Flip v
            old_label = current_labels[v]
            new_label = 1 - old_label
            current_labels[v] = new_label
            
            # Update gains
            # 1. Update gain for v
            gains[v] = -gains[v]
            
            # 2. Update gains for neighbors of v
            for u in neighbors[v]:
                # If u and v were same label, they contributed 0 to cut. Now they contribute 1.
                # If u and v were diff label, they contributed 1. Now 0.
                
                # Let's use the formula derived:
                # If u == old_label (same as v before flip):
                #   They were in same set.
                #   If u==0, v==0. G[u] = N1(u) - N0(u).
                #   v was in N0(u). Now v is in N1(u).
                #   N0(u) decreases by 1, N1(u) increases by 1.
                #   G[u] increases by 2.
                # If u != old_label (diff from v before flip):
                #   They were in diff sets.
                #   If u==1, v==0. G[u] = N0(u) - N1(u).
                #   v was in N0(u). Now v is in N1(u).
                #   N0(u) decreases by 1, N1(u) increases by 1.
                #   G[u] = (N0-1) - (N1+1) = G[u] - 2.
                
                if current_labels[u] == new_label:
                    # u is now same as v.
                    # Before: u was same as old_v?
                    # If current_labels[u] == new_label, and new_label != old_label.
                    # Then current_labels[u] != old_label.
                    # So u and v were in DIFFERENT sets before.
                    # So G[u] decreases by 2.
                    gains[u] -= 2
                else:
                    # u is now different from v.
                    # Before: u was same as old_v.
                    # So u and v were in SAME set before.
                    # So G[u] increases by 2.
                    gains[u] += 2
                    
        return current_labels

    best_labels = None
    best_cut = -1
    
    total_edges = int(np.sum(adjacency) // 2)
    
    # Strategy 1: Random Restarts
    # With faster improve, we can do more restarts.
    
    for i in range(500):
        if time.time() > deadline - 10.0:
            break
            
        if i == 0:
            labels = np.zeros(n, dtype=np.int8)
        elif i == 1:
            labels = np.ones(n, dtype=np.int8)
        elif i == 2:
            labels = np.arange(n) % 2
        else:
            labels = np.random.randint(0, 2, size=n)
            
        improved_labels = improve(labels)
        c = cut_size(improved_labels)
        
        if c > best_cut:
            best_cut = c
            best_labels = improved_labels.copy()
            
        if best_cut == total_edges:
            break
            
    # Strategy 2: Simulated Annealing
    # Use incremental gain for cost evaluation?
    # SA move: flip v. Delta = G[v].
    # If accepted, update gains as in improve.
    
    if best_labels is not None and time.time() < deadline - 5.0:
        current_labels = best_labels.copy()
        current_gains = compute_gains(current_labels)
        current_cut = best_cut
        
        T = 10.0
        cooling = 0.995
        min_T = 0.01
        
        while time.time() < deadline - 2.0 and T > min_T:
            v = np.random.randint(n)
            
            delta = current_gains[v]
            
            # Accept if delta > 0 or with prob exp(-delta/T)
            if delta > 0 or np.random.random() < np.exp(-delta / T):
                # Accept move
                old_label = current_labels[v]
                new_label = 1 - old_label
                current_labels[v] = new_label
                current_cut += delta
                
                # Update gains
                current_gains[v] = -delta
                
                for u in neighbors[v]:
                    if current_labels[u] == new_label:
                        current_gains[u] -= 2
                    else:
                        current_gains[u] += 2
                        
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = current_labels.copy()
            else:
                # Reject, no change to state or gains
                
            T *= cooling
            
            # Periodic polish
            if int(T * 1000) % 50 == 0:
                polished = improve(current_labels)
                pc = cut_size(polished)
                if pc > current_cut:
                    current_labels = polished
                    current_cut = pc
                    current_gains = compute_gains(current_labels)
                    if pc > best_cut:
                        best_cut = pc
                        best_labels = polished.copy()

    return best_labels if best_labels is not None else np.zeros(n, dtype=np.int8)
