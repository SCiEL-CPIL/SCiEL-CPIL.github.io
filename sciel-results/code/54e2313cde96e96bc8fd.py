import numpy as np
import time
import random

def partition(adjacency):
    adj_np = np.asarray(adjacency, dtype=np.int8)
    n = adj_np.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    
    # Build neighbor list for fast iteration
    neighbors = [[] for _ in range(n)]
    degrees = np.zeros(n, dtype=int)
    for i in range(n):
        # Find non-zero entries in row i
        # Since it's a bool/int matrix, we can use np.nonzero
        idxs = np.nonzero(adj_np[i])[0]
        neighbors[i] = idxs.tolist()
        degrees[i] = len(idxs)
    
    E = int(np.sum(adj_np)) // 2
    
    start_time = time.time()
    DEADLINE = start_time + 110.0
    
    def time_left():
        return DEADLINE - time.time()
        
    def cut_size(labels):
        # Fast calculation using cross array if available, or direct
        # Direct: 0.5 * sum(adj[i,j] * (labels[i] != labels[j]))
        # This is O(n^2) but vectorized.
        # Better: sum(cross) / 2
        pass # We will maintain cross in the search loops
        
    def run_local_search(labels, max_rounds=1000):
        labels = labels.copy()
        # Initialize cross array
        cross = np.zeros(n, dtype=int)
        for v in range(n):
            # count neighbors with different label
            # vectorized for init
            cross[v] = np.sum(adj_np[v] * (labels != labels[v]))
            
        best_cut = 0.5 * np.sum(cross)
        best_labels = labels.copy()
        
        for rnd in range(max_rounds):
            if time_left() < 1.0:
                break
            
            improved = False
            # Greedy pass: find vertex with max gain
            # gain[v] = degrees[v] - 2 * cross[v]
            gains = degrees - 2 * cross
            
            # To be more efficient, we can just iterate and pick best
            # But finding max is O(n). Updating is O(deg).
            # Let's do a full scan for best gain.
            max_gain = -1
            best_v = -1
            for v in range(n):
                if gains[v] > max_gain:
                    max_gain = gains[v]
                    best_v = v
            
            if max_gain <= 0:
                break
                
            # Flip best_v
            v = best_v
            labels[v] = 1 - labels[v]
            
            # Update cross for v and neighbors
            old_cross_v = cross[v]
            cross[v] = degrees[v] - old_cross_v
            
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    # u and v are now same. 
                    # Before flip, if they were same, cross[u] didn't count v.
                    # Wait, let's trace carefully.
                    # 
                    # State BEFORE flip:
                    # x_v, x_u.
                    # cross[u] counts neighbors of u with diff label.
                    # 
                    # We flip v.
                    # 
                    # Case 1: x_u == x_v (old)
                    # Before: v was NOT counted in cross[u].
                    # After: x_v_new != x_u. v IS counted in cross[u].
                    # So cross[u] += 1.
                    # 
                    # Case 2: x_u != x_v (old)
                    # Before: v WAS counted in cross[u].
                    # After: x_v_new == x_u. v IS NOT counted in cross[u].
                    # So cross[u] -= 1.
                    
                    # We are in the loop AFTER flipping labels[v].
                    # So labels[v] is the NEW label.
                    # If labels[u] == labels[v] (NEW), it means they are SAME now.
                    # This implies they were DIFFERENT before.
                    # So cross[u] should DECREASE by 1.
                    cross[u] -= 1
                else:
                    # labels[u] != labels[v] (NEW) -> They are DIFFERENT now.
                    # This implies they were SAME before.
                    # So cross[u] should INCREASE by 1.
                    cross[u] += 1
            
            current_cut = 0.5 * np.sum(cross)
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
            else:
                # If no improvement in cut size (shouldn't happen if max_gain > 0 
                # because gain = delta cut * 2? 
                # delta cut = (deg - 2*cross)/2 = gain/2.
                # If gain > 0, cut increases.
                pass
                
        return best_labels, best_cut

    def run_simulated_annealing(labels, max_iters=50000, T_start=10.0, T_end=0.1, alpha=0.99995):
        labels = labels.copy()
        cross = np.zeros(n, dtype=int)
        for v in range(n):
            cross[v] = np.sum(adj_np[v] * (labels != labels[v]))
            
        current_cut = 0.5 * np.sum(cross)
        best_cut = current_cut
        best_labels = labels.copy()
        
        T = T_start
        for it in range(max_iters):
            if time_left() < 2.0:
                break
                
            v = random.randint(0, n-1)
            
            # Gain = degrees[v] - 2 * cross[v]
            gain = degrees[v] - 2 * cross[v]
            
            # delta cut = gain / 2
            # Accept if gain > 0 or random < exp(gain / (2*T)) ?
            # Standard SA: accept if delta > 0 or rand < exp(delta/T)
            # Here delta is change in cut size.
            delta = gain / 2.0
            
            if delta > 0 or (T > 0 and random.random() < np.exp(delta / T)):
                # Flip v
                labels[v] = 1 - labels[v]
                
                # Update cross
                old_cross_v = cross[v]
                cross[v] = degrees[v] - old_cross_v
                
                for u in neighbors[v]:
                    if labels[u] == labels[v]:
                        cross[u] -= 1
                    else:
                        cross[u] += 1
                        
                current_cut += delta
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
                    
            T = max(T * alpha, T_end)
            
        return best_labels, best_cut

    # Strategy
    best_labels = None
    best_cut = -1
    
    # 1. Alternating start
    labels_alt = np.arange(n) % 2
    l, c = run_local_search(labels_alt)
    if c > best_cut:
        best_cut = c
        best_labels = l
        
    # 2. Random starts with Local Search
    num_restarts = 100 # Can increase if fast
    for _ in range(num_restarts):
        if time_left() < 2.0:
            break
        labels_rand = np.random.randint(0, 2, size=n)
        l, c = run_local_search(labels_rand, max_rounds=100)
        if c > best_cut:
            best_cut = c
            best_labels = l
            
    # 3. Simulated Annealing from best
    if time_left() > 10.0:
        l, c = run_simulated_annealing(best_labels)
        if c > best_cut:
            best_cut = c
            best_labels = l
            
    # 4. More SA restarts
    if time_left() > 5.0:
        for _ in range(20):
            if time_left() < 2.0:
                break
            labels_rand = np.random.randint(0, 2, size=n)
            l, c = run_simulated_annealing(labels_rand, max_iters=10000, T_start=5.0, T_end=0.1)
            if c > best_cut:
                best_cut = c
                best_labels = l
                
    # 5. Final polish
    if time_left() > 2.0:
        l, c = run_local_search(best_labels, max_rounds=500)
        if c > best_cut:
            best_cut = c
            best_labels = l
            
    return best_labels.astype(int)
