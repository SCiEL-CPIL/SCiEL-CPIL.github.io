"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def partition(adjacency):
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Ensure symmetric
    adj = np.maximum(adj, adj.T)
    # Remove self-loops if any
    np.fill_diagonal(adj, 0)
    
    # Precompute neighbors
    neighbors = []
    for i in range(n):
        nbrs = np.nonzero(adj[i])[0]
        neighbors.append(nbrs)
        
    def get_initial_gain(labels):
        gain = np.zeros(n, dtype=np.int32)
        for v in range(n):
            # gain[v] = sum_{u in nbrs(v)} (1 if labels[u]==labels[v] else -1)
            same_count = 0
            diff_count = 0
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count += 1
                else:
                    diff_count += 1
            gain[v] = same_count - diff_count
        return gain

    def compute_cut(labels):
        mask = (labels[:, None] != labels[None, :])
        return np.triu(adj * mask, k=1).sum()

    def run_sa(seed):
        rng = np.random.default_rng(seed)
        # Random initialization
        labels = rng.integers(0, 2, n)
        gain = get_initial_gain(labels)
        
        current_cut = compute_cut(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        # SA Parameters
        iterations = 30000
        T_start = 10.0
        T_end = 0.01
        cool_factor = (T_end / T_start) ** (1.0 / iterations)
        
        for i in range(iterations):
            v = rng.integers(0, n)
            
            delta = gain[v]
            
            if delta > 0:
                pass # Accept
            else:
                T = T_start * (cool_factor ** i)
                if rng.random() >= np.exp(delta / T):
                    continue # Reject
            
            # Flip v
            old_label = labels[v]
            new_label = 1 - old_label
            labels[v] = new_label
            current_cut += delta
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
                
            # Update gains for neighbors of v
            for u in neighbors[v]:
                if labels[u] == new_label:
                    # x_v_new == x_u => x_v_old != x_u => gain[u] increases by 2
                    gain[u] += 2
                else:
                    # x_v_new != x_u => x_v_old == x_u => gain[u] decreases by 2
                    gain[u] -= 2
                    
        # Final local search
        improved = True
        while improved:
            improved = False
            for v in range(n):
                if gain[v] > 0:
                    old_label = labels[v]
                    new_label = 1 - old_label
                    labels[v] = new_label
                    current_cut += gain[v]
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_labels = labels.copy()
                    improved = True
                    for u in neighbors[v]:
                        if labels[u] == new_label:
                            gain[u] += 2
                        else:
                            gain[u] -= 2
                            
        return best_labels, best_cut

    # Multi-start
    best_labels_overall = None
    best_cut_overall = 0
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer for overhead
    
    seed_counter = 0
    # Ensure at least one run
    while time.time() - start_time < time_limit:
        labels, cut = run_sa(seed_counter)
        if cut > best_cut_overall:
            best_cut_overall = cut
            best_labels_overall = labels
        seed_counter += 1
        # Optional: break if we've run enough or time is tight
        if seed_counter > 100: # Safety break
             break

    if best_labels_overall is None:
        # Fallback to alternating
        return np.arange(n) % 2
        
    return best_labels_overall
# EVOLVE-BLOCK-END
