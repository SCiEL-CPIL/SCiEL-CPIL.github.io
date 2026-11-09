"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    # Simulated Annealing with greedy local improvement for Max-Cut
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Deadline guard: 150s wall-clock limit, return best with safety margin
    deadline = time.time() + 140.0
    
    # Precompute neighbor lists for fast per-vertex queries
    neighbors = [np.where(adj[v] > 0)[0] for v in range(n)]
    neighbor_arrays = [adj[v][neighbors[v]] for v in range(n)]
    
    def cut_size(labels):
        """Compute cut size using precomputed neighbor info."""
        total = 0
        for v in range(n):
            total += np.sum(neighbor_arrays[v][labels[neighbors[v]] != labels[v]])
        return total // 2
    
    def local_improve(labels, max_passes=100):
        """Greedy local improvement with incremental tracking."""
        labels = labels.copy().astype(np.int8)
        for _ in range(max_passes):
            improved = False
            for v in range(n):
                cur_cross = np.sum(neighbor_arrays[v][labels[neighbors[v]] != labels[v]])
                new_cross = np.sum(neighbor_arrays[v][labels[neighbors[v]] == labels[v]])
                if new_cross > cur_cross:
                    labels[v] = 1 - labels[v]
                    improved = True
            if not improved:
                break
        return labels
    
    def sa_run(labels, cur_cut, n_steps, T_start, T_end, rng):
        """Run simulated annealing with incremental delta tracking."""
        sa_labels = labels.copy().astype(np.int8)
        sa_cut = cur_cut
        cooling_rate = (T_end / T_start) ** (1.0 / n_steps)
        T = T_start
        
        # Precompute current cross counts for each vertex
        cross_counts = np.zeros(n, dtype=np.int32)
        for v in range(n):
            cross_counts[v] = np.sum(neighbor_arrays[v][sa_labels[neighbors[v]] != sa_labels[v]])
        
        for step in range(n_steps):
            if time.time() > deadline:
                break
            
            v = rng.integers(0, n)
            cur_cross_v = cross_counts[v]
            # When we flip v: edges that were crossing become non-crossing and vice versa
            # delta = (old_same) - (old_cross) = (deg_v - cross_v) - cross_v = deg_v - 2*cross_v
            deg_v = len(neighbors[v])
            delta = deg_v - 2 * cur_cross_v
            
            if delta > 0 or (T > 0.01 and rng.random() < np.exp(delta / T)):
                sa_labels[v] = 1 - sa_labels[v]
                sa_cut += delta
                # Update cross counts for v and its neighbors
                cross_counts[v] = deg_v - cur_cross_v
                for u in neighbors[v]:
                    # Edge (u,v) flips: if it was crossing, now it's not, and vice versa
                    if sa_labels[u] != sa_labels[v]:
                        # After flip, u and v are same, so edge was crossing before flip
                        cross_counts[u] -= 1
                    else:
                        # After flip, u and v differ, so edge was not crossing before
                        cross_counts[u] += 1
            
            if sa_cut > best_cut[0]:
                best_cut[0] = sa_cut
                best_labels[0] = sa_labels.copy()
            
            T *= cooling_rate
        
        return sa_labels, sa_cut
    
    # Multiple restarts with simulated annealing
    best_labels = [None]
    best_cut = [-1]
    
    # Initial solutions: alternating, random, and spectral-inspired
    initial_solutions = []
    
    # Alternating
    initial_solutions.append(np.arange(n) % 2)
    
    # Random
    rng = np.random.default_rng(42)
    initial_solutions.append(rng.integers(0, 2, n).astype(np.int8))
    
    # Another random
    initial_solutions.append(rng.integers(0, 2, n).astype(np.int8))
    
    # Spectral-like: use degree-based initialization
    degrees = adj.sum(axis=1)
    initial_solutions.append((degrees > degrees.mean()).astype(np.int8))
    
    # Add a few more diverse random starts
    for _ in range(2):
        initial_solutions.append(rng.integers(0, 2, n).astype(np.int8))
    
    for init_idx, init_labels in enumerate(initial_solutions):
        if time.time() > deadline:
            break
        
        labels = init_labels.copy().astype(np.int8)
        
        # Local improvement first
        labels = local_improve(labels)
        cur_cut = cut_size(labels)
        
        if cur_cut > best_cut[0]:
            best_cut[0] = cur_cut
            best_labels[0] = labels.copy()
        
        # Simulated Annealing - use more steps since each is now O(deg) not O(n)
        elapsed = time.time() - (deadline - 140.0)
        remaining = deadline - time.time()
        # Budget per restart: split remaining time
        n_restarts_left = len(initial_solutions) - init_idx
        time_per_restart = remaining / max(n_restarts_left, 1)
        
        # Estimate steps: each step is ~O(avg_degree) ≈ O(n/2) for 100-node graph
        # With 100 nodes, avg degree ~10-20, so each step is very fast
        # We can afford many more steps
        n_steps = 20000
        T_start = 5.0
        T_end = 0.01
        
        sa_labels, sa_cut = sa_run(labels, cur_cut, n_steps, T_start, T_end, rng)
        
        # Final local improvement on SA result
        if time.time() < deadline:
            polished = local_improve(sa_labels, max_passes=200)
            pol_cut = cut_size(polished)
            if pol_cut > best_cut[0]:
                best_cut[0] = pol_cut
                best_labels[0] = polished
    
    # Final high-precision polish: multiple passes on the best solution
    if best_labels[0] is not None and time.time() < deadline:
        final = best_labels[0].copy()
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for v in range(n):
                cur_cross = np.sum(neighbor_arrays[v][final[neighbors[v]] != final[v]])
                new_cross = np.sum(neighbor_arrays[v][final[neighbors[v]] == final[v]])
                if new_cross > cur_cross:
                    final[v] = 1 - final[v]
                    improved = True
            if not improved:
                break
        final_cut = cut_size(final)
        if final_cut > best_cut[0]:
            best_labels[0] = final
            best_cut[0] = final_cut
    
    # Final check
    if best_labels[0] is None:
        best_labels[0] = np.arange(n) % 2
    
    return best_labels[0]
# EVOLVE-BLOCK-END
