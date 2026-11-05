"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    # Convert to numpy int matrix for fast vectorized ops
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    # Deadline guard: kill after ~145s to be safe under 150s limit
    start_time = time.time()
    DEADLINE = start_time + 145.0

    def time_left():
        return DEADLINE - time.time()

    E = int(np.sum(adj)) // 2

    # Precompute degrees
    degrees = adj.sum(axis=1)

    def cut_size_from_labels(labels):
        """Compute cut size from 0/1 labels using vectorized x^T A x."""
        x = 2 * labels.astype(np.float64) - 1
        xTadjx = x @ (adj @ x)
        return int(E - 0.5 * xTadjx)

    def local_search(labels, max_rounds=100):
        """Fast hill-climbing with incremental gain updates."""
        best_labels = labels.copy()
        best_cut = cut_size_from_labels(best_labels)
        current_cut = best_cut
        labels = labels.copy()
        x = 2 * labels.astype(np.float64) - 1
        gains = x * (adj @ x)
        
        for rnd in range(max_rounds):
            if time_left() < 1.0:
                break
            pos_gains = np.where(gains > 0)[0]
            if len(pos_gains) == 0:
                break
            order = pos_gains[np.argsort(-gains[pos_gains])]
            improved = False
            
            for v in order:
                if time_left() < 0.5:
                    break
                if gains[v] <= 0:
                    continue
                # Flip v
                labels[v] = 1 - labels[v]
                current_cut += int(gains[v])
                # Update x and gains incrementally
                old_xv = x[v]
                x[v] = -old_xv
                # For each neighbor u of v, gain[u] changes by:
                # delta_gain[u] = x[u] * adj[u,v] * (new_x[v] - old_x[v]) = x[u] * adj[u,v] * (-2*old_x[v])
                neighbors = np.where(adj[v] == 1)[0]
                if len(neighbors) > 0:
                    gains[neighbors] += x[neighbors] * adj[v, neighbors] * (-2.0 * old_xv)
                # v's own gain becomes 0 (or we can recompute)
                gains[v] = 0.0
                improved = True
            
            if not improved:
                break
        
        final_cut = cut_size_from_labels(labels)
        if final_cut > best_cut:
            best_labels = labels.copy()
            best_cut = final_cut
        return best_labels, best_cut

    def goe_heuristic(labels):
        """Vectorized Greedy Ordering Elimination heuristic for Max Cut."""
        result = labels.copy()
        assigned = np.zeros(n, dtype=bool)
        
        for _ in range(n):
            if time_left() < 1.0:
                break
            # Find the vertex with highest degree among unassigned
            unassigned_mask = ~assigned
            best_v = -1
            best_deg = -1
            for v in np.where(unassigned_mask)[0]:
                deg = degrees[v]
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                break
            
            # Assign best_v to the side that gives better cut
            neighbors = np.where(adj[best_v] == 1)[0]
            unassigned_neighbors = neighbors[unassigned_mask[neighbors]] if len(neighbors) > 0 else np.array([], dtype=int)
            count_side0 = np.sum(result[unassigned_neighbors] == 0) if len(unassigned_neighbors) > 0 else 0
            count_side1 = np.sum(result[unassigned_neighbors] == 1) if len(unassigned_neighbors) > 0 else 0
            
            if count_side1 >= count_side0:
                result[best_v] = 0
            else:
                result[best_v] = 1
            
            assigned[best_v] = True
        
        return result

    def simulated_annealing(labels, max_iters=200000, T_start=8.0, T_end=0.01, alpha=0.99995):
        """Simulated annealing with incremental gain updates."""
        best_labels = labels.copy()
        best_cut = cut_size_from_labels(best_labels)
        current_cut = best_cut
        T = T_start
        labels = labels.copy()
        x = 2 * labels.astype(np.float64) - 1
        gains = x * (adj @ x)
        
        for it in range(max_iters):
            if time_left() < 2.0:
                break
            
            v = np.random.randint(0, n)
            delta = int(gains[v])

            if delta >= 0:
                # Flip v
                labels[v] = 1 - labels[v]
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
                # Update gains incrementally
                old_xv = x[v]
                x[v] = -old_xv
                neighbors = np.where(adj[v] == 1)[0]
                if len(neighbors) > 0:
                    gains[neighbors] += x[neighbors] * adj[v, neighbors] * (-2.0 * old_xv)
                gains[v] = 0.0
            else:
                # Accept with probability exp(delta / T)
                if T > 0.001:
                    if delta > -100 * T:
                        if np.random.random() < np.exp(delta / T):
                            labels[v] = 1 - labels[v]
                            current_cut += delta
                            old_xv = x[v]
                            x[v] = -old_xv
                            neighbors = np.where(adj[v] == 1)[0]
                            if len(neighbors) > 0:
                                gains[neighbors] += x[neighbors] * adj[v, neighbors] * (-2.0 * old_xv)
                            gains[v] = 0.0

            T = max(T * alpha, T_end)

        return best_labels, best_cut

    def multi_start_local(num_starts):
        """Try multiple random starts with local search."""
        best_labels = None
        best_cut = -1
        for s in range(num_starts):
            if time_left() < 2.0:
                break
            labels = np.random.randint(0, 2, size=n)
            labels, cut = local_search(labels, max_rounds=50)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
        return best_labels, best_cut

    # Strategy:
    # 1. Start with alternating pattern, run local search
    # 2. Try GOE heuristic
    # 3. Try multiple random restarts with local search
    # 4. If time permits, run simulated annealing from best solution
    # 5. Final polish with local search

    # Step 1: Alternating start
    labels_alt = np.arange(n) % 2
    best_labels, best_cut = local_search(labels_alt, max_rounds=100)

    # Step 2: GOE heuristic
    labels_goe = goe_heuristic(np.zeros(n, dtype=int))
    labels_goe_ls, cut_goe = local_search(labels_goe, max_rounds=100)
    if cut_goe > best_cut:
        best_cut = cut_goe
        best_labels = labels_goe_ls

    # Step 3: Random restarts with local search
    if time_left() > 8.0:
        labels_rs, cut_rs = multi_start_local(25)
        if cut_rs > best_cut:
            best_cut = cut_rs
            best_labels = labels_rs

    # Step 4: Simulated annealing from best solution
    if time_left() > 15.0:
        labels_sa, cut_sa = simulated_annealing(best_labels, max_iters=200000, T_start=8.0, T_end=0.01)
        if cut_sa > best_cut:
            best_cut = cut_sa
            best_labels = labels_sa

        # Try SA from a random start
        if time_left() > 8.0:
            labels_rand = np.random.randint(0, 2, size=n)
            labels_sa2, cut_sa2 = simulated_annealing(labels_rand, max_iters=100000, T_start=10.0, T_end=0.01)
            if cut_sa2 > best_cut:
                best_cut = cut_sa2
                best_labels = labels_sa2

    # Step 5: Final polish with local search
    if time_left() > 2.0:
        labels_polish, cut_polish = local_search(best_labels, max_rounds=100)
        if cut_polish > best_cut:
            best_cut = cut_polish
            best_labels = labels_polish

    return best_labels.astype(int)
# EVOLVE-BLOCK-END
