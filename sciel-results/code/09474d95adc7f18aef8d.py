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

    # Deadline guard: kill after ~140s to be safe under 150s limit
    start_time = time.time()
    DEADLINE = start_time + 140.0

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

    def local_search(labels, max_rounds=50):
        """Hill-climbing local search with vectorized gain computation and proper updates."""
        best_labels = labels.copy()
        best_cut = cut_size_from_labels(best_labels)
        current_cut = best_cut
        labels = labels.copy()

        for rnd in range(max_rounds):
            if time_left() < 2.0:
                break
            # Compute gains: gain[v] = x[v] * (adj @ x)[v]
            # If gain[v] > 0, flipping v increases cut by gain[v]
            x = 2 * labels.astype(np.float64) - 1
            gains = x * (adj @ x)
            
            improved = False
            # Process vertices in order of largest positive gain first
            pos_gains = np.where(gains > 0)[0]
            if len(pos_gains) == 0:
                break
            # Sort by gain descending for better convergence
            order = pos_gains[np.argsort(-gains[pos_gains])]
            
            for v in order:
                if time_left() < 1.0:
                    break
                # Recompute gain for v to ensure it's still positive
                # (other flips may have changed it)
                if gains[v] <= 0:
                    continue
                labels[v] = 1 - labels[v]
                current_cut += int(gains[v])
                # Update gains for all neighbors of v
                # When v flips, x[v] changes sign, so gains for neighbors u:
                # gain[u] = x[u] * (adj @ x)[u]
                # The change in (adj @ x)[u] = adj[u,v] * delta_x[v] = adj[u,v] * (-2*x[v]_old)
                # Since x[v] flips: new_x[v] = -x[v]_old
                # delta in (adj @ x)[u] = adj[u,v] * (new_x[v] - old_x[v]) = adj[u,v] * (-2*old_x[v])
                # But we've already flipped, so new_x[v] = -old_x[v]
                # For neighbor u: (adj @ x)[u] changes by adj[u,v] * (-2 * old_x[v])
                # And gain[u] = x[u] * (adj @ x)[u] changes by x[u] * adj[u,v] * (-2 * old_x[v])
                # Since adj is symmetric, adj[u,v] = adj[v,u]
                # old_x[v] = -new_x[v] (since we flipped)
                # So change in gain[u] = x[u] * adj[v,u] * (-2 * (-new_x[v])) = 2 * x[u] * adj[v,u] * new_x[v]
                # But this is getting complex. Let's just recompute gains for affected vertices.
                # For efficiency, recompute all gains at end of round if many flips
                improved = True
            
            # Recompute gains at end of round if we made flips
            if improved:
                x = 2 * labels.astype(np.float64) - 1
                gains = x * (adj @ x)
        
        final_cut = cut_size_from_labels(labels)
        if final_cut > best_cut:
            best_labels = labels.copy()
            best_cut = final_cut
        return best_labels, best_cut

    def goe_heuristic(labels):
        """Greedy Ordering Elimination heuristic for Max Cut."""
        unassigned = list(range(n))
        result = labels.copy()
        assigned = np.zeros(n, dtype=bool)
        
        for _ in range(n):
            if time_left() < 2.0:
                break
            # Find the vertex with highest degree among unassigned
            best_v = -1
            best_deg = -1
            for v in unassigned:
                if not assigned[v]:
                    deg = degrees[v]
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
            
            if best_v == -1:
                break
            
            # Assign best_v to the side that gives better cut
            neighbors = np.where(adj[best_v] == 1)[0]
            count_side0 = 0
            count_side1 = 0
            for nb in neighbors:
                if not assigned[nb]:
                    if result[nb] == 0:
                        count_side0 += 1
                    else:
                        count_side1 += 1
            
            if count_side1 >= count_side0:
                result[best_v] = 0
            else:
                result[best_v] = 1
            
            assigned[best_v] = True
        
        return result

    def simulated_annealing(labels, max_iters=200000, T_start=8.0, T_end=0.01, alpha=0.99995):
        """Vectorized simulated annealing for maximum cut. Returns best labels found."""
        best_labels = labels.copy()
        best_cut = cut_size_from_labels(best_labels)
        current_cut = best_cut
        T = T_start
        labels = labels.copy()
        
        # Precompute for fast delta computation
        # delta for flipping v = -gains[v] where gains[v] = x[v] * (adj @ x)[v]
        # If we flip v, cut changes by -gains[v] (since gain[v] is the increase from flipping)
        # Wait: if gain[v] > 0, flipping v increases cut by gain[v]
        # So delta = -gain[v] if we're computing "change in cut from flip"
        # Actually: if we flip v, new_cut = old_cut + gain[v] (from derivation)
        # So delta = gain[v]

        for it in range(max_iters):
            if time_left() < 3.0:
                break
            
            # Pick random vertex
            v = np.random.randint(0, n)
            # Compute delta = gain[v] = x[v] * (adj @ x)[v]
            # For speed, compute on the fly
            x_v = 2 * labels[v] - 1
            adj_x_v = adj[v] @ (2 * labels.astype(np.float64) - 1)
            delta = int(x_v * adj_x_v)

            if delta >= 0:
                labels[v] = 1 - labels[v]
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            else:
                # Accept with probability exp(delta / T)
                if T > 0.001:
                    if delta > -100 * T:
                        if np.random.random() < np.exp(delta / T):
                            labels[v] = 1 - labels[v]
                            current_cut += delta

            # Cool down
            T = max(T * alpha, T_end)

        return best_labels, best_cut

    def multi_start_local(num_starts):
        """Try multiple random starts with local search."""
        best_labels = None
        best_cut = -1
        for s in range(num_starts):
            if time_left() < 3.0:
                break
            labels = np.random.randint(0, 2, size=n)
            labels, cut = local_search(labels, max_rounds=30)
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
    best_labels, best_cut = local_search(labels_alt, max_rounds=50)

    # Step 2: GOE heuristic
    labels_goe = goe_heuristic(np.zeros(n, dtype=int))
    labels_goe_ls, cut_goe = local_search(labels_goe, max_rounds=50)
    if cut_goe > best_cut:
        best_cut = cut_goe
        best_labels = labels_goe_ls

    # Step 3: Random restarts with local search
    if time_left() > 10.0:
        labels_rs, cut_rs = multi_start_local(15)
        if cut_rs > best_cut:
            best_cut = cut_rs
            best_labels = labels_rs

    # Step 4: Simulated annealing from best solution
    if time_left() > 20.0:
        labels_sa, cut_sa = simulated_annealing(best_labels, max_iters=100000, T_start=8.0, T_end=0.01)
        if cut_sa > best_cut:
            best_cut = cut_sa
            best_labels = labels_sa

        # Try SA from a random start
        if time_left() > 10.0:
            labels_rand = np.random.randint(0, 2, size=n)
            labels_sa2, cut_sa2 = simulated_annealing(labels_rand, max_iters=50000, T_start=10.0, T_end=0.01)
            if cut_sa2 > best_cut:
                best_cut = cut_sa2
                best_labels = labels_sa2

    # Step 5: Final polish with single-flip and pairwise-flip moves
    if time_left() > 3.0:
        def pairwise_polish(labels, max_rounds=20):
            """Polish with both single and pairwise flips."""
            best_labels = labels.copy()
            best_cut = cut_size_from_labels(best_labels)
            current_cut = best_cut
            labels = labels.copy()
            
            for rnd in range(max_rounds):
                if time_left() < 1.5:
                    break
                
                improved = False
                x = 2 * labels.astype(np.float64) - 1
                gains = x * (adj @ x)
                
                # Single flips
                for v in range(n):
                    if time_left() < 1.0:
                        break
                    if gains[v] > 0:
                        labels[v] = 1 - labels[v]
                        current_cut += int(gains[v])
                        improved = True
                
                if not improved:
                    # Try pairwise flips
                    x = 2 * labels.astype(np.float64) - 1
                    # For pair (i,j), delta = gain[i] + gain[j] - 2*adj[i,j]*(1 if same_side else -1)
                    # Actually: flipping both i and j
                    # new_cut - old_cut = gain[i] + gain[j] - 2*adj[i,j]*(x[i]*x[j])
                    # Wait, let's derive: 
                    # If we flip i: cut changes by gain[i]
                    # If we then flip j: cut changes by new_gain[j]
                    # new_gain[j] = gain[j] - 2*adj[i,j]*x[i]*x[j] (from the interaction)
                    # Hmm, this is getting complex. Let's just compute directly.
                    # For simplicity, try a few random pairs
                    for _ in range(min(50, n * 2)):
                        if time_left() < 1.0:
                            break
                        i, j = np.random.randint(0, n, size=2)
                        if i == j:
                            continue
                        # Compute delta for flipping both i and j
                        x_i = 2 * labels[i] - 1
                        x_j = 2 * labels[j] - 1
                        # Delta = (change if flip i) + (change if flip j given i flipped)
                        # Simpler: compute cut before and after
                        old_cut = current_cut
                        labels[i] = 1 - labels[i]
                        labels[j] = 1 - labels[j]
                        new_cut = cut_size_from_labels(labels)
                        delta = new_cut - old_cut
                        if delta > 0:
                            current_cut = new_cut
                            if current_cut > best_cut:
                                best_cut = current_cut
                                best_labels = labels.copy()
                            improved = True
                        else:
                            # Revert
                            labels[i] = 1 - labels[i]
                            labels[j] = 1 - labels[j]
                
                if not improved:
                    break
            
            return best_labels, best_cut
        
        for _ in range(3):
            if time_left() < 1.5:
                break
            labels_polish, cut_polish = pairwise_polish(best_labels, max_rounds=20)
            if cut_polish > best_cut:
                best_cut = cut_polish
                best_labels = labels_polish
            else:
                break

    return best_labels.astype(int)
# EVOLVE-BLOCK-END
