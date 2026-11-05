"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


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
        """Compute cut size from 0/1 labels."""
        x = 2 * labels.astype(np.float64) - 1
        xTadjx = x @ (adj @ x)
        return E - 0.5 * xTadjx

    def fast_cut_delta(labels, v):
        """Compute change in cut size if we flip vertex v."""
        same_side = adj[v][labels == labels[v]].sum()
        opp_side = adj[v][labels != labels[v]].sum()
        return same_side - opp_side

    def pairwise_swap_delta(labels, u, v):
        """Compute change in cut size if we swap labels of vertices u and v."""
        if labels[u] == labels[v]:
            return 0  # Swapping identical labels changes nothing
        
        # Current: u is on side labels[u], v is on side labels[v]
        # After swap: u is on side labels[v], v is on side labels[u]
        
        # Edges affected: those incident to u or v (but not both, since they're on different sides)
        # For edges (u, w) where w != v:
        #   If w was on same side as u, after swap w is on opposite side -> contributes +1
        #   If w was on opposite side from u, after swap w is on same side -> contributes -1
        # For edges (v, w) where w != u:
        #   If w was on same side as v, after swap w is on opposite side -> contributes +1
        #   If w was on opposite side from v, after swap w is on same side -> contributes -1
        # Edge (u,v): was crossing (different sides), after swap still crossing -> no change
        
        lu = labels[u]
        lv = labels[v]
        
        # Vertices on same side as u (excluding u itself)
        same_as_u = (labels == lu)
        same_as_u[u] = False  # exclude u
        same_as_u[v] = False  # exclude v (v is on opposite side)
        
        # Vertices on same side as v (excluding v itself)
        same_as_v = (labels == lv)
        same_as_v[v] = False
        same_as_u[u] = False  # already done
        same_as_v[u] = False  # exclude u
        
        # For vertex u: neighbors on same side as u (excluding v) will become opposite
        u_gain = adj[u, same_as_u].sum()  # these edges become crossing (+1 each)
        u_loss = adj[u, ~same_as_u & ~same_as_v].sum()  # wait, need to be more careful
        
        # Let me recompute more carefully
        # After swap, u moves to side lv, v moves to side lu
        # For each neighbor w of u (w != v):
        #   Before: if labels[w] == lu, edge (u,w) not in cut; else in cut
        #   After: if labels[w] == lv, edge (u,w) not in cut; else in cut
        #   Delta for edge (u,w): +1 if labels[w]==lu (was not cut, now cut), -1 if labels[w]!=lu and labels[w]!=lv... 
        #   Actually since labels are binary: labels[w] is either lu or lv
        #   If labels[w] == lu: before not cut, after cut (since u is now on lv) -> +1
        #   If labels[w] == lv: before cut, after not cut -> -1
        #   But w==v is excluded, and v has label lv, so we don't count edge (u,v)
        
        # For each neighbor w of v (w != u):
        #   Before: if labels[w] == lv, edge (v,w) not in cut; else in cut
        #   After: if labels[w] == lu, edge (v,w) not in cut; else in cut
        #   If labels[w] == lv: before not cut, after cut -> +1
        #   If labels[w] == lu: before cut, after not cut -> -1
        
        # So:
        # u_delta = (neighbors of u on side lu, excluding v) - (neighbors of u on side lv, excluding v)
        # v_delta = (neighbors of v on side lv, excluding u) - (neighbors of v on side lu, excluding u)
        
        # Neighbors of u on side lu (excluding u and v)
        mask_u_lu = (labels == lu)
        mask_u_lu[u] = False
        mask_u_lu[v] = False
        
        # Neighbors of u on side lv (excluding u and v)
        mask_u_lv = (labels == lv)
        mask_u_lv[u] = False
        mask_u_lv[v] = False
        
        u_delta = adj[u, mask_u_lu].sum() - adj[u, mask_u_lv].sum()
        
        # Neighbors of v on side lv (excluding u and v)
        mask_v_lv = (labels == lv)
        mask_v_lv[v] = False
        mask_v_lv[u] = False
        
        # Neighbors of v on side lu (excluding u and v)
        mask_v_lu = (labels == lu)
        mask_v_lu[v] = False
        mask_v_lu[u] = False
        
        v_delta = adj[v, mask_v_lv].sum() - adj[v, mask_v_lu].sum()
        
        return u_delta + v_delta

    def local_search_with_swaps(labels, max_rounds=30, swap_prob=0.1):
        """Hill-climbing local search with occasional pairwise swap moves."""
        best_labels = labels.copy()
        best_cut = cut_size_from_labels(best_labels)
        current_cut = best_cut
        
        for rnd in range(max_rounds):
            if time_left() < 2.0:
                break
            improved = False
            
            # Single-flip moves
            for v in range(n):
                if time_left() < 1.0:
                    break
                delta = fast_cut_delta(labels, v)
                if delta > 0:
                    labels[v] = 1 - labels[v]
                    current_cut += delta
                    improved = True
            
            # Pairwise swap moves (try a few random pairs)
            num_swaps = max(1, n // 10)
            for _ in range(num_swaps):
                if time_left() < 1.0:
                    break
                u, v = np.random.choice(n, size=2, replace=False)
                if labels[u] != labels[v]:
                    delta = pairwise_swap_delta(labels, u, v)
                    if delta > 0:
                        labels[u], labels[v] = labels[v], labels[u]
                        current_cut += delta
                        improved = True
            
            if not improved:
                break
        
        final_cut = cut_size_from_labels(labels)
        if final_cut > best_cut:
            best_labels = labels.copy()
            best_cut = final_cut
        return best_labels, best_cut

    def spectral_partition(adj, degrees):
        """Compute spectral partition using Fiedler vector of graph Laplacian."""
        n = adj.shape[0]
        if n < 3:
            return np.zeros(n, dtype=int)
        
        # Build Laplacian L = D - A
        L = np.diag(degrees.astype(np.float64)) - adj.astype(np.float64)
        
        # Compute eigenvalues and eigenvectors
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
        except np.linalg.LinAlgError:
            return np.zeros(n, dtype=int)
        
        # Fiedler vector is the second eigenvector (index 1)
        # The first eigenvector (index 0) corresponds to eigenvalue 0
        fiedler = eigvecs[:, 1]
        
        # Partition: vertices with positive Fiedler values go to side 1, negative to side 0
        labels = (fiedler > 0).astype(int)
        
        return labels

    def goe_heuristic(labels):
        """Greedy Ordering Elimination heuristic for Max Cut."""
        unassigned = list(range(n))
        result = labels.copy()
        assigned = np.zeros(n, dtype=bool)
        
        for _ in range(n):
            if time_left() < 2.0:
                break
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
        """Simulated annealing for maximum cut."""
        best_labels = labels.copy()
        best_cut = cut_size_from_labels(best_labels)
        current_cut = best_cut
        T = T_start

        for it in range(max_iters):
            if time_left() < 3.0:
                break
            
            v = np.random.randint(0, n)
            delta = fast_cut_delta(labels, v)

            if delta >= 0:
                labels[v] = 1 - labels[v]
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            else:
                if T > 0.001:
                    if delta > -100 * T:
                        if np.random.random() < np.exp(delta / T):
                            labels[v] = 1 - labels[v]
                            current_cut += delta

            T = max(T * alpha, T_end)

        return best_labels, best_cut

    def multi_start_local(num_starts, use_swaps=True):
        """Try multiple random starts with local search."""
        best_labels = None
        best_cut = -1
        for s in range(num_starts):
            if time_left() < 3.0:
                break
            labels = np.random.randint(0, 2, size=n)
            if use_swaps:
                labels, cut = local_search_with_swaps(labels, max_rounds=25)
            else:
                labels, cut = local_search_with_swaps(labels, max_rounds=25, swap_prob=0.0)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
        return best_labels, best_cut

    # Strategy:
    # 1. Spectral partition (Fiedler vector) as primary initialization
    # 2. Refine with local search including pairwise swaps
    # 3. Try alternating pattern
    # 4. Try GOE heuristic
    # 5. Multiple random restarts with local search + swaps
    # 6. Simulated annealing if time permits
    # 7. Final polish with swaps

    # Step 1: Spectral partition
    labels_spec = spectral_partition(adj, degrees)
    best_labels, best_cut = local_search_with_swaps(labels_spec, max_rounds=40)

    # Step 2: Alternating start
    if time_left() > 15.0:
        labels_alt = np.arange(n) % 2
        labels_alt_ls, cut_alt = local_search_with_swaps(labels_alt, max_rounds=40)
        if cut_alt > best_cut:
            best_cut = cut_alt
            best_labels = labels_alt_ls

    # Step 3: GOE heuristic
    if time_left() > 15.0:
        labels_goe = goe_heuristic(np.zeros(n, dtype=int))
        labels_goe_ls, cut_goe = local_search_with_swaps(labels_goe, max_rounds=40)
        if cut_goe > best_cut:
            best_cut = cut_goe
            best_labels = labels_goe_ls

    # Step 4: Random restarts with local search + swaps
    if time_left() > 15.0:
        labels_rs, cut_rs = multi_start_local(12, use_swaps=True)
        if cut_rs > best_cut:
            best_cut = cut_rs
            best_labels = labels_rs

    # Step 5: Simulated annealing from best solution
    if time_left() > 20.0:
        labels_sa, cut_sa = simulated_annealing(best_labels, max_iters=100000, T_start=8.0, T_end=0.01)
        if cut_sa > best_cut:
            best_cut = cut_sa
            best_labels = labels_sa

        # Try SA from spectral start
        if time_left() > 10.0:
            labels_spec2 = spectral_partition(adj, degrees)
            labels_sa2, cut_sa2 = simulated_annealing(labels_spec2, max_iters=50000, T_start=10.0, T_end=0.01)
            if cut_sa2 > best_cut:
                best_cut = cut_sa2
                best_labels = labels_sa2

    # Step 6: Final polish with swaps
    if time_left() > 3.0:
        for _ in range(3):
            if time_left() < 1.5:
                break
            labels_polish, cut_polish = local_search_with_swaps(best_labels, max_rounds=20)
            if cut_polish > best_cut:
                best_cut = cut_polish
                best_labels = labels_polish
            else:
                break

    return best_labels.astype(int)
# EVOLVE-BLOCK-END
