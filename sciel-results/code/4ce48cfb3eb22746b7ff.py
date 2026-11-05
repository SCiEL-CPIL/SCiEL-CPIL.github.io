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
        # cut = sum over edges (i,j) where labels[i] != labels[j]
        # Using: cut = (sum over i of deg[i] - sum over i of (adj[i] * (1 - 2*|labels[i]-labels[j]|)) ) / 2
        # More directly:
        # x = 2*labels - 1, cut = (n*deg_sum - x^T A x) / 4 ... 
        # Actually: cut = (sum_i deg[i] - sum_{i,j} A[i,j]*(1 if labels[i]==labels[j] else 0)) / 2
        # Let's use: x = 2*labels - 1 (values in {-1, 1})
        # x^T A x = sum_{i,j} A[i,j] x[i] x[j]
        # For i=j: A[i,i]=0 (assuming no self-loops)
        # For i!=j: x[i]x[j] = 1 if same, -1 if different
        # So x^T A x = sum_{i,j: A[i,j]=1} (1 if same else -1) = (sum_same - sum_diff)
        # And sum_same + sum_diff = 2E
        # So x^T A x = sum_same - (2E - sum_same) = 2*sum_same - 2E
        # sum_same = (x^T A x + 2E) / 2 = (x^T A x)/2 + E
        # cut = E - sum_same = E - (x^T A x)/2 - E = -x^T A x / 2
        # Wait, that doesn't seem right. Let me recompute.
        # cut = sum_diff = 2E - sum_same
        # x^T A x = 2*sum_same - 2E
        # sum_same = (x^T A x + 2E) / 2
        # cut = 2E - (x^T A x + 2E) / 2 = (4E - x^T A x - 2E) / 2 = (2E - x^T A x) / 2 = E - x^T A x / 2
        x = 2 * labels.astype(np.float64) - 1
        xTadjx = x @ (adj @ x)
        return E - 0.5 * xTadjx

    def fast_cut_delta(labels, v):
        """Compute change in cut size if we flip vertex v.
        Returns the delta (new_cut - old_cut)."""
        # delta = 2 * (sum of edges from v to opposite side) - deg[v]
        # Because: new_cut - old_cut = (edges to same) - (edges to opp)
        #          = (deg[v] - edges_to_opp) - edges_to_opp
        #          = deg[v] - 2 * edges_to_opp
        # Wait, let me recompute:
        # old cut contribution from v's edges: edges from v to opposite side
        # new cut contribution from v's edges: edges from v to same side (after flip)
        # delta = (edges to same) - (edges to opp)
        # edges to same + edges to opp = deg[v]
        # delta = (deg[v] - edges_to_opp) - edges_to_opp = deg[v] - 2*edges_to_opp
        opp_side = adj[v][labels != labels[v]].sum()
        return int(degrees[v]) - 2 * int(opp_side)

    def local_search(labels, max_rounds=50):
        """Hill-climbing local search with incremental gain updates. Returns best labels found."""
        best_labels = labels.copy()
        best_cut = cut_size_from_labels(best_labels)
        current_cut = best_cut
        
        # Precompute gains for all vertices
        # gain[v] = delta if we flip v = deg[v] - 2 * (edges from v to opposite side)
        # We can compute this vectorized:
        # For each v, edges_to_opp[v] = sum of adj[v][labels != labels[v]]
        # = sum(adj[v] * (labels != labels[v]))
        # Let's use: x = 2*labels - 1, then (labels != labels[v]) means x[j]*x[v] = -1
        # edges_to_opp[v] = sum_j adj[v,j] * (1 - x[j]*x[v]) / 2
        # = (deg[v] - x[v] * sum_j adj[v,j] * x[j]) / 2
        # = (deg[v] - x[v] * (A @ x)[v]) / 2
        # gain[v] = deg[v] - 2 * edges_to_opp[v] = deg[v] - (deg[v] - x[v]*(Ax)[v]) = x[v]*(Ax)[v]
        
        def compute_gains(labels_arr):
            x = 2 * labels_arr.astype(np.float64) - 1
            Ax = adj @ x
            return x * Ax  # This is the gain for flipping each vertex
        
        gains = compute_gains(labels)
        
        for rnd in range(max_rounds):
            if time_left() < 2.0:
                break
            improved = False
            for v in range(n):
                if time_left() < 1.0:
                    break
                if gains[v] > 0:
                    # Flip vertex v
                    labels[v] = 1 - labels[v]
                    current_cut += int(gains[v])
                    improved = True
                    
                    # Update gains for v and all its neighbors
                    # When we flip v, the gain for v itself changes sign
                    # For neighbor u, the gain changes by 2*adj[u,v] (if u is on same side as new v, etc.)
                    # Actually, let's just recompute gains for neighbors
                    neighbors = np.where(adj[v] == 1)[0]
                    # The gain for each neighbor u changes by:
                    # Before flip: gain[u] included contribution from edge (u,v)
                    # After flip: the contribution changes
                    # Simplest: recompute gains for v and all neighbors
                    affected = np.concatenate([[v], neighbors])
                    affected = np.unique(affected)
                    # Recompute gains only for affected vertices
                    x = 2 * labels.astype(np.float64) - 1
                    Ax = adj @ x
                    new_gains = x * Ax
                    gains[affected] = new_gains[affected]
            
            if not improved:
                break
        
        final_cut = cut_size_from_labels(labels)
        if final_cut > best_cut:
            best_labels = labels.copy()
            best_cut = final_cut
        return best_labels, best_cut

    def goe_heuristic(labels):
        """Greedy Ordering Elimination heuristic for Max Cut (vectorized)."""
        result = labels.copy()
        assigned = np.zeros(n, dtype=bool)
        
        # Process vertices in order of decreasing degree
        order = np.argsort(-degrees)
        
        for best_v in order:
            if time_left() < 2.0:
                break
            if assigned[best_v]:
                continue
            
            # Count neighbors on each side among unassigned vertices
            neighbors = np.where(adj[best_v] == 1)[0]
            unassigned_mask = ~assigned
            unassigned_neighbors = neighbors[unassigned_mask[neighbors]]
            
            count_side0 = np.sum(result[unassigned_neighbors] == 0)
            count_side1 = np.sum(result[unassigned_neighbors] == 1)
            
            # If we put best_v in side 0, cut gets count_side1 (edges to side 1)
            # If we put best_v in side 1, cut gets count_side0 (edges to side 0)
            if count_side1 >= count_side0:
                result[best_v] = 0
            else:
                result[best_v] = 1
            
            assigned[best_v] = True
        
        return result

    def simulated_annealing(labels, max_iters=200000, T_start=8.0, T_end=0.01, alpha=0.99995):
        """Simulated annealing for maximum cut with vectorized gain computation. Returns best labels found."""
        best_labels = labels.copy()
        best_cut = cut_size_from_labels(best_labels)
        current_cut = best_cut
        T = T_start
        
        # Precompute gains
        x = 2 * labels.astype(np.float64) - 1
        Ax = adj @ x
        gains = x * Ax

        for it in range(max_iters):
            if time_left() < 3.0:
                break
            
            # Pick random vertex
            v = np.random.randint(0, n)
            delta = int(gains[v])

            if delta >= 0:
                labels[v] = 1 - labels[v]
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
                
                # Update gains for v and neighbors
                neighbors = np.where(adj[v] == 1)[0]
                affected = np.concatenate([[v], neighbors])
                affected = np.unique(affected)
                x = 2 * labels.astype(np.float64) - 1
                Ax = adj @ x
                new_gains = x * Ax
                gains[affected] = new_gains[affected]
            else:
                # Accept with probability exp(delta / T)
                if T > 0.001:
                    if delta > -100 * T:
                        if np.random.random() < np.exp(delta / T):
                            labels[v] = 1 - labels[v]
                            current_cut += delta
                            
                            # Update gains for v and neighbors
                            neighbors = np.where(adj[v] == 1)[0]
                            affected = np.concatenate([[v], neighbors])
                            affected = np.unique(affected)
                            x = 2 * labels.astype(np.float64) - 1
                            Ax = adj @ x
                            new_gains = x * Ax
                            gains[affected] = new_gains[affected]

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
        labels_sa, cut_sa = simulated_annealing(best_labels, max_iters=150000, T_start=10.0, T_end=0.01, alpha=0.99998)
        if cut_sa > best_cut:
            best_cut = cut_sa
            best_labels = labels_sa

        # Try SA from a random start
        if time_left() > 15.0:
            labels_rand = np.random.randint(0, 2, size=n)
            labels_sa2, cut_sa2 = simulated_annealing(labels_rand, max_iters=100000, T_start=12.0, T_end=0.01, alpha=0.99997)
            if cut_sa2 > best_cut:
                best_cut = cut_sa2
                best_labels = labels_sa2

    # Step 5: Final polish
    if time_left() > 3.0:
        for _ in range(5):
            if time_left() < 1.5:
                break
            labels_polish, cut_polish = local_search(best_labels, max_rounds=50)
            if cut_polish > best_cut:
                best_cut = cut_polish
                best_labels = labels_polish
            else:
                break

    return best_labels.astype(int)
# EVOLVE-BLOCK-END
