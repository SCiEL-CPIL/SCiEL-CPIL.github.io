"""Rewritten program for Maximum Cut using a direct structural approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def partition(adjacency):
    """
    Assigns 0/1 labels to vertices to maximize the number of edges crossing the partition.
    Uses a combination of:
      1. Spectral initialization (Laplacian eigenvector)
      2. Fast local search (greedy hill-climbing) with vectorized gains
      3. Quantum-inspired annealing for escape from local optima
      4. Simulated annealing with vectorized gains
      5. Multi-start with random restarts
      6. Final high-precision polish pass
    Deadline guard ensures return well before 150s.
    """
    try:
        adj = np.asarray(adjacency, dtype=np.int32)
        n = adj.shape[0]
        if n == 0:
            return []

        # Ensure symmetric and no self loops for safety
        adj = (adj + adj.T) // 2
        np.fill_diagonal(adj, 0)

        start_time = time.time()
        DEADLINE = start_time + 140.0  # Safety margin before 150s kill

        def time_left():
            return DEADLINE - time.time()

        # Precompute
        degrees = adj.sum(axis=1)
        E = int(degrees.sum()) // 2
        adj_float = adj.astype(np.float64)
        degrees_float = degrees.astype(np.float64)

        # Cut size: number of edges with endpoints in different partitions
        def cut_size(labels):
            """Fast cut size: count edges crossing the cut."""
            diff = labels[:, None] != labels[None, :]
            return int(np.sum(adj * diff)) // 2

        def compute_gains(labels):
            """Vectorized computation of gain for flipping each vertex.
            gain[v] = 2 * (sum of adj[v,u] where labels[u] == labels[v]) - degrees[v]
            If gain > 0, flipping v increases the cut."""
            l0_mask = (labels == 0).astype(np.int8)
            l1_mask = (labels == 1).astype(np.int8)
            
            # For vertices with label 0: same_count = adj @ l0_mask
            # For vertices with label 1: same_count = adj @ l1_mask
            adj_l0 = adj_float @ l0_mask
            adj_l1 = adj_float @ l1_mask
            
            same_count = np.where(labels == 0, adj_l0, adj_l1)
            gains = 2.0 * same_count - degrees_float
            return gains

        def single_gain(labels, v):
            """Compute gain for flipping vertex v only. O(n) instead of O(n^2)."""
            # Count neighbors of v with same label as v
            same_label = labels[v]
            same_neighbors = int(np.sum(adj[v] * (labels == same_label)))
            gain = 2 * same_neighbors - int(degrees[v])
            return gain

        def local_search(labels, max_rounds=500):
            """Greedy hill-climbing with fast single-vertex gain computation."""
            best_labels = labels.copy().astype(np.int8)
            best_cut = cut_size(best_labels)
            
            for rnd in range(max_rounds):
                if time_left() < 2.0:
                    break
                improved = False
                # Try flipping each vertex once in random order
                order = np.random.permutation(n)
                for v in order:
                    if time_left() < 1.0:
                        break
                    
                    # Calculate gain of flipping v using fast O(n) method
                    delta = single_gain(best_labels, v)
                    
                    if delta > 0:
                        best_labels[v] = 1 - best_labels[v]
                        best_cut += delta
                        improved = True
                
                if not improved:
                    break
            return best_labels, best_cut

        def simulated_annealing(labels, max_iters=50000, T_start=4.0, T_end=0.05, cooling=0.99997):
            """Simulated annealing to escape local optima with fast gains and 2-flip moves."""
            best_labels = labels.copy().astype(np.int8)
            best_cut = cut_size(best_labels)
            current_labels = labels.copy().astype(np.int8)
            current_cut = best_cut
            T = T_start

            for it in range(max_iters):
                if time_left() < 3.0:
                    break
                
                # Occasionally try a 2-flip move to escape local optima
                if np.random.random() < 0.1:
                    v1 = np.random.randint(0, n)
                    v2 = np.random.randint(0, n)
                    if v1 == v2:
                        v2 = (v1 + 1) % n
                    
                    # Compute gain for flipping both v1 and v2
                    # Gain of flipping v1 alone + gain of flipping v2 alone - interaction
                    g1 = single_gain(current_labels, v1)
                    # After flipping v1, recompute gain for v2
                    current_labels[v1] = 1 - current_labels[v1]
                    g2_after = single_gain(current_labels, v2)
                    current_labels[v1] = 1 - current_labels[v1]  # revert
                    
                    delta = g1 + g2_after
                    if delta >= 0:
                        current_labels[v1] = 1 - current_labels[v1]
                        current_labels[v2] = 1 - current_labels[v2]
                        current_cut += delta
                        if current_cut > best_cut:
                            best_cut = current_cut
                            best_labels = current_labels.copy()
                    else:
                        if T > 0.001:
                            prob = np.exp(delta / T)
                            if np.random.random() < prob:
                                current_labels[v1] = 1 - current_labels[v1]
                                current_labels[v2] = 1 - current_labels[v2]
                                current_cut += delta
                else:
                    v = np.random.randint(0, n)
                    delta = single_gain(current_labels, v)
                    
                    if delta >= 0:
                        current_labels[v] = 1 - current_labels[v]
                        current_cut += delta
                        if current_cut > best_cut:
                            best_cut = current_cut
                            best_labels = current_labels.copy()
                    else:
                        if T > 0.001:
                            prob = np.exp(delta / T)
                            if np.random.random() < prob:
                                current_labels[v] = 1 - current_labels[v]
                                current_cut += delta

                T = max(T * cooling, T_end)

            return best_labels, best_cut

        def quantum_annealing(labels, max_iters=80000, s_start=1.0, s_end=0.0, cooling=0.99998):
            """Quantum-inspired annealing: interpolate between transverse field and problem Hamiltonian.
            The transverse field strength s controls the effective temperature for accepting worse moves."""
            best_labels = labels.copy().astype(np.int8)
            best_cut = cut_size(best_labels)
            current_labels = labels.copy().astype(np.int8)
            current_cut = best_cut
            s = s_start  # transverse field strength (1.0 = pure quantum, 0.0 = pure classical)
            
            for it in range(max_iters):
                if time_left() < 3.0:
                    break
                
                # Compute gains for the selected vertex
                v = np.random.randint(0, n)
                gains = compute_gains(current_labels)
                delta = int(gains[v])

                # Quantum tunneling: effective temperature scales with transverse field
                T_eff = s * 3.0 + 0.05  # minimum temperature to allow some exploration
                
                if delta >= 0:
                    # Accept
                    current_labels[v] = 1 - current_labels[v]
                    current_cut += delta
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_labels = current_labels.copy()
                else:
                    # Accept with probability exp(delta / T_eff)
                    if T_eff > 0.01:
                        prob = np.exp(delta / T_eff)
                        if np.random.random() < prob:
                            current_labels[v] = 1 - current_labels[v]
                            current_cut += delta

                # Update transverse field
                s = max(s * cooling, s_end)

            return best_labels, best_cut

        def spectral_init():
            """Use the Laplacian eigenvector to initialize the partition."""
            try:
                D = np.diag(degrees)
                L = D - adj_float
                from scipy.linalg import eigh
                evals, evecs = eigh(L, subset_by_index=[0, 1])
                v = evecs[:, 1]
                labels = (v > 0).astype(int)
                if labels.sum() == 0 or labels.sum() == n:
                    labels = np.random.randint(0, 2, size=n)
                return labels
            except Exception:
                return np.random.randint(0, 2, size=n)

        def greedy_sequential_init():
            """Greedy sequential initialization: assign vertices in order of degree,
            placing each vertex on the side that maximizes immediate cut contribution."""
            labels = np.full(n, -1, dtype=np.int8)  # -1 means unplaced
            order = np.argsort(-degrees)
            
            for v in order:
                if time_left() < 1.0:
                    break
                # Count placed neighbors on each side
                placed = labels >= 0
                adj_v = adj[v]
                
                # Neighbors of v that are already placed on side 0
                placed0 = placed & (labels == 0)
                placed1 = placed & (labels == 1)
                
                count0 = int(np.sum(adj_v * placed0))
                count1 = int(np.sum(adj_v * placed1))
                
                # Put v on the side with fewer placed neighbors (maximizes cut)
                if count0 < count1:
                    labels[v] = 0
                elif count1 < count0:
                    labels[v] = 1
                else:
                    labels[v] = np.random.randint(0, 2)
            
            # Any unplaced vertices get random labels
            unplaced = labels < 0
            if np.any(unplaced):
                labels[unplaced] = np.random.randint(0, 2, size=int(np.sum(unplaced)))
            
            return labels.astype(np.int8)

        # Strategy:
        # 1. Spectral initialization + local search
        # 2. Multiple random restarts with local search
        # 3. Quantum-inspired annealing from best solutions
        # 4. Simulated annealing from best solutions
        # 5. Final high-precision polish

        best_labels = None
        best_cut = -1

        # Phase 1: Spectral init + local search
        labels_spec = spectral_init()
        labels_ls, cut_ls = local_search(labels_spec, max_rounds=200)
        if cut_ls > best_cut:
            best_cut = cut_ls
            best_labels = labels_ls.copy()

        # Phase 1b: Greedy init + local search
        labels_greedy = greedy_sequential_init()
        labels_gls, cut_gls = local_search(labels_greedy, max_rounds=200)
        if cut_gls > best_cut:
            best_cut = cut_gls
            best_labels = labels_gls.copy()

        # Phase 2: Random restarts with local search
        num_restarts = 30
        for r in range(num_restarts):
            if time_left() < 5.0:
                break
            labels_rand = np.random.randint(0, 2, size=n)
            labels_rs, cut_rs = local_search(labels_rand, max_rounds=100)
            if cut_rs > best_cut:
                best_cut = cut_rs
                best_labels = labels_rs.copy()

        # Phase 3: Quantum-inspired annealing from best
        if time_left() > 15.0 and best_labels is not None:
            labels_qa, cut_qa = quantum_annealing(best_labels, max_iters=100000)
            if cut_qa > best_cut:
                best_cut = cut_qa
                best_labels = labels_qa.copy()
            # Polish after quantum annealing
            if time_left() > 5.0:
                labels_polish, cut_polish = local_search(best_labels, max_rounds=100)
                if cut_polish > best_cut:
                    best_cut = cut_polish
                    best_labels = labels_polish.copy()

        # Phase 4: Simulated annealing from best (with 2-flip moves)
        if time_left() > 15.0 and best_labels is not None:
            labels_sa, cut_sa = simulated_annealing(best_labels, max_iters=80000)
            if cut_sa > best_cut:
                best_cut = cut_sa
                best_labels = labels_sa.copy()
            # Polish after simulated annealing
            if time_left() > 5.0:
                labels_polish, cut_polish = local_search(best_labels, max_rounds=100)
                if cut_polish > best_cut:
                    best_cut = cut_polish
                    best_labels = labels_polish.copy()

        # Phase 4b: Additional SA rounds from current best if time permits
        if time_left() > 20.0 and best_labels is not None:
            labels_sa2, cut_sa2 = simulated_annealing(best_labels, max_iters=50000, T_start=2.0)
            if cut_sa2 > best_cut:
                best_cut = cut_sa2
                best_labels = labels_sa2.copy()
            if time_left() > 3.0:
                labels_polish, cut_polish = local_search(best_labels, max_rounds=100)
                if cut_polish > best_cut:
                    best_cut = cut_polish
                    best_labels = labels_polish.copy()

        # Phase 5: More random restarts with SA (faster than quantum annealing) if time permits
        if time_left() > 25.0:
            for r in range(8):
                if time_left() < 5.0:
                    break
                labels_rand = np.random.randint(0, 2, size=n)
                labels_sa3, cut_sa3 = simulated_annealing(labels_rand, max_iters=30000, T_start=3.0)
                if cut_sa3 > best_cut:
                    best_cut = cut_sa3
                    best_labels = labels_sa3.copy()
                # Polish
                if time_left() > 3.0:
                    labels_polish, cut_polish = local_search(best_labels, max_rounds=50)
                    if cut_polish > best_cut:
                        best_cut = cut_polish
                        best_labels = labels_polish.copy()

        # Phase 6: Final high-precision polish with more iterations
        if time_left() > 3.0 and best_labels is not None:
            for _ in range(30):
                if time_left() < 1.5:
                    break
                labels_polish, cut_polish = local_search(best_labels, max_rounds=300)
                if cut_polish > best_cut:
                    best_cut = cut_polish
                    best_labels = labels_polish.copy()
                else:
                    break

        if best_labels is None:
            best_labels = np.random.randint(0, 2, size=n)

        return best_labels.tolist()

    except Exception as e:
        # Fallback to random partition if anything fails
        try:
            n = np.asarray(adjacency).shape[0]
            return np.random.randint(0, 2, size=n).tolist()
        except:
            return []
# EVOLVE-BLOCK-END
