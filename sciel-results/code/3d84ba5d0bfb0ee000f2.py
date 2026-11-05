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
        return E - 0.5 * xTadjx

    def local_search_vectorized(labels, max_rounds=100):
        """
        Vectorized hill-climbing local search.
        Flips all vertices that improve the cut in a single pass.
        This is faster than sequential flipping for dense-ish graphs or when many moves are possible.
        """
        x = 2 * labels.astype(np.float64) - 1
        best_cut = cut_size_from_labels(labels)
        best_x = x.copy()
        current_cut = best_cut
        
        for rnd in range(max_rounds):
            if time_left() < 2.0:
                break
            
            # Compute gain for flipping each vertex
            # delta = 2 * x[v] * (adj @ x)[v]
            Ax = adj @ x
            gains = 2 * x * Ax
            
            # Find vertices with positive gain
            to_flip = np.where(gains > 0)[0]
            
            if len(to_flip) == 0:
                break
            
            # Flip all identified vertices
            x[to_flip] = -x[to_flip]
            
            # Update cut size
            # We can recompute or update. Recomputing is O(n^2) but vectorized.
            # For n=100, it's fast.
            new_cut = cut_size_from_labels((x + 1) // 2)
            
            if new_cut > current_cut:
                current_cut = new_cut
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_x = x.copy()
            else:
                # If flipping all together decreased cut, it might be due to interactions.
                # However, usually if individual gains are positive, the sum is positive 
                # unless there are strong negative correlations between flips.
                # In Max Cut, flipping two adjacent nodes both with positive gain 
                # might cancel out. 
                # To be safe, if the batch flip fails, we could fall back to sequential,
                # but for speed, we often accept the batch if it's a local search step.
                # Actually, if gains are computed based on current state, and we flip 
                # a set S, the true gain is sum(gains[v] for v in S) - 2 * (edges within S).
                # If we just flip and it gets worse, we revert?
                # Let's revert if it got worse to maintain monotonicity or track best.
                # But local search usually implies monotonic improvement.
                # If batch flip worsens, we should probably not have done it.
                # Let's just keep the best_x found so far.
                pass

        return (best_x + 1) // 2, best_cut

    def simulated_annealing_optimized(labels, max_iters=500000, T_start=10.0, T_end=0.01, alpha=0.99995):
        """
        Optimized Simulated Annealing.
        Uses pre-computed Ax and O(n) updates for flips.
        """
        x = 2 * labels.astype(np.float64) - 1
        Ax = adj @ x
        current_cut = cut_size_from_labels(labels)
        best_cut = current_cut
        best_x = x.copy()
        T = T_start

        for it in range(max_iters):
            if time_left() < 3.0:
                break
            
            # Pick random vertex
            v = np.random.randint(0, n)
            
            # Compute delta = 2 * x[v] * Ax[v]
            delta = 2 * x[v] * Ax[v]
            
            if delta >= 0:
                # Accept
                # Update Ax
                Ax -= 2 * x[v] * adj[:, v]
                # Flip x
                x[v] = -x[v]
                # Update cut
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_x = x.copy()
            else:
                # Accept with probability exp(delta / T)
                if T > 0.001:
                    # Optimization: if delta is very negative, skip exp
                    if delta > -100 * T:
                        if np.random.random() < np.exp(delta / T):
                            # Update Ax
                            Ax -= 2 * x[v] * adj[:, v]
                            # Flip x
                            x[v] = -x[v]
                            # Update cut
                            current_cut += delta
                            if current_cut > best_cut:
                                best_cut = current_cut
                                best_x = x.copy()

            # Cool down
            T = max(T * alpha, T_end)

        return (best_x + 1) // 2, best_cut

    def multi_start_optimized(num_starts):
        """Try multiple random starts with optimized local search."""
        best_labels = None
        best_cut = -1
        for s in range(num_starts):
            if time_left() < 5.0:
                break
            labels = np.random.randint(0, 2, size=n)
            labels, cut = local_search_vectorized(labels, max_rounds=50)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
        return best_labels, best_cut

    # Strategy:
    # 1. Start with alternating pattern, run vectorized local search
    # 2. Try multiple random restarts with vectorized local search
    # 3. Run optimized SA from the best local search result
    # 4. Final polish with vectorized local search

    # Step 1: Alternating start
    labels_alt = np.arange(n) % 2
    best_labels, best_cut = local_search_vectorized(labels_alt, max_rounds=50)

    # Step 2: Random restarts with vectorized local search
    # We have ~130s left. Local search is very fast.
    if time_left() > 10.0:
        labels_rs, cut_rs = multi_start_optimized(50)
        if cut_rs > best_cut:
            best_cut = cut_rs
            best_labels = labels_rs

    # Step 3: Optimized SA
    # SA can escape local minima.
    if time_left() > 15.0:
        labels_sa, cut_sa = simulated_annealing_optimized(best_labels, max_iters=200000, T_start=10.0, T_end=0.01)
        if cut_sa > best_cut:
            best_cut = cut_sa
            best_labels = labels_sa

        # Try SA from a random start if time permits
        if time_left() > 10.0:
            labels_rand = np.random.randint(0, 2, size=n)
            labels_sa2, cut_sa2 = simulated_annealing_optimized(labels_rand, max_iters=100000, T_start=15.0, T_end=0.01)
            if cut_sa2 > best_cut:
                best_cut = cut_sa2
                best_labels = labels_sa2

    # Step 4: Final polish
    if time_left() > 2.0:
        labels_polish, cut_polish = local_search_vectorized(best_labels, max_rounds=50)
        if cut_polish > best_cut:
            best_cut = cut_polish
            best_labels = labels_polish

    return best_labels.astype(int)
# EVOLVE-BLOCK-END
