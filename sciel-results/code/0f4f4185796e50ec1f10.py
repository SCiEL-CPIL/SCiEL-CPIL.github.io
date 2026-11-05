<<<<<<< SEARCH
    def simulated_annealing(labels, max_iters=50000, T_start=8.0, T_end=0.01, alpha=0.99995):
        """Simulated annealing to escape local optima."""
        best_labels = labels.copy()
        best_cut = cut_size(best_labels)
        current_labels = labels.copy()
        current_cut = best_cut
        T = T_start

        for it in range(max_iters):
            if time_left() < 3.0:
                break
            v = np.random.randint(0, n)
            delta = fast_cut_delta(current_labels, v)

            if delta >= 0:
                # Accept
                current_labels[v] = 1 - current_labels[v]
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = current_labels.copy()
            else:
                # Accept with probability exp(delta / T)
                if T > 0.001:
                    prob = np.exp(delta / T)
                    if np.random.random() < prob:
                        current_labels[v] = 1 - current_labels[v]
                        current_cut += delta

            T = max(T * alpha, T_end)

        return best_labels, best_cut
=======
    def simulated_annealing(labels, max_iters=50000, T_start=4.0, T_end=0.05, cooling=0.99997):
        """Simulated annealing to escape local optima.
        Uses vectorized gain computation and a smoother cooling schedule."""
        best_labels = labels.copy()
        best_cut = cut_size(best_labels)
        current_labels = labels.copy()
        current_cut = best_cut
        T = T_start

        # Precompute adj as float32 for fast matrix-vector products
        adj_f = adj.astype(np.float32)

        for it in range(max_iters):
            if time_left() < 3.0:
                break
            
            # Compute gains for all vertices
            s = 2 * current_labels.astype(np.float32) - 1
            As = adj_f @ s
            gains = (s * As).astype(np.int32)
            
            # Select vertex: mix of random and biased toward high-gain
            # Use a softmax-like selection to balance exploration/exploitation
            # But for speed, just pick randomly with some bias
            v = np.random.randint(0, n)
            delta = int(gains[v])

            if delta >= 0:
                # Accept
                current_labels[v] = 1 - current_labels[v]
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = current_labels.copy()
            else:
                # Accept with probability exp(delta / T)
                if T > 0.001:
                    # delta is negative, so exp(delta/T) < 1
                    prob = np.exp(delta / T)
                    if np.random.random() < prob:
                        current_labels[v] = 1 - current_labels[v]
                        current_cut += delta

            T = max(T * cooling, T_end)

        return best_labels, best_cut
>>>>>>> REPLACE

