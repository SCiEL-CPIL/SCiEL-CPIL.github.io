<<<<<<< SEARCH
    def local_improve(labels):
        # Greedy local improvement: flip vertex if it improves cut
        improved = True
        while improved:
            improved = False
            for v in range(n):
                # Current contribution of v to cut: neighbors with different label
                cur_cross = adj[v][labels != labels[v]].sum()
                # If we flip v, new cross count = neighbors with same label (old)
                new_cross = adj[v][labels == labels[v]].sum()
                if new_cross > cur_cross:
                    labels[v] = 1 - labels[v]
                    improved = True
        return labels
=======
    def local_improve(labels):
        # Greedy local improvement: flip vertex if it improves cut
        # Use vectorized delta computation for speed
        improved = True
        while improved:
            improved = False
            # For each vertex v: delta = (neighbors with same label) - (neighbors with diff label)
            # = 2 * (same_count) - degree[v]
            # If delta > 0, flipping improves the cut
            same_counts = np.zeros(n, dtype=np.int32)
            for v in range(n):
                same_counts[v] = adj[v][labels == labels[v]].sum()
            degrees = adj.sum(axis=1)
            deltas = 2 * same_counts - degrees
            flip_mask = deltas > 0
            if np.any(flip_mask):
                labels[flip_mask] = 1 - labels[flip_mask]
                improved = True
        return labels
>>>>>>> REPLACE

