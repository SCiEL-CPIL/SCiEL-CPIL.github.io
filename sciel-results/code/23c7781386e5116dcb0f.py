<<<<<<< SEARCH
    def smooth_penalty(x, y, r, w, h, beta=50.0):
        """Smooth penalty for constraint violations using soft-min."""
        # Wall constraints: x - r >= 0, w - x - r >= 0, y - r >= 0, h - y - r >= 0
        wall_viol = np.concatenate([
            np.minimum(x - r, 0),
            np.minimum(w - x - r, 0),
            np.minimum(y - r, 0),
            np.minimum(h - y - r, 0)
        ])
        
        # Overlap constraints: dist - ri - rj >= 0
        overlap_viol = []
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                overlap_viol.append(np.minimum(dist - r[i] - r[j], 0))
        
        all_viol = np.concatenate([wall_viol, overlap_viol])
        # Soft penalty: sum of soft-min
        neg_viol = -all_viol  # positive when violated
        # log-sum-exp for smooth max
        max_val = np.max(neg_viol)
        if max_val > 1e-12:
            penalty = np.log(np.sum(np.exp(beta * (neg_viol - max_val)))) / beta + max_val
        else:
            penalty = 0.0
        return penalty * 100.0
=======
    def smooth_penalty(x, y, r, w, h, mu=100.0):
        """Quadratic penalty for constraint violations - smoother gradients than log-sum-exp."""
        # Wall constraints: x - r >= 0, w - x - r >= 0, y - r >= 0, h - y - r >= 0
        wall_viol = np.concatenate([
            np.minimum(x - r, 0),
            np.minimum(w - x - r, 0),
            np.minimum(y - r, 0),
            np.minimum(h - y - r, 0)
        ])
        
        # Overlap constraints: dist - ri - rj >= 0
        overlap_viol = []
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                overlap_viol.append(np.minimum(dist - r[i] - r[j], 0))
        
        all_viol = np.concatenate([wall_viol, overlap_viol])
        # Quadratic penalty: 0.5 * mu * sum(violation^2)
        neg_viol = -all_viol  # positive when violated
        penalty = 0.5 * mu * np.sum(neg_viol * neg_viol)
        return penalty
>>>>>>> REPLACE

