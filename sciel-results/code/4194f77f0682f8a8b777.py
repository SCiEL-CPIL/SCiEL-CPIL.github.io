    def init_hex(N):
        # Estimate spacing
        # Area = pi. N circles.
        # Hex lattice area per circle ~ sqrt(3)/2 * s^2
        # N * sqrt(3)/2 * s^2 ~ pi
        # s ~ sqrt(2*pi / (N*sqrt(3)))
        s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
        # Generate grid
        xs = np.arange(-2, 2, s)
        ys = np.arange(-2, 2, s * np.sqrt(3)/2)
        points = []
        for i, x in enumerate(xs):
            for y in ys:
                # Shift every other row
                y_shift = y + (s/2 if i % 2 else 0)
                points.append([x, y_shift])
        points = np.array(points)
        # Keep only those in disk (with margin)
        r = np.linalg.norm(points, axis=1)
        points = points[r <= 1.0]
        
        if len(points) < N:
             # If not enough, expand grid or use random
             return np.random.rand(N, 2) * 2 - 1 # Random fallback
             
        # If too many, select N
        # Random selection is bad for packing.
        # Better: Keep those closest to center? Or random?
        # Random selection from a dense grid is okay, but might be clustered.
        # Let's just take the first N after shuffling?
        # Actually, for a hex grid, if we have more than N, we can just randomly drop some.
        indices = np.random.choice(len(points), N, replace=False)
        return points[indices]
    