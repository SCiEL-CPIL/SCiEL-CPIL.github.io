<<<<<<< SEARCH
def _make_random_compact(n, spacing=0.72, seed=None, min_dist=0.5):
    """Random compact cluster with minimum distance constraint (vectorized)."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    max_attempts = 20000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0:
                points.append(p)
            else:
                arr = np.array(points)
                dists = np.linalg.norm(arr - p, axis=1)
                if np.min(dists) >= min_dist:
                    points.append(p)
        attempts += 1
    if len(points) < n:
        # Fallback: allow closer packing
        points = []
        attempts = 0
        while len(points) < n and attempts < max_attempts:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if len(points) == 0:
                    points.append(p)
                else:
                    arr = np.array(points)
                    dists = np.linalg.norm(arr - p, axis=1)
                    if np.min(dists) >= 0.36:
                        points.append(p)
            attempts += 1
        if len(points) < n:
            # Last resort: just fill randomly
            while len(points) < n:
                p = rng.uniform(-radius, radius, 3)
                points.append(p)
    return np.array(points)
=======
def _make_random_compact(n, spacing=0.72, seed=None, min_dist=0.5):
    """Random compact cluster with minimum distance constraint (vectorized)."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    max_attempts = 5000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0:
                points.append(p)
            else:
                arr = np.array(points)
                dists = np.linalg.norm(arr - p, axis=1)
                if np.min(dists) >= min_dist:
                    points.append(p)
        attempts += 1
    if len(points) < n:
        # Fallback: allow closer packing
        points = []
        attempts = 0
        while len(points) < n and attempts < max_attempts:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if len(points) == 0:
                    points.append(p)
                else:
                    arr = np.array(points)
                    dists = np.linalg.norm(arr - p, axis=1)
                    if np.min(dists) >= 0.36:
                        points.append(p)
            attempts += 1
        if len(points) < n:
            # Last resort: just fill randomly
            while len(points) < n:
                p = rng.uniform(-radius, radius, 3)
                points.append(p)
    return np.array(points)
>>>>>>> REPLACE

