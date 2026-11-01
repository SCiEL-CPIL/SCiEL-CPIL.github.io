<<<<<<< SEARCH
def _ring_decomposition_init(n, num_rings=None, ring_spread=0.92):
    """
    Initialize points in concentric rings with optimized radial distances.
    Each ring gets a proportional number of points based on circumference.
    """
    if num_rings is None:
        num_rings = int(np.sqrt(n) / 2.0) + 3
    
    # Center point
    points = [[0.0, 0.0]]
    remaining = n - 1
    
    # Distribute points into rings based on circumference
    # Ring i (1-indexed) at radius r_i has circumference 2*pi*r_i
    # We want r_i proportional to i (roughly)
    # Number of points in ring i ~ 2*pi*r_i / (2*r) where r is circle radius
    
    # Estimate circle radius from area: N * pi * r^2 ~ pi * 1^2
    # r ~ 1/sqrt(N)
    r_est = 1.0 / np.sqrt(n)
    
    # Ring radii: r_i = ring_spread * i / num_rings
    ring_radii = [ring_spread * (i + 1) / (num_rings + 1) for i in range(num_rings)]
    
    # Distribute points proportionally to circumference
    circumferences = [2.0 * np.pi * rr for rr in ring_radii]
    total_circ = sum(circumferences)
    
    ring_counts = []
    for i, circ in enumerate(circumferences):
        count = int(remaining * circ / total_circ)
        ring_counts.append(max(1, count))
    
    # Adjust to get exact count
    diff = remaining - sum(ring_counts)
    if diff > 0:
        # Add to rings with most space
        for _ in range(diff):
            ring_counts[-1] += 1
    elif diff < 0:
        # Remove from outer rings
        for i in range(len(ring_counts) - 1, -1, -1):
            if ring_counts[i] > 1 and -diff > 0:
                ring_counts[i] -= 1
                diff += 1
    
    # Place points in each ring with angular offsets
    current_count = 1
    for i, (count, rr) in enumerate(zip(ring_counts, ring_radii)):
        if count == 0:
            continue
        # Angular spacing
        d_theta = 2.0 * np.pi / count
        # Offset to avoid alignment with previous rings
        offset = np.pi / count * (i % 2)
        for k in range(count):
            theta = k * d_theta + offset
            points.append([rr * np.cos(theta), rr * np.sin(theta)])
        current_count += count
    
    points = np.array(points[:n])
    
    # Add small random perturbation to break symmetry
    points += np.random.normal(0, 1e-4, points.shape)
    
    return points
=======
def _smart_ring_init(n, num_rings):
    """
    Initialize points in concentric rings by optimizing radial positions
    to maximize the minimum distance, assuming a specific ring count.
    """
    # Center point
    points = [[0.0, 0.0]]
    remaining = n - 1
    
    # Distribute points into rings proportional to index (approximates circumference)
    # n_i ~ i
    weights = np.arange(1, num_rings + 1)
    ring_counts = np.round(remaining * weights / weights.sum()).astype(int)
    
    # Adjust to exact count
    diff = remaining - sum(ring_counts)
    idx = 0
    while diff != 0:
        if diff > 0:
            ring_counts[idx % num_rings] += 1
            diff -= 1
        else:
            if ring_counts[idx % num_rings] > 1:
                ring_counts[idx % num_rings] -= 1
                diff += 1
        idx += 1

    # Binary search for max radius r
    # Constraints:
    # 1. r_k + r <= 1  => r_k <= 1 - r
    # 2. r_{i+1} - r_i >= 2r (radial gap)
    # 3. r_i * sin(pi / n_i) >= r (angular gap)
    
    # We can express r_i in terms of r and the outermost ring radius r_k.
    # Let r_k = 1 - r (tight boundary).
    # Then r_{k-1} = r_k - 2r = 1 - 3r
    # r_{k-2} = r_{k-1} - 2r = 1 - 5r
    # ...
    # r_i = 1 - (2(k-i) + 1) r
    
    def check_radius(r):
        for i in range(num_rings):
            r_i = 1.0 - (2 * (num_rings - 1 - i) + 1) * r
            if r_i <= 0:
                return False
            n_i = ring_counts[i]
            # Angular constraint: r_i * sin(pi/n_i) >= r
            if r_i * np.sin(np.pi / n_i) < r:
                return False
        return True

    lo, hi = 0.0, 1.0 / np.sqrt(n) * 1.1
    for _ in range(50):
        mid = (lo + hi) / 2
        if check_radius(mid):
            lo = mid
        else:
            hi = mid
    
    r_opt = lo
    # Generate points
    current_count = 1
    for i in range(num_rings):
        n_i = ring_counts[i]
        r_i = 1.0 - (2 * (num_rings - 1 - i) + 1) * r_opt
        if r_i < 0:
            r_i = 0.0
        d_theta = 2.0 * np.pi / n_i
        offset = np.pi / n_i * (i % 2)
        for k in range(n_i):
            theta = k * d_theta + offset
            points.append([r_i * np.cos(theta), r_i * np.sin(theta)])
        current_count += n_i
    
    points = np.array(points[:n])
    points += np.random.normal(0, 1e-4, points.shape)
    return points
>>>>>>> REPLACE

