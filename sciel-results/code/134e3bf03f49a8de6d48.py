def _relax(pts0, t_start, k=6, steps=40000, lr0=1e-3, decay=0.99995, beta=1.0):
    pts = pts0.copy()
    v = np.zeros_like(pts)
    m = np.zeros_like(pts)
    t = 0
    lr = lr0
    best_r, best_pts = -1.0, pts.copy()
    
    # We will use all pairs for simplicity and speed with vectorization
    # N=200 -> 40k pairs. NumPy handles this easily.
    
    for step in range(steps):
        if time.time() - t_start > DEADLINE - SAFE:
            break
            
        # 1. Pairwise distances and differences
        # diff[i, j] = pts[i] - pts[j]
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        # Avoid division by zero
        np.fill_diagonal(dist, 1e12)
        
        # Force magnitude: 1/dist^2 (repulsion)
        # We want to push apart.
        # f = 1 / dist^2
        f_mag = 1.0 / (dist * dist)
        # Set diagonal to 0
        np.fill_diagonal(f_mag, 0.0)
        
        # Force vector on i from j: unit(i-j) * f_mag
        # unit = diff / dist[..., None]
        # But we need to handle dist=0 carefully, though we set diag to large.
        # For i != j, dist > 0.
        unit = diff / dist[..., None]
        
        # F[i] = sum_j (unit[i, j] * f_mag[i, j])
        # Note: unit[i, j] is vector from j to i.
        # Force on i is repulsive, so it points from j to i. Correct.
        F = np.sum(unit * f_mag[..., None], axis=1)
        
        # 2. Boundary forces
        # Push inward if close to walls
        # c = 0.5 - |pts|
        # If c < margin, push towards 0
        margin = 0.1
        # For each coordinate
        for ax in range(2):
            # Distance to + wall
            over_plus = 0.5 - pts[:, ax]
            close_plus = over_plus < margin
            if np.any(close_plus):
                strength = (margin - over_plus[close_plus]) / margin
                direction = -np.sign(pts[close_plus, ax])
                F[close_plus, ax] += direction * strength * 2.0
            
            # Distance to - wall
            over_minus = pts[:, ax] + 0.5
            close_minus = over_minus < margin
            if np.any(close_minus):
                strength = (margin - over_minus[close_minus]) / margin
                direction = -np.sign(pts[close_minus, ax])
                F[close_minus, ax] += direction * strength * 2.0
        
        # 3. Centering
        F -= 1e-4 * pts
        
        # 4. Adam update
        v = 0.9 * v + 0.1 * F
        m = 0.99 * m + 0.01 * (F * F)
        m = np.maximum(m, 1e-8)
        pts += lr * v / np.sqrt(m)
        lr *= decay
        
        # 5. Clip
        pts = np.clip(pts, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # 6. Score check (less frequent to save time, or use a cheap proxy?)
        # _score is O(N^2) too. With N=200, it's fast in NumPy.
        # Let's check every 100 steps.
        if step % 100 == 0:
            r, _, _ = _score(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
                
    return best_pts, best_r
