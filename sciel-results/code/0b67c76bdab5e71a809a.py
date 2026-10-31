<<<<<<< SEARCH
def _post_process(pos, radii, deadline):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(300):
        if time.time() > deadline:
            break
        valid = True
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max(1e-8, max_r * 0.99)
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    if dist < 1e-12:
                        scale = 0.5
                    else:
                        scale = dist / (min_dist + 1e-12) * 0.99
                    radii[i] = max(1e-8, radii[i] * scale)
                    radii[j] = max(1e-8, radii[j] * scale)
                    valid = False
        if valid:
            break
    
    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(1e-8, min(radii[i], max_r))
    
    return pos, radii
=======
def _post_process(pos, radii, deadline):
    """Iteratively shrink circles to ensure validity (vectorized)."""
    pos = pos.copy()
    radii = radii.copy()

    for iteration in range(300):
        if time.time() > deadline:
            break
        valid = True

        # Boundary check (vectorized)
        max_rs = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                           np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
        over_mask = radii > max_rs
        if np.any(over_mask):
            radii[over_mask] = np.maximum(1e-8, max_rs[over_mask] * 0.99)
            valid = False

        # Overlap check (vectorized)
        idx = np.triu_indices(N, k=1)
        dx = pos[idx[0], 0] - pos[idx[1], 0]
        dy = pos[idx[0], 1] - pos[idx[1], 1]
        dists = np.sqrt(dx * dx + dy * dy)
        min_dists = radii[idx[0]] + radii[idx[1]]
        overlap_mask = dists < min_dists
        if np.any(overlap_mask):
            for k in range(len(overlap_mask)):
                if overlap_mask[k]:
                    i, j = idx[0][k], idx[1][k]
                    dist = dists[k]
                    if dist < 1e-12:
                        scale = 0.5
                    else:
                        scale = dist / (min_dists[k] + 1e-12) * 0.99
                    radii[i] = max(1e-8, radii[i] * scale)
                    radii[j] = max(1e-8, radii[j] * scale)
            valid = False

        if valid:
            break

    # Final boundary clamp
    max_rs = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                       np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    radii = np.maximum(1e-8, np.minimum(radii, max_rs))

    return pos, radii
>>>>>>> REPLACE

