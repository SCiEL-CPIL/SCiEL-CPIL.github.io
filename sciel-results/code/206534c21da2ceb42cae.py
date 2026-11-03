def _relax_hard_disk(c, R, iters=1000, eta0=0.1, eta1=0.01):
    c = c.copy()
    for it in range(iters):
        t = it / max(iters - 1, 1)
        eta = eta0 + (eta1 - eta0) * t
        # Pairwise forces
        diff = c[:, None, :] - c[None, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(d, 1e12)
        # Overlap magnitude
        mag = np.maximum(2 * R - d, 0.0) / np.maximum(d, 1e-12)
        np.fill_diagonal(mag, 0.0)
        f = np.sum(mag[:, :, None] * diff, axis=1)
        
        # Wall forces
        # If c < -0.5 + R, push right (+x)
        # If c > 0.5 - R, push left (-x)
        f[:, 0] += np.maximum((-0.5 + R) - c[:, 0], 0.0) - np.maximum(c[:, 0] - (0.5 - R), 0.0)
        f[:, 1] += np.maximum((-0.5 + R) - c[:, 1], 0.0) - np.maximum(c[:, 1] - (0.5 - R), 0.0)
        
        c += eta * f
        c = np.clip(c, -0.5, 0.5)
    return c
