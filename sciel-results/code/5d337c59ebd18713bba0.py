<<<<<<< SEARCH
def _sa_aspect_ratio(best_circles, best_val, time_budget):
    """Simulated annealing over aspect ratio w (h = 2 - w)."""
    start = time.time()
    w = 1.0
    best_c = best_circles.copy()
    best_v = best_val
    T = 0.1
    for _ in range(200):
        if time.time() - start > time_budget:
            break
        w_new = np.clip(w + random.gauss(0, 0.05), 0.3, 1.7)
        h_new = 2.0 - w_new
        # re-init at new aspect
        c = _hex_init(w_new, h_new, seed=int(w_new * 1000))
        c = _improve(c, w_new, h_new, time_budget - (time.time() - start))
        if _strictly_valid(c, w_new, h_new):
            v = float(np.sum(c[:, 2]))
            if v > best_v or random.random() < np.exp((v - best_v) / max(T, 1e-6)):
                best_v = v; best_c = c.copy(); w = w_new
        T *= 0.97
    return best_c, best_v
=======
def _sa_aspect_ratio(best_circles, best_val, time_budget):
    """Simulated annealing over aspect ratio w (h = 2 - w)."""
    start = time.time()
    w = 1.0
    best_c = best_circles.copy()
    best_v = best_val
    T = 0.08
    for it in range(300):
        if time.time() - start > time_budget:
            break
        w_new = np.clip(w + random.gauss(0, 0.04), 0.3, 1.7)
        h_new = 2.0 - w_new
        remaining = time_budget - (time.time() - start)
        if remaining < 3.0:
            break
        # re-init at new aspect
        c = _hex_init(w_new, h_new, seed=int(w_new * 1000))
        c = _improve(c, w_new, h_new, min(remaining - 1.0, 6.0))
        if _strictly_valid(c, w_new, h_new):
            v = float(np.sum(c[:, 2]))
            if v > best_v or random.random() < np.exp((v - best_v) / max(T, 1e-6)):
                best_v = v; best_c = c.copy(); w = w_new
        T *= 0.98
    return best_c, best_v
>>>>>>> REPLACE

