<<<<<<< SEARCH
def _coordinate_descent_polish(pts, n_iters=500):
    """Coordinate descent polish: optimize each point one at a time."""
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    current = pts.copy()
    current_score = _min_area(current)
    
    for step in step_sizes:
        for _ in range(n_iters // len(step_sizes)):
            improved = False
            for i in range(N):
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        trial = current.copy()
                        trial[i] = _clamp_to_triangle(current[i] + np.array([dx, dy]))
                        trial_score = _min_area(trial)
                        if trial_score > current_score + 1e-15:
                            current = trial
                            current_score = trial_score
                            improved = True
            if not improved:
                break
    return current, current_score
=======
def _coordinate_descent_polish(pts, n_iters=500):
    """Coordinate descent polish: optimize each point one at a time."""
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    current = pts.copy()
    current_score = _min_area(current)
    
    for step in step_sizes:
        for _ in range(n_iters // len(step_sizes)):
            improved = False
            for i in range(N):
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        trial = current.copy()
                        trial[i] = _clamp_to_triangle(current[i] + np.array([dx, dy]))
                        trial_score = _min_area(trial)
                        if trial_score > current_score + 1e-15:
                            current = trial
                            current_score = trial_score
                            improved = True
            if not improved:
                break
    return current, current_score

def _high_precision_polish(pts, maxiter=5000):
    """High-precision Nelder-Mead polish with very tight tolerances."""
    try:
        res = minimize(
            _objective_hard,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-15, 'fatol': 1e-16, 'adaptive': True}
        )
        hp_pts = _clamp_all(res.x.reshape(-1, 2))
        return hp_pts, _min_area(hp_pts)
    except Exception:
        return pts, _min_area(pts)
>>>>>>> REPLACE

