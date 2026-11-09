import numpy as np
from scipy.optimize import minimize

def compute_scores(polygon):
    n = len(polygon)
    scores = np.zeros(n)
    for i in range(n):
        v = polygon[i]
        dists = np.linalg.norm(polygon - v, axis=1)
        dists[i] = np.inf
        d = np.sort(dists)
        if np.any(d < 0.05):
            scores[i] = 0.0
            continue
        gaps = np.diff(d)
        k = np.argmin(gaps)
        D = (d[k] + d[k+1]) / 2
        if D < 0.1:
            scores[i] = 0.0
            continue
        if k == 0:
            idx = [0, 1, 2, 3]
        elif k == 17:
            idx = [15, 16, 17, 18]
        else:
            idx = [k-1, k, k+1, k+2]
        selected = d[idx]
        raw = np.mean(np.maximum(selected / D, D / selected))
        scores[i] = min(1.0, 1.0 / raw)
    return scores

def neg_min_score(params):
    n = 20
    radii = params
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    scores = compute_scores(polygon)
    return -np.min(scores)

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    x0 = np.ones(n)
    result = minimize(neg_min_score, x0, method='Nelder-Mead',
                      options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-10})
    radii = result.x
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    return polygon
