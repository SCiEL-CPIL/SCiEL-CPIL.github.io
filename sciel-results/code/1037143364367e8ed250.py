import numpy as np

def erdos_szekeres_problem7() -> np.ndarray:
    n = 33
    points = np.zeros((n, 2))
    
    def generate_es(k, x0, y0, w, h):
        if k == 1:
            return np.array([[x0 + 0.5 * w, y0 + 0.5 * h]])
        
        top = generate_es(k - 1, x0, y0 + 0.5 * h, w, 0.5 * h)
        bottom = generate_es(k - 1, x0, y0, w, 0.5 * h)
        
        return np.vstack((top, bottom))

    pts_32 = generate_es(6, 0.0, 0.0, 1.0, 1.0)
    
    # Add 33rd point
    p33 = np.array([0.5, 0.5])
    
    # Check if p33 is too close to any existing point
    dists = np.linalg.norm(pts_32 - p33, axis=1)
    if np.min(dists) < 1e-4:
        p33 = np.array([0.5, 0.51])
        
    points = np.vstack((pts_32, p33))
    
    # Perturb to avoid collinearity
    rng = np.random.default_rng(42)
    points += rng.uniform(-1e-6, 1e-6, size=(n, 2))
    
    # Clamp
    points = np.clip(points, 0.0, 1.0)
    
    return points
