import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s buffer
    
    def score_points(points):
        """Compute the score for a set of points."""
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(8, k=1)
        d_pairs = dists[iu]
        if np.any(d_pairs < 1e-12):
            return 0.0
        D = np.max(d_pairs)
        if D < 1e-12:
            return 0.0
        # Scale to diameter 2
        scaled_dists = d_pairs * (2.0 / D)
        log_prod = np.sum(np.log(scaled_dists))
        score = np.exp(log_prod / 28.0)
        return score
    
    def objective(params):
        """Negative log of product of distances, normalized by diameter^28."""
        points = params.reshape(8, 2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(8, k=1)
        d_pairs = dists[iu]
        if np.any(d_pairs < 1e-12):
            return 1e10
        D = np.max(d_pairs)
        if D < 1e-12:
            return 1e10
        # Maximize sum(log(d_i)) - 28*log(D)
        val = np.sum(np.log(d_pairs)) - 28.0 * np.log(D)
        return -val
    
    def make_initial_guesses():
        """Generate diverse initial guesses."""
        guesses = []
        
        # Regular octagon
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        guesses.append(np.column_stack((x, y)).flatten())
        
        # Slightly perturbed regular octagons
        for scale in [0.1, 0.2, 0.3, 0.5]:
            for i in range(4):
                params = np.column_stack((x, y)).flatten().copy()
                # Perturb point i and its opposite
                idx = i * 2
                params[idx] += scale
                params[idx + 8] -= scale  # opposite point
                # Center
                cx = np.mean(params[0::2])
                cy = np.mean(params[1::2])
                params[0::2] -= cx
                params[1::2] -= cy
                guesses.append(params)
        
        # Non-symmetric patterns
        # Two squares
        for offset in [0.2, 0.3, 0.5]:
            angles1 = np.linspace(0, 2*np.pi, 4, endpoint=False)
            angles2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
            r1 = 1.0
            r2 = 1.0 - offset
            x1 = r1 * np.cos(angles1)
            y1 = r1 * np.sin(angles1)
            x2 = r2 * np.cos(angles2)
            y2 = r2 * np.sin(angles2)
            x = np.concatenate([x1, x2])
            y = np.concatenate([y1, y2])
            cx = np.mean(x)
            cy = np.mean(y)
            x -= cx
            y -= cy
            guesses.append(np.column_stack((x, y)).flatten())
        
        # Random points
        np.random.seed(42)
        for _ in range(10):
            angles = np.sort(np.random.uniform(0, 2*np.pi, 8))
            radii = np.random.uniform(0.5, 1.5, 8)
            x = radii * np.cos(angles)
            y = radii * np.sin(angles)
            cx = np.mean(x)
            cy = np.mean(y)
            x -= cx
            y -= cy
            guesses.append(np.column_stack((x, y)).flatten())
        
        return guesses
    
    guesses = make_initial_guesses()
    
    best_val = np.inf
    best_params = guesses[0].copy()
    
    # Stage 1: Coarse optimization with Nelder-Mead
    for guess in guesses:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, guess, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            continue
    
    # Stage 2: Fine optimization with L-BFGS-B from best result
    if time.time() < deadline:
        try:
            res = minimize(objective, best_params, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Stage 3: Try more diverse optimizations from top candidates
    if time.time() < deadline:
        # Collect top candidates
        candidate_vals = []
        for guess in guesses[:10]:
            if time.time() > deadline:
                break
            try:
                res = minimize(objective, guess, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8})
                candidate_vals.append((res.fun, res.x))
            except Exception:
                continue
        
        candidate_vals.sort(key=lambda x: x[0])
        
        # Refine top 3 candidates with L-BFGS-B
        for val, params in candidate_vals[:3]:
            if time.time() > deadline:
                break
            try:
                res = minimize(objective, params, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                continue
    
    points = best_params.reshape(8, 2)
    
    # Normalize to diameter 2
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    D = np.max(dists)
    if D > 0:
        points = points * (2.0 / D)
    
    # Verify score
    final_score = score_points(points)
    
    return points.tolist()
