import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Precompute all triangle indices for vectorized area computation
    tri_indices = np.array(list(combinations(range(N), 3)))
    n_tris = len(tri_indices)
    
    def _all_triangle_areas(pts):
        """Vectorized computation of all triangle areas."""
        i, j, k = tri_indices[:, 0], tri_indices[:, 1], tri_indices[:, 2]
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(cross)
    
    def _hull_area_fast(pts):
        """Compute convex hull area quickly."""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 0.0
    
    def _score(pts):
        """Compute min triangle area / hull area."""
        hull_area = _hull_area_fast(pts)
        if hull_area < 1e-12:
            return 0.0
        areas = _all_triangle_areas(pts)
        return np.min(areas) / hull_area
    
    def _objective_hard(flat_pts):
        """Hard objective: negative of the score."""
        pts = flat_pts.reshape(N, 2)
        s = _score(pts)
        return -s
    
    def _objective_soft(flat_pts, beta=200.0):
        """Soft-min surrogate for smooth gradient computation."""
        pts = flat_pts.reshape(N, 2)
        hull_area = _hull_area_fast(pts)
        if hull_area < 1e-12:
            return 1.0
        areas = _all_triangle_areas(pts)
        # Soft minimum: -log(sum(exp(-area/beta)))/beta
        min_val = -np.log(np.sum(np.exp(-areas / beta))) / beta
        return min_val / hull_area
    
    def _objective_soft_grad(flat_pts, beta=200.0):
        """Gradient of soft objective."""
        pts = flat_pts.reshape(N, 2)
        hull_area = _hull_area_fast(pts)
        if hull_area < 1e-12:
            return np.ones(2 * N)
        areas = _all_triangle_areas(pts)
        # Soft min value
        log_sum = np.log(np.sum(np.exp(-areas / beta)))
        soft_min = -log_sum / beta
        
        # Gradient of soft_min w.r.t. areas
        weights = np.exp(-areas / beta) / np.sum(np.exp(-areas / beta))
        
        # Gradient of soft_min w.r.t. points
        grad = np.zeros(2 * N)
        for idx in range(n_tris):
            i, j, k = tri_indices[idx]
            pi = pts[i]
            pj = pts[j]
            pk = pts[k]
            v1 = pj - pi
            v2 = pk - pi
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            # d(area)/d(points) for triangle (i,j,k)
            # area = 0.5 * |cross|, assuming positive orientation for simplicity
            # d(area)/dp_i = 0.5 * (-v2, v1) ... actually let's use the signed cross
            # For the absolute value, sign matters. Let's use the raw cross.
            sgn = 1.0 if cross >= 0 else -1.0
            # d(0.5*cross)/dp_i = 0.5 * (-(pk-pj)_y, (pk-pj)_x) = 0.5 * (pj_y - pk_y, pk_x - pj_x)
            # d(0.5*cross)/dp_j = 0.5 * ((pk-pi)_y, -(pk-pi)_x) = 0.5 * (pk_y - pi_y, pi_x - pk_x)
            # d(0.5*cross)/dp_k = 0.5 * (-(pj-pi)_y, (pj-pi)_x) = 0.5 * (pi_y - pj_y, pj_x - pi_x)
            w = weights[idx] * sgn / beta
            grad[2*i] += w * (pj[1] - pk[1])
            grad[2*i+1] += w * (pk[0] - pj[0])
            grad[2*j] += w * (pk[1] - pi[1])
            grad[2*j+1] += w * (pi[0] - pk[0])
            grad[2*k] += w * (pi[1] - pj[1])
            grad[2*k+1] += w * (pj[0] - pi[0])
        
        # Now divide by hull_area and add the derivative from hull_area in denominator
        # d(soft_min/hull_area)/dx = (grad_soft_min * hull_area - soft_min * grad_hull_area) / hull_area^2
        # Approximate: just use grad_soft_min / hull_area (ignore hull_area gradient for simplicity)
        return grad / hull_area
    
    def _normalize_unit_area(pts):
        """Normalize points to unit area convex hull."""
        hull_area = _hull_area_fast(pts)
        if hull_area < 1e-12:
            return pts
        centroid = np.mean(pts, axis=0)
        scaled = (pts - centroid) / np.sqrt(hull_area)
        return scaled
    
    def _hull_pull(pts, strength=0.01):
        """Pull interior points slightly toward hull to avoid degenerate configurations."""
        hull_area = _hull_area_fast(pts)
        if hull_area < 1e-12:
            return pts
        hull = ConvexHull(pts)
        hull_indices = set(hull.vertices)
        result = pts.copy()
        centroid = np.mean(pts[hull_indices], axis=0)
        for i in range(N):
            if i not in hull_indices:
                result[i] += strength * (centroid - pts[i])
        return result
    
    def _best_of_two(score1, pts1, score2, pts2):
        if score1 >= score2:
            return score1, pts1
        return score2, pts2
    
    def _nm(flat_pts, maxiter=2000):
        """Nelder-Mead optimization."""
        try:
            res = minimize(_objective_hard, flat_pts, method='Nelder-Mead',
                          options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False})
            if res.fun < 0:
                return -res.fun, res.x
        except Exception:
            pass
        return 0.0, flat_pts
    
    def _lbfgs(flat_pts, maxiter=1000):
        """L-BFGS with soft objective for gradient-based search."""
        try:
            res = minimize(_objective_soft, flat_pts, method='L-BFGS-B',
                          jac=lambda x: _objective_soft_grad(x),
                          options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10, 'disp': False})
            pts = res.x.reshape(N, 2)
            s = _score(pts)
            if s > 0:
                return s, res.x
        except Exception:
            pass
        return 0.0, flat_pts
    
    def _powell(flat_pts, maxiter=3000):
        """Powell optimization."""
        try:
            res = minimize(_objective_hard, flat_pts, method='Powell',
                          options={'maxiter': maxiter, 'ftol': 1e-14, 'disp': False})
            if res.fun < 0:
                return -res.fun, res.x
        except Exception:
            pass
        return 0.0, flat_pts
    
    def _coordinate_descent(pts, n_iters=50, step=0.001):
        """Simple coordinate descent refinement."""
        best = pts.copy()
        best_score = _score(best)
        for it in range(n_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                for coord in range(2):
                    for direction in [-1, 1]:
                        trial = best.copy()
                        trial[i, coord] += direction * step
                        s = _score(trial)
                        if s > best_score:
                            best = trial
                            best_score = s
                            improved = True
            if not improved:
                step *= 0.5
                if step < 1e-8:
                    break
        return best_score, best
    
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    pts0 = x0.reshape(N, 2)
    
    best_score_val = _score(pts0)
    best_result = x0.copy()
    
    # Phase 1: L-BFGS with soft objective for gradient-based exploration
    if time.time() < deadline - 30:
        s, x = _lbfgs(x0, maxiter=500)
        best_score_val, best_result = _best_of_two(best_score_val, best_result, s, x)
    
    # Phase 2: Powell from best result
    if time.time() < deadline - 20:
        s, x = _powell(best_result, maxiter=3000)
        best_score_val, best_result = _best_of_two(best_score_val, best_result, s, x)
    
    # Phase 3: Multiple perturbed restarts
    np.random.seed(42)
    for pert_scale in [0.01, 0.02, 0.05, 0.1]:
        if time.time() > deadline - 15:
            break
        x0_pert = best_result + np.random.normal(0, pert_scale, best_result.shape)
        # Try L-BFGS first (faster convergence from good starts)
        s1, x1 = _lbfgs(x0_pert, maxiter=300)
        s, x = _best_of_two(s1, x1, 0.0, x0_pert)
        # Then Powell polish
        if time.time() < deadline - 10:
            s2, x2 = _powell(x, maxiter=1500)
            s, x = _best_of_two(s, x, s2, x2)
        best_score_val, best_result = _best_of_two(best_score_val, best_result, s, x)
    
    # Phase 4: Coordinate descent fine-tuning
    if time.time() < deadline - 5:
        pts_best = best_result.reshape(N, 2)
        s_cd, pts_cd = _coordinate_descent(pts_best, n_iters=30, step=0.002)
        if s_cd > best_score_val:
            best_score_val = s_cd
            best_result = pts_cd.flatten()
    
    # Phase 5: Final tight Powell polish
    if time.time() < deadline - 3 and best_score_val > 0:
        s_final, x_final = _powell(best_result, maxiter=1000)
        best_score_val, best_result = _best_of_two(best_score_val, best_result, s_final, x_final)
    
    return best_result.reshape(N, 2)
