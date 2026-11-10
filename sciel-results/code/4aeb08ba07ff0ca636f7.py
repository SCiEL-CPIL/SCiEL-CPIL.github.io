import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 600  # Safety margin

    def true_score(points):
        """Calculates (d_min / d_max)^2"""
        dists_sq = pdist(points, metric='sqeuclidean')
        if len(dists_sq) == 0:
            return 0.0
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        if d_max_sq < 1e-12:
            return 0.0
        return (d_min_sq / d_max_sq) ** 2

    def normalize_rms(points):
        """Normalizes points to have RMS radius 1"""
        rms = np.sqrt(np.mean(np.sum(points**2, axis=1)))
        if rms < 1e-6:
            return points
        return points / rms

    def smooth_obj(x_flat):
        """
        Smooth surrogate objective to maximize minimum distance.
        Minimizes sum of inverse squared distances.
        This pushes all points apart.
        """
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        dists_sq = pdist(pts, metric='sqeuclidean')
        # Avoid division by zero
        dists_sq = np.maximum(dists_sq, 1e-8)
        return np.sum(1.0 / dists_sq)

    def smooth_obj_grad(x_flat):
        """
        Gradient of smooth_obj for L-BFGS-B.
        Objective: sum_{i<j} 1 / ||x_i - x_j||^2
        """
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        
        # Gradient of normalize_rms is complex. 
        # For simplicity and robustness, we might skip analytic gradient 
        # and let L-BFGS-B use numerical gradient if needed, 
        # or implement a simpler gradient assuming fixed scale.
        # However, RMS normalization changes with x.
        # Let's stick to numerical gradient for safety and correctness of constraints,
        # or use a penalty method instead.
        
        # Actually, L-BFGS-B is fast enough with numerical gradients for 42 variables.
        return None # Force numerical gradient

    candidates = []

    # Candidate 1: Icosahedron vertices + 2 poles (standard best)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
        [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0], [-phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, -1.0]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    poles = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    candidates.append(np.vstack([ico, poles]))

    # Candidate 2: Icosahedron with poles shifted slightly (optimization seed)
    for shift in [0.0, 0.1, 0.2, 0.3]:
        p1 = np.array([0.0, 0.0, 1.0 - shift])
        p2 = np.array([0.0, 0.0, -(1.0 - shift)])
        # Renormalize poles to unit length? No, let RMS normalization handle it.
        # But if we want to test "off-sphere", we should scale them.
        # Let's try scaling the poles relative to the icosahedron.
        scale_poles = 1.0 - shift
        p1 = np.array([0.0, 0.0, scale_poles])
        p2 = np.array([0.0, 0.0, -scale_poles])
        cand = np.vstack([ico, p1.reshape(1, 3), p2.reshape(1, 3)])
        candidates.append(cand)

    # Candidate 3: Two concentric rings (7+7)
    # 7 points on a circle at z=h, 7 points on a circle at z=-h, rotated
    for h in [0.2, 0.4, 0.6, 0.8]:
        r = np.sqrt(1 - h**2)
        angles1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        angles2 = angles1 + np.pi / 7  # Rotate second ring
        ring1 = np.column_stack([r * np.cos(angles1), r * np.sin(angles1), np.full(7, h)])
        ring2 = np.column_stack([r * np.cos(angles2), r * np.sin(angles2), np.full(7, -h)])
        candidates.append(np.vstack([ring1, ring2]))

    # Candidate 4: Random perturbations of Icosahedron + Poles
    rng = np.random.RandomState(42)
    base = np.vstack([ico, poles])
    for _ in range(10):
        perturbed = base + rng.randn(n, d) * 0.1
        candidates.append(perturbed)

    best_score = 0.0
    best_points = None

    # Score candidates
    for cand in candidates:
        cand = normalize_rms(cand)
        s = true_score(cand)
        if s > best_score:
            best_score = s
            best_points = cand.copy()

    # Polish candidates
    # Use L-BFGS-B on smooth objective, then Nelder-Mead on true objective
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 30:
            break

        cand = normalize_rms(cand)
        
        try:
            # Phase 1: Smooth optimization
            res1 = minimize(
                smooth_obj,
                cand.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            pts1 = normalize_rms(res1.x.reshape(n, d))
            
            # Phase 2: Nelder-Mead on true objective for high precision
            def nm_obj(x_flat):
                pts = x_flat.reshape(n, d)
                pts = normalize_rms(pts)
                return -true_score(pts)

            res2 = minimize(
                nm_obj,
                pts1.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            pts2 = normalize_rms(res2.x.reshape(n, d))

            s = true_score(pts2)
            if s > best_score:
                best_score = s
                best_points = pts2.copy()

        except Exception:
            continue

    # Final polish on the best found
    if best_points is not None and time.time() < deadline - 10:
        try:
            def nm_obj_final(x_flat):
                pts = x_flat.reshape(n, d)
                pts = normalize_rms(pts)
                return -true_score(pts)

            # Multiple restarts of NM from best point with tiny perturbations
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                perturbed = best_points + np.random.randn(n, d) * 0.01
                perturbed = normalize_rms(perturbed)
                
                res_final = minimize(
                    nm_obj_final,
                    perturbed.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                )
                pts_final = normalize_rms(res_final.x.reshape(n, d))
                s = true_score(pts_final)
                if s > best_score:
                    best_score = s
                    best_points = pts_final.copy()
        except Exception:
            pass

    # Fallback
    if best_points is None:
        best_points = np.vstack([ico, poles])

    if not np.all(np.isfinite(best_points)):
        best_points = np.vstack([ico, poles])

    return best_points
