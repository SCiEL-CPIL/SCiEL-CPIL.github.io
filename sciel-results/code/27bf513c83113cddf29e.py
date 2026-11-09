import numpy as np
from scipy.spatial.distance import pdist
import time


def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # safety margin before 630s limit
    n = 14

    def to_sphere(pts):
        """Convert 2D spherical coords (theta, phi) to 3D unit sphere points."""
        theta = pts[:, 0]  # polar angle [0, pi]
        phi = pts[:, 1]    # azimuthal angle [0, 2*pi]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack([x, y, z])

    def ratio_from_sphere(pts_3d):
        """Compute d_min/d_max ratio."""
        d = pdist(pts_3d)
        d_min = float(d.min())
        d_max = float(d.max())
        if d_max <= 1e-15:
            return 0.0
        return d_min / d_max

    def neg_ratio_spherical(sph_coords):
        """Negative ratio for minimization, working in spherical coordinates."""
        pts_3d = to_sphere(sph_coords)
        return -ratio_from_sphere(pts_3d)

    def soft_neg_ratio_spherical(sph_coords, beta=50.0):
        """Soft version of the ratio for smoother optimization."""
        pts_3d = to_sphere(sph_coords)
        d = pdist(pts_3d)
        d_min_soft = -np.logaddexp.reduce(-d / beta) * beta
        d_max_soft = np.logaddexp.reduce(d / beta) * beta
        if d_max_soft <= 1e-15:
            return 0.0
        return -(d_min_soft / d_max_soft)

    def icosahedron_14():
        """Create 14 points: 12 icosahedron vertices + 2 poles, in spherical coords."""
        t = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [-1, 1, t], [1, 1, t], [-1, -1, t], [1, -1, t],
            [1, t, -1], [1, t, 1], [-1, t, -1], [-1, t, 1],
            [t, -1, -1], [t, 1, -1], [t, -1, 1], [t, 1, 1]
        ], dtype=float)
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        poles = np.array([[0, 0, 1.0], [0, 0, -1.0]])
        pts = np.vstack([ico, poles])
        # Convert to spherical coordinates
        theta = np.arccos(np.clip(pts[:, 2], -1, 1))
        phi = np.arctan2(pts[:, 1], pts[:, 0])
        phi = np.mod(phi, 2 * np.pi)
        return np.column_stack([theta, phi])

    def fibonacci_sphere():
        """Fibonacci sphere distribution in spherical coordinates."""
        k = np.arange(n)
        phi_angle = np.arccos(1 - 2 * (k + 0.5) / n)
        golden = np.pi * (1 + np.sqrt(5))
        theta_angle = golden * k
        theta_angle = np.mod(theta_angle, 2 * np.pi)
        return np.column_stack([phi_angle, theta_angle])

    def random_sphere(seed=42):
        """Random points on sphere in spherical coordinates."""
        rng = np.random.RandomState(seed)
        theta = rng.uniform(0, np.pi, size=n)
        phi = rng.uniform(0, 2 * np.pi, size=n)
        return np.column_stack([theta, phi])

    def nelder_mead_optimize(sph_coords, max_iter=300):
        """Nelder-Mead optimization in spherical coordinates with wrapping."""
        from scipy.optimize import minimize
        
        def wrapped_neg_ratio(sph):
            # Wrap angles to valid ranges
            sph_wrapped = sph.copy()
            sph_wrapped[:, 0] = np.mod(sph_wrapped[:, 0], np.pi)  # theta in [0, pi]
            sph_wrapped[:, 1] = np.mod(sph_wrapped[:, 1], 2 * np.pi)  # phi in [0, 2*pi]
            return neg_ratio_spherical(sph_wrapped)
        
        try:
            res = minimize(wrapped_neg_ratio, sph_coords, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'initial_simplex': None})
            sph_result = res.x
            # Wrap angles
            sph_result[:, 0] = np.mod(sph_result[:, 0], np.pi)
            sph_result[:, 1] = np.mod(sph_result[:, 1], 2 * np.pi)
            r = -float(res.fun)
            if np.isfinite(r):
                return sph_result, r
        except Exception:
            pass
        return sph_coords, -float('inf')

    def simulated_annealing_spherical(sph_coords, steps=2000, T_start=0.1, T_end=0.001, seed=42):
        """Simulated annealing in spherical coordinates."""
        rng = np.random.RandomState(seed)
        sph = sph_coords.copy()
        current_score = -neg_ratio_spherical(sph)
        best_sph = sph.copy()
        best_score = current_score
        
        for step in range(steps):
            if time.time() > deadline - 20:
                break
            T = T_start * (T_end / T_start) ** (step / max(steps - 1, 1))
            
            # Propose move: perturb 1-2 points' spherical coordinates
            n_perturb = rng.choice([1, 2], p=[0.7, 0.3])
            indices = rng.choice(n, size=n_perturb, replace=False)
            
            new_sph = sph.copy()
            for idx in indices:
                # Perturb theta and phi
                dtheta = rng.randn() * 0.05 * (T / T_start + 0.1)
                dphi = rng.randn() * 0.05 * (T / T_start + 0.1)
                new_sph[idx, 0] += dtheta
                new_sph[idx, 1] += dphi
                # Wrap
                new_sph[idx, 0] = np.mod(new_sph[idx, 0], np.pi)
                new_sph[idx, 1] = np.mod(new_sph[idx, 1], 2 * np.pi)
            
            new_score = -neg_ratio_spherical(new_sph)
            
            if new_score > current_score or rng.rand() < np.exp((new_score - current_score) / max(T, 1e-10)):
                sph = new_sph
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_sph = sph.copy()
        
        return best_sph, best_score

    # Generate diverse initial candidates in spherical coordinates
    np.random.seed(12345)
    candidates = [
        icosahedron_14(),
        fibonacci_sphere(),
        random_sphere(seed=1),
        random_sphere(seed=2),
        random_sphere(seed=3),
        random_sphere(seed=4),
        random_sphere(seed=5),
    ]
    
    # Add perturbed versions of structured starts
    for base in [icosahedron_14(), fibonacci_sphere()]:
        for noise_level in [0.05, 0.1, 0.2]:
            perturbed = base + noise_level * np.random.randn(n, 2)
            perturbed[:, 0] = np.mod(perturbed[:, 0], np.pi)
            perturbed[:, 1] = np.mod(perturbed[:, 1], 2 * np.pi)
            candidates.append(perturbed)
    
    best_sph = None
    best_r = -1.0

    # Phase 1: Simulated Annealing from multiple starts
    for i, cand in enumerate(candidates[:8]):
        if time.time() > deadline - 200:
            break
        try:
            sa_sph, sa_r = simulated_annealing_spherical(cand, steps=1500, T_start=0.05, T_end=0.001, seed=100+i)
            if np.isfinite(sa_r) and sa_r > best_r:
                best_r = sa_r
                best_sph = sa_sph
        except Exception:
            pass

    # Phase 2: Nelder-Mead from best SA result and original candidates
    opt_candidates = []
    if best_sph is not None:
        opt_candidates.append(best_sph)
    opt_candidates.extend(candidates[:4])
    
    for i, cand in enumerate(opt_candidates):
        if time.time() > deadline - 120:
            break
        try:
            nm_sph, nm_r = nelder_mead_optimize(cand, max_iter=200)
            if np.isfinite(nm_r) and nm_r > best_r:
                best_r = nm_r
                best_sph = nm_sph
        except Exception:
            pass

    # Phase 3: Soft objective refinement
    if best_sph is not None and time.time() < deadline - 80:
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 60:
                break
            try:
                from scipy.optimize import minimize
                res_soft = minimize(lambda x: soft_neg_ratio_spherical(x, beta),
                                   best_sph, method='Nelder-Mead',
                                   options={'maxiter': 150, 'xatol': 1e-9, 'fatol': 1e-11})
                sph_soft = res_soft.x
                sph_soft[:, 0] = np.mod(sph_soft[:, 0], np.pi)
                sph_soft[:, 1] = np.mod(sph_soft[:, 1], 2 * np.pi)
                r_exact = -neg_ratio_spherical(sph_soft)
                if np.isfinite(r_exact) and r_exact > best_r:
                    best_r = r_exact
                    best_sph = sph_soft
            except Exception:
                pass

    # Phase 4: Final polish with high-precision Nelder-Mead
    if best_sph is not None and time.time() < deadline - 30:
        try:
            nm_sph, nm_r = nelder_mead_optimize(best_sph, max_iter=400)
            if np.isfinite(nm_r) and nm_r > best_r:
                best_r = nm_r
                best_sph = nm_sph
        except Exception:
            pass

    # Phase 5: One more SA round from best
    if best_sph is not None and time.time() < deadline - 10:
        try:
            sa_sph2, sa_r2 = simulated_annealing_spherical(best_sph, steps=800, T_start=0.02, T_end=0.0005, seed=999)
            if np.isfinite(sa_r2) and sa_r2 > best_r:
                best_r = sa_r2
                best_sph = sa_sph2
        except Exception:
            pass

    if best_sph is None:
        best_sph = fibonacci_sphere()

    # Convert final spherical coordinates to 3D Cartesian
    best_pts_3d = to_sphere(best_sph)
    best_pts_3d = np.nan_to_num(best_pts_3d, nan=0.0, posinf=0.0, neginf=0.0)
    
    return best_pts_3d
