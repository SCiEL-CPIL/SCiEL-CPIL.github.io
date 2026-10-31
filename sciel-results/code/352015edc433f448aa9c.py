"""Combined program: SA + soft objective refinement + particle equilibrium + diverse seeds."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def get_sq_dists(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return sq_dists[iu]

    def score_raw(pts):
        d_sq = get_sq_dists(pts)
        d_min = np.sqrt(d_sq.min())
        d_max = np.sqrt(d_sq.max())
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def soft_score(pts, beta):
        """Smooth surrogate using log-sum-exp for min and max distances."""
        d_sq = get_sq_dists(pts)
        d_sq = np.maximum(d_sq, 1e-12)
        max_d_sq = np.max(d_sq)
        soft_max = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
        min_d_sq = np.min(d_sq)
        soft_min = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
        if soft_min < 1e-8:
            return 1e10
        return soft_max / soft_min

    def normalize(pts):
        pts = pts - np.mean(pts, axis=0)
        scale = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if scale > 1e-9:
            pts = pts / scale
        return pts

    def simulated_annealing(init_pts, max_iters=20000, T_start=1.0, T_end=0.001, alpha=0.99995):
        pts = init_pts.copy()
        current_score = score_raw(pts)
        best_pts = pts.copy()
        best_score = current_score
        T = T_start
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            idx = np.random.randint(0, N)
            displacement = np.random.normal(0, 0.1, 2)
            new_pts = pts.copy()
            new_pts[idx] += displacement
            new_pts = normalize(new_pts)
            
            new_score = score_raw(new_pts)
            
            if new_score < current_score or np.random.random() < np.exp(-(new_score - current_score) / T):
                current_score = new_score
                pts = new_pts
                if new_score < best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
            
            T *= alpha
            
        return best_pts, best_score

    def particle_equilibrium_seed(rng, repulsion_strength=1.0, boundary_radius=2.0, n_iters=500):
        """Generate seed from particle equilibrium with short-range repulsion."""
        pts = rng.uniform(-boundary_radius, boundary_radius, (N, 2))
        
        for _ in range(n_iters):
            if time.time() > deadline - 60:
                break
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i+1, N):
                    diff = pts[i] - pts[j]
                    dist_sq = np.sum(diff**2)
                    dist = np.sqrt(dist_sq) + 1e-10
                    # Short-range repulsion: 1/r^4
                    force_mag = repulsion_strength / (dist_sq**2 + 0.01)
                    force = force_mag * diff / dist
                    forces[i] += force
                    forces[j] -= force
            
            # Boundary pressure: push points inward if too far
            for i in range(N):
                r = np.linalg.norm(pts[i])
                if r > boundary_radius:
                    forces[i] -= 0.5 * (r - boundary_radius) * pts[i] / r
            
            # Update positions with damping
            pts += 0.01 * forces
            # Keep within bounds
            pts = np.clip(pts, -boundary_radius*1.5, boundary_radius*1.5)
        
        return normalize(pts)

    def generate_seeds():
        """Generate diverse initial configurations."""
        seeds = []
        rng = np.random.default_rng(42)
        
        # 1. Random points
        for _ in range(2):
            pts = rng.uniform(-2, 2, (N, 2))
            seeds.append(normalize(pts))
        
        # 2. Perturbed 4x4 grid
        x = np.arange(4) - 1.5
        y = np.arange(4) - 1.5
        xx, yy = np.meshgrid(x, y)
        base_grid = np.column_stack([xx.ravel(), yy.ravel()])
        for _ in range(2):
            noise = rng.normal(0, 0.2, (N, 2))
            seeds.append(normalize(base_grid + noise))
        
        # 3. Ring configurations
        for r_in in [0.5, 0.65]:
            for r_out in [1.3, 1.5]:
                angles_in = np.linspace(0, 2*np.pi, 8, endpoint=False) + rng.uniform(0, 2*np.pi)
                angles_out = np.linspace(0, 2*np.pi, 8, endpoint=False) + rng.uniform(0, 2*np.pi)
                pts_in = np.column_stack([r_in*np.cos(angles_in), r_in*np.sin(angles_in)])
                pts_out = np.column_stack([r_out*np.cos(angles_out), r_out*np.sin(angles_out)])
                seeds.append(normalize(np.vstack([pts_in, pts_out])))
        
        # 4. Hexagonal lattice subset
        hex_pts = []
        R = 4
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                x = q + 0.5 * r
                y = (np.sqrt(3) / 2) * r
                hex_pts.append((x, y))
        hex_pts = np.array(hex_pts)
        dists = np.sum(hex_pts**2, axis=1)
        idx = np.argsort(dists)[:16]
        hex_seed = hex_pts[idx]
        for _ in range(2):
            seeds.append(normalize(hex_seed + rng.normal(0, 0.1, size=(16, 2))))
        
        # 5. Particle equilibrium seeds with different parameters
        for rep_str in [0.5, 1.0, 2.0]:
            for bound_r in [1.5, 2.0, 2.5]:
                seed = particle_equilibrium_seed(rng, rep_str, bound_r, n_iters=300)
                seeds.append(seed)
        
        return seeds

    # Generate seeds
    all_seeds = generate_seeds()
    
    # Score all seeds
    scored_seeds = [(score_raw(s), s) for s in all_seeds]
    scored_seeds.sort(key=lambda x: x[0])
    
    best_overall = None
    best_overall_score = 1e10
    
    # Process top seeds with SA + soft refinement
    num_to_process = min(8, len(scored_seeds))
    
    for i in range(num_to_process):
        if time.time() > deadline - 35:
            break
        
        curr_score, pts = scored_seeds[i]
        if curr_score < best_overall_score:
            best_overall_score = curr_score
            best_overall = pts.copy()
        
        # Phase 1: Simulated Annealing
        sa_pts, sa_score = simulated_annealing(pts, max_iters=12000)
        if sa_score < best_overall_score:
            best_overall_score = sa_score
            best_overall = sa_pts.copy()
        
        # Phase 2: Soft objective refinement with adaptive beta
        if time.time() < deadline - 20:
            try:
                def soft_obj(x, beta=80.0):
                    return soft_score(x.reshape(N, 2), beta)
                
                res1 = minimize(soft_obj, sa_pts.ravel(), method='L-BFGS-B',
                               options={'maxiter': 400, 'ftol': 1e-13})
                exact_score = score_raw(res1.x.reshape(N, 2))
                if exact_score < best_overall_score:
                    best_overall_score = exact_score
                    best_overall = res1.x.reshape(N, 2).copy()
            except Exception:
                pass
        
        # Phase 3: Powell on exact objective
        if time.time() < deadline - 12:
            try:
                def exact_obj(x):
                    return score_raw(x.reshape(N, 2))
                
                res2 = minimize(exact_obj, best_overall.ravel(), method='Powell',
                               options={'maxiter': 300, 'maxfev': 3000})
                if res2.fun < best_overall_score:
                    best_overall_score = res2.fun
                    best_overall = res2.x.reshape(N, 2).copy()
            except Exception:
                pass
        
        # Phase 4: Higher beta soft refinement
        if time.time() < deadline - 6:
            try:
                def soft_obj_high(x, beta=300.0):
                    return soft_score(x.reshape(N, 2), beta)
                
                res3 = minimize(soft_obj_high, best_overall.ravel(), method='L-BFGS-B',
                               options={'maxiter': 600, 'ftol': 1e-15})
                exact_score = score_raw(res3.x.reshape(N, 2))
                if exact_score < best_overall_score:
                    best_overall_score = exact_score
                    best_overall = res3.x.reshape(N, 2).copy()
            except Exception:
                pass
    
    # Final high-precision polish
    if best_overall is not None and time.time() < deadline - 2:
        try:
            def soft_obj_final(x, beta=1000.0):
                return soft_score(x.reshape(N, 2), beta)
            
            res_final = minimize(soft_obj_final, best_overall.ravel(), method='L-BFGS-B',
                                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14})
            exact_score = score_raw(res_final.x.reshape(N, 2))
            if exact_score < best_overall_score:
                best_overall_score = exact_score
                best_overall = res_final.x.reshape(N, 2).copy()
        except Exception:
            pass
    
    # Ultra-fine SLSQP polish
    if best_overall is not None and time.time() < deadline - 1:
        try:
            def exact_obj_slsqp(x):
                return score_raw(x.reshape(N, 2))
            
            res_slsqp = minimize(exact_obj_slsqp, best_overall.ravel(), method='SLSQP',
                                options={'maxiter': 500, 'ftol': 1e-15})
            exact_score = score_raw(res_slsqp.x.reshape(N, 2))
            if exact_score < best_overall_score:
                best_overall_score = exact_score
                best_overall = res_slsqp.x.reshape(N, 2).copy()
        except Exception:
            pass
    
    if best_overall is None:
        return all_seeds[0]
    
    # Verify distinctness
    d_check = pdist(best_overall, 'euclidean')
    if np.min(d_check) < 1e-10:
        return all_seeds[0]
    
    return best_overall

# EVOLVE-BLOCK-END
