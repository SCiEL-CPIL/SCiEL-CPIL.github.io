import numpy as np
from scipy.spatial import Delaunay, Voronoi
from scipy.optimize import minimize
import time

def construct():
    N = 26
    deadline = time.time() + 140  # Safety margin for 150s limit
    rng = np.random.default_rng(12345)
    
    def is_valid(pos, radii):
        """Check if configuration is valid (no overlaps, all in unit square)"""
        for i in range(N):
            if pos[i][0] - radii[i] < -1e-8 or pos[i][0] + radii[i] > 1 + 1e-8:
                return False
            if pos[i][1] - radii[i] < -1e-8 or pos[i][1] + radii[i] > 1 + 1e-8:
                return False
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < radii[i] + radii[j] - 1e-8:
                    return False
        return True
    
    def get_score(pos, radii):
        if not is_valid(pos, radii):
            return -np.inf
        return np.sum(radii)
    
    def max_valid_radius(pos, i):
        """Find maximum valid radius for circle i given positions"""
        r_max = min(pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
        for j in range(N):
            if i == j:
                continue
            d = np.linalg.norm(pos[i] - pos[j])
            r_max = min(r_max, max(0, d - 0.01))  # 0.01 buffer
        return max(0.001, r_max)
    
    def compute_max_radii(pos):
        """Compute maximum valid radii for all circles given positions"""
        radii = np.zeros(N)
        for i in range(N):
            radii[i] = max_valid_radius(pos, i)
        return radii
    
    def force_directed_step(pos, radii, dt=0.001, iterations=50):
        """Apply force-directed relaxation to resolve overlaps and boundaries"""
        pos = pos.copy()
        for _ in range(iterations):
            forces = np.zeros_like(pos)
            
            # Boundary forces
            for i in range(N):
                if pos[i][0] < radii[i]:
                    forces[i][0] += (radii[i] - pos[i][0]) * 10
                elif pos[i][0] > 1 - radii[i]:
                    forces[i][0] -= (pos[i][0] + radii[i] - 1) * 10
                if pos[i][1] < radii[i]:
                    forces[i][1] += (radii[i] - pos[i][1]) * 10
                elif pos[i][1] > 1 - radii[i]:
                    forces[i][1] -= (pos[i][1] + radii[i] - 1) * 10
            
            # Overlap forces
            for i in range(N):
                for j in range(i+1, N):
                    d_vec = pos[i] - pos[j]
                    d = np.linalg.norm(d_vec)
                    if d < 1e-10:
                        continue
                    overlap = radii[i] + radii[j] - d
                    if overlap > 0:
                        force_mag = overlap * 5
                        forces[i] += (d_vec / d) * force_mag
                        forces[j] -= (d_vec / d) * force_mag
            
            # Update positions
            pos += forces * dt
            pos = np.clip(pos, 0.001, 0.999)
        
        return pos
    
    def voronoi_based_radius(pos):
        """Use Voronoi diagram to estimate maximum radii"""
        radii = np.zeros(N)
        try:
            vor = Voronoi(pos)
            for i in range(N):
                # Find region vertices
                region_idx = vor.point_region[i]
                region = vor.regions[region_idx]
                if len(region) == 0:
                    # Point on convex hull, use distance to boundary
                    radii[i] = min(pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
                else:
                    vertices = vor.vertices[region]
                    # Distance to each edge of the Voronoi cell
                    min_dist = np.inf
                    for k in range(len(vertices)):
                        v1 = vertices[k]
                        v2 = vertices[(k+1) % len(vertices)]
                        # Distance from pos[i] to line segment v1-v2
                        d_vec = v2 - v1
                        d_len = np.linalg.norm(d_vec)
                        if d_len < 1e-10:
                            continue
                        t = np.dot(pos[i] - v1, d_vec) / (d_len * d_len)
                        t = np.clip(t, 0, 1)
                        closest = v1 + t * d_vec
                        dist = np.linalg.norm(pos[i] - closest)
                        min_dist = min(min_dist, dist)
                    
                    # Also consider distance to square boundaries
                    min_dist = min(min_dist, pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
                    radii[i] = max(0.001, min_dist * 0.99)
        except Exception:
            # Fallback to simple distance-based
            radii = compute_max_radii(pos)
        
        return radii
    
    def simulated_annealing(pos, radii, steps=2000, T_start=0.05, T_end=0.001):
        """Simulated annealing to explore configuration space"""
        best_pos = pos.copy()
        best_radii = radii.copy()
        best_score = get_score(pos, radii)
        
        current_score = best_score
        T = T_start
        
        for step in range(steps):
            T = T_start * (T_end / T_start) ** (step / max(1, steps - 1))
            
            # Propose move: perturb one or two positions
            n_perturb = rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            indices = rng.choice(N, size=n_perturb, replace=False)
            delta = rng.normal(0, 0.03 * T / T_start + 0.005, size=(n_perturb, 2))
            
            new_pos = pos.copy()
            new_pos[indices] += delta
            new_pos = np.clip(new_pos, 0.001, 0.999)
            
            new_radii = compute_max_radii(new_pos)
            
            new_score = get_score(new_pos, new_radii)
            
            if new_score > current_score or (new_score > -np.inf and rng.random() < np.exp((new_score - current_score) / T)):
                pos = new_pos
                radii = new_radii
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_pos = pos.copy()
                    best_radii = radii.copy()
        
        return best_pos, best_radii
    
    def refine_with_lbfgs(pos, radii):
        """Final refinement using L-BFGS on a penalty objective"""
        def penalty_obj(x):
            p = x[:2*N].reshape(N, 2)
            r = x[2*N:]
            penalty = 0.0
            
            # Boundary penalties
            for i in range(N):
                if p[i][0] < r[i]:
                    penalty += (r[i] - p[i][0])**2
                if p[i][0] > 1 - r[i]:
                    penalty += (p[i][0] + r[i] - 1)**2
                if p[i][1] < r[i]:
                    penalty += (r[i] - p[i][1])**2
                if p[i][1] > 1 - r[i]:
                    penalty += (p[i][1] + r[i] - 1)**2
            
            # Overlap penalties
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(p[i] - p[j])
                    overlap = r[i] + r[j] - d
                    if overlap > 0:
                        penalty += overlap**2
            
            return penalty - np.sum(r)
        
        x0 = np.concatenate([pos.flatten(), radii])
        bounds = [(0.001, 0.999)] * (2*N) + [(0.001, 0.5)] * N
        
        try:
            result = minimize(penalty_obj, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 50, 'gtol': 1e-6})
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = x_opt[2*N:]
            
            if is_valid(pos_opt, radii_opt):
                return pos_opt, radii_opt
        except Exception:
            pass
        
        return pos, radii
    
    # === Main algorithm ===
    
    # Phase 1: Multiple initial configurations
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    initial_configs = []
    
    # Config 1: Hexagonal grid
    cols, rows = 6, 5
    xs = np.linspace(1/(2*cols), 1 - 1/(2*cols), cols)
    ys = np.linspace(1/(2*rows), 1 - 1/(2*rows), rows)
    centers = []
    for x in xs:
        for y in ys:
            centers.append([x, y])
    centers = np.array(centers)
    d2c = np.sum((centers - 0.5)**2, axis=1)
    idx = np.argsort(d2c)[:N]
    initial_configs.append(centers[idx].copy())
    
    # Config 2: Random with jitter
    pos_rand = rng.uniform(0.1, 0.9, size=(N, 2))
    initial_configs.append(pos_rand)
    
    # Config 3: Structured 5x5 + 1
    xs5 = np.linspace(0.1, 0.9, 5)
    ys5 = np.linspace(0.1, 0.9, 5)
    centers5 = []
    for x in xs5:
        for y in ys5:
            centers5.append([x, y])
    centers5 = np.array(centers5)
    # Add one more at center
    centers5 = np.vstack([centers5, [[0.5, 0.5]]])
    d2c5 = np.sum((centers5 - 0.5)**2, axis=1)
    idx5 = np.argsort(d2c5)[:N]
    initial_configs.append(centers5[idx5].copy())
    
    # Config 4: Force-directed from random
    pos_fd = rng.uniform(0.15, 0.85, size=(N, 2))
    r_fd = np.full(N, 0.05)
    pos_fd = force_directed_step(pos_fd, r_fd, dt=0.002, iterations=100)
    initial_configs.append(pos_fd)
    
    # Evaluate each initial config
    for config in initial_configs:
        pos_init = config.copy()
        radii_init = compute_max_radii(pos_init)
        
        # Force-directed relaxation
        pos_relaxed = force_directed_step(pos_init, radii_init, dt=0.001, iterations=200)
        radii_relaxed = compute_max_radii(pos_relaxed)
        
        # Voronoi-based radius estimation
        radii_vor = voronoi_based_radius(pos_relaxed)
        # Use the more conservative estimate
        radii_combined = np.minimum(radii_relaxed, radii_vor)
        
        score = get_score(pos_relaxed, radii_combined)
        if score > best_score:
            best_score = score
            best_pos = pos_relaxed.copy()
            best_radii = radii_combined.copy()
    
    # Phase 2: Simulated annealing from best config
    if time.time() < deadline - 40:
        sa_pos, sa_radii = simulated_annealing(
            best_pos, best_radii, 
            steps=3000, T_start=0.04, T_end=0.0005
        )
        sa_score = get_score(sa_pos, sa_radii)
        if sa_score > best_score:
            best_score = sa_score
            best_pos = sa_pos.copy()
            best_radii = sa_radii.copy()
    
    # Phase 3: Force-directed refinement
    if time.time() < deadline - 20:
        pos_refined = force_directed_step(best_pos, best_radii, dt=0.0005, iterations=500)
        radii_refined = compute_max_radii(pos_refined)
        radii_vor2 = voronoi_based_radius(pos_refined)
        radii_final = np.minimum(radii_refined, radii_vor2)
        
        score_refined = get_score(pos_refined, radii_final)
        if score_refined > best_score:
            best_score = score_refined
            best_pos = pos_refined.copy()
            best_radii = radii_final.copy()
    
    # Phase 4: L-BFGS polish
    if time.time() < deadline - 10:
        pos_lbfgs, radii_lbfgs = refine_with_lbfgs(best_pos, best_radii)
        score_lbfgs = get_score(pos_lbfgs, radii_lbfgs)
        if score_lbfgs > best_score:
            best_score = score_lbfgs
            best_pos = pos_lbfgs.copy()
            best_radii = radii_lbfgs.copy()
    
    # Phase 5: Final radius maximization
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        new_radii = compute_max_radii(best_pos)
        new_radii = np.minimum(new_radii, best_radii * 1.01)  # Don't increase too much
        if is_valid(best_pos, new_radii):
            best_radii = new_radii
    
    # Final validation
    if not is_valid(best_pos, best_radii):
        scale = 1.0
        while scale > 0.01 and not is_valid(best_pos, best_radii * scale):
            scale *= 0.95
        best_radii = best_radii * scale
    
    return np.column_stack((best_pos, best_radii))
