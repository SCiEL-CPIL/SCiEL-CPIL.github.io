"""Program for packing 50 equal circles in a unit square; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Rigid-body assembly: circle centers as nodes in a contact graph.
    Solve for optimal radius via iterative projection with adaptive
    contact graph updates and momentum-based dynamics.
    """
    
    def hex_lattice(n):
        """Generate hexagonal lattice points centered in [-0.5, 0.5]^2."""
        pts = []
        for row in range(15):
            for col in range(15):
                x = col + 0.5 * (row % 2)
                y = row * np.sqrt(3) / 2.0
                pts.append([x, y])
        pts = np.array(pts[:n])
        pts -= pts.mean(axis=0)
        max_abs = np.max(np.abs(pts))
        pts *= (0.45 / max_abs)
        return pts
    
    def grid_layout(n):
        """Square grid layout."""
        m = int(np.ceil(np.sqrt(n)))
        xs, ys = np.meshgrid(np.linspace(-0.45, 0.45, m), np.linspace(-0.45, 0.45, m))
        pts = np.column_stack([xs.ravel(), ys.ravel()])[:n]
        return pts
    
    def random_layout(n, seed):
        """Random layout with Poisson-disk-like spacing."""
        np.random.seed(seed)
        pts = np.random.uniform(-0.45, 0.45, (n, 2))
        return pts
    
    def dist_matrix(pts):
        """Compute pairwise distance matrix."""
        d = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(d**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        return dist
    
    def packing_radius(pts):
        """Compute the feasible packing radius."""
        dm = dist_matrix(pts)
        r_pair = np.min(dm) / 2.0
        r_bound = 0.5 - np.max(np.abs(pts), axis=1)
        r_bound = np.min(r_bound)
        return min(r_pair, r_bound)
    
    def find_contacts(pts, k=7):
        """Identify contact edges: k nearest neighbors per node."""
        dm = dist_matrix(pts)
        n = len(pts)
        contacts = set()
        for i in range(n):
            nn = np.argsort(dm[i])[:k]
            for j in nn:
                a, b = min(int(i), int(j)), max(int(i), int(j))
                contacts.add((a, b))
        return list(contacts)
    
    def project_constraints(pos, r, contacts, n_steps=2000, momentum=0.85, step_scale=0.15):
        """
        Iteratively project positions to satisfy:
        - dist(i,j) >= 2r for all contact pairs
        - pos[i] in [-0.5, 0.5]^2
        Uses momentum to help escape local minima.
        """
        pos = pos.copy().astype(np.float64)
        vel = np.zeros_like(pos)
        n = len(pos)
        contact_set = set(contacts)
        
        for step in range(n_steps):
            total_correction = 0.0
            
            # Enforce inter-circle distance constraints
            for (i, j) in contacts:
                d = pos[j] - pos[i]
                dist = np.linalg.norm(d)
                min_dist = 2.0 * r
                
                if dist < min_dist:
                    if dist > 1e-12:
                        error = min_dist - dist
                        direction = d / dist
                        corr = direction * error * 0.5
                        pos[i] -= corr
                        pos[j] += corr
                        vel[i] -= corr * 0.5
                        vel[j] += corr * 0.5
                        total_correction += error
                    else:
                        # Coincident: push apart randomly
                        angle = np.random.uniform(0, 2 * np.pi)
                        offset = np.array([np.cos(angle), np.sin(angle)]) * r * 0.5
                        pos[i] -= offset
                        pos[j] += offset
                        vel[i] -= offset * 0.5
                        vel[j] += offset * 0.5
                        total_correction += r
            
            # Enforce boundary constraints
            for i in range(n):
                for c in range(2):
                    if pos[i, c] < -0.5:
                        pos[i, c] = -0.5
                        vel[i, c] = 0.0
                    elif pos[i, c] > 0.5:
                        pos[i, c] = 0.5
                        vel[i, c] = 0.0
            
            # Update velocity and position
            vel *= momentum
            pos += vel * step_scale
            pos = np.clip(pos, -0.5, 0.5)
            
            if total_correction < 1e-10 and np.max(np.abs(vel)) < 1e-10:
                break
        
        return pos
    
    def refine_with_adaptive_contacts(pos, r, n_rounds=30, n_steps_per_round=1500):
        """
        Iteratively refine: update contact graph based on current positions,
        then project. This helps escape local minima.
        """
        pos = pos.copy()
        
        for round_idx in range(n_rounds):
            # Rebuild contact graph based on current positions
            k = 6 + (round_idx % 3)  # Vary k: 6, 7, 8
            contacts = find_contacts(pos, k=k)
            
            # Project with current contacts
            pos = project_constraints(pos, r, contacts, n_steps=n_steps_per_round)
            
            # Check if we're still feasible
            dm = dist_matrix(pos)
            if np.min(dm) < 2.0 * r - 1e-7:
                # Infeasible, revert and break
                break
            
            if np.any(np.abs(pos) > 0.5 + 1e-7):
                break
        
        return pos
    
    def attempt_radius(pts, r, n_rounds=20, n_steps=1000):
        """Attempt to find feasible configuration for radius r."""
        pos = pts.copy()
        pos = refine_with_adaptive_contacts(pos, r, n_rounds=n_rounds, n_steps_per_round=n_steps)
        
        # Verify
        dm = dist_matrix(pos)
        if np.min(dm) < 2.0 * r - 1e-7:
            return None, False
        if np.any(np.abs(pos) > 0.5 + 1e-7):
            return None, False
        if np.any(~np.isfinite(pos)):
            return None, False
        return pos, True
    
    # Generate diverse initial configurations
    np.random.seed(12345)
    configs = []
    
    # Hex lattice variants
    configs.append(hex_lattice(N))
    configs.append(hex_lattice(N) + np.random.randn(N, 2) * 0.01)
    
    # Grid variants
    configs.append(grid_layout(N))
    configs.append(grid_layout(N) + np.random.randn(N, 2) * 0.015)
    
    # Random layouts
    for seed in [42, 137, 256]:
        configs.append(random_layout(N, seed))
    
    # Scaled hex with different aspect
    pts_hex = hex_lattice(N)
    pts_hex[:, 0] *= 1.1
    pts_hex[:, 1] *= 0.95
    max_abs = np.max(np.abs(pts_hex))
    if max_abs > 0.48:
        pts_hex *= 0.48 / max_abs
    configs.append(pts_hex)
    
    best_pos = None
    best_r = 0.0
    start_time = time.time()
    time_budget = 110.0  # Leave 10s margin
    
    for cfg_idx, cfg in enumerate(configs):
        if time.time() - start_time > time_budget:
            break
        
        cfg = np.clip(cfg, -0.49, 0.49)
        r0 = packing_radius(cfg)
        
        # Initial refinement at r0
        contacts = find_contacts(cfg, k=7)
        pos_imp = project_constraints(cfg, r0, contacts, n_steps=2000)
        r_imp = packing_radius(pos_imp)
        
        if r_imp > r0:
            cfg = pos_imp
            r0 = r_imp
        
        # Binary search for max radius
        r_lo = r0 * 0.98
        r_hi = r0 * 1.20
        
        for bs_iter in range(20):
            if time.time() - start_time > time_budget:
                break
            
            r_mid = (r_lo + r_hi) / 2.0
            pos_try, feasible = attempt_radius(cfg, r_mid, n_rounds=15, n_steps=800)
            
            if feasible:
                r_lo = r_mid
                if r_mid > best_r:
                    best_r = r_mid
                    best_pos = pos_try.copy()
            else:
                r_hi = r_mid
            
            if r_hi - r_lo < 1e-9:
                break
    
    # Final refinement on best configuration
    if best_pos is not None and time.time() - start_time < time_budget:
        r_cur = packing_radius(best_pos)
        # More aggressive refinement
        pos_final = refine_with_adaptive_contacts(best_pos, r_cur, n_rounds=40, n_steps_per_round=2000)
        r_final = packing_radius(pos_final)
        
        if r_final > best_r:
            best_r = r_final
            best_pos = pos_final
    
    # Safety fallback
    if best_pos is None or np.any(~np.isfinite(best_pos)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pos = grid[:N]
    
    best_pos = np.clip(best_pos, -0.5, 0.5)
    
    # Ensure no coincident centers
    dm = dist_matrix(best_pos)
    if np.min(dm) < 1e-10:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pos = grid[:N]
    
    return best_pos
# EVOLVE-BLOCK-END
