# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    def score(points):
        """Calculate R = (d_min/d_max)^2"""
        dists = pdist(points, 'euclidean')
        d_min = dists.min()
        d_max = dists.max()
        if d_max < 1e-10:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def smooth_score(points, beta=50.0):
        """Smooth approximation of R using soft min/max for better gradients"""
        dists = pdist(points, 'euclidean')
        if dists.max() < 1e-10:
            return 0.0
        # Soft min: -log(sum(exp(-beta*d)))/beta
        soft_min = -np.log(np.sum(np.exp(-beta * dists))) / beta
        # Soft max: log(sum(exp(beta*d)))/beta
        soft_max = np.log(np.sum(np.exp(beta * dists))) / beta
        if soft_max < 1e-10:
            return 0.0
        ratio = soft_min / soft_max
        return ratio * ratio

    def objective(params, beta=50.0):
        """Minimize -R (smooth version for gradient-based optimization)"""
        points = params.reshape((n, 2))
        return -smooth_score(points, beta)

    def generate_initializations():
        """Generate diverse initial configurations"""
        inits = []
        
        # 4x4 grid
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        X, Y = np.meshgrid(x, y)
        inits.append(np.column_stack([X.flatten(), Y.flatten()]))
        
        # Circular arrangement: 4 points on inner circle, 12 on outer
        angles_inner = np.linspace(0, 2*np.pi, 4, endpoint=False)
        inner_r = 0.5
        inner_pts = np.column_stack([inner_r*np.cos(angles_inner), inner_r*np.sin(angles_inner)])
        
        angles_outer = np.linspace(0, 2*np.pi, 12, endpoint=False)
        outer_r = 1.5
        outer_pts = np.column_stack([outer_r*np.cos(angles_outer), outer_r*np.sin(angles_outer)])
        inits.append(np.vstack([inner_pts, outer_pts]))
        
        # Two concentric squares
        angles_sq1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        r1 = 0.6
        sq1 = np.column_stack([r1*np.cos(angles_sq1), r1*np.sin(angles_sq1)])
        
        angles_sq2 = angles_sq1 + np.pi/4
        r2 = 1.4
        sq2 = np.column_stack([r2*np.cos(angles_sq2), r2*np.sin(angles_sq2)])
        
        # Add 8 more points on middle circle
        angles_mid = np.linspace(0, 2*np.pi, 8, endpoint=False)
        r_mid = 1.0
        mid_pts = np.column_stack([r_mid*np.cos(angles_mid), r_mid*np.sin(angles_mid)])
        
        inits.append(np.vstack([sq1, sq2, mid_pts]))
        
        # Known good structure: center + 3 concentric rings
        # Ring 1: 1 point at center
        center = np.array([[0.0, 0.0]])
        # Ring 2: 5 points
        angles2 = np.linspace(0, 2*np.pi, 5, endpoint=False)
        r2 = 0.7
        ring2 = np.column_stack([r2*np.cos(angles2), r2*np.sin(angles2)])
        # Ring 3: 5 points
        angles3 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
        r3 = 1.3
        ring3 = np.column_stack([r3*np.cos(angles3), r3*np.sin(angles3)])
        # Ring 4: 5 points
        angles4 = np.linspace(0, 2*np.pi, 5, endpoint=False)
        r4 = 1.9
        ring4 = np.column_stack([r4*np.cos(angles4), r4*np.sin(angles4)])
        inits.append(np.vstack([center, ring2, ring3, ring4]))
        
        # Another structured init: hexagonal-like packing
        # 1 center, 6 in hex ring, 9 in outer
        center2 = np.array([[0.0, 0.0]])
        hex_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        hex_r = 0.8
        hex_ring = np.column_stack([hex_r*np.cos(hex_angles), hex_r*np.sin(hex_angles)])
        outer_angles = np.linspace(0, 2*np.pi, 9, endpoint=False)
        outer_r2 = 1.6
        outer_ring = np.column_stack([outer_r2*np.cos(outer_angles), outer_r2*np.sin(outer_angles)])
        inits.append(np.vstack([center2, hex_ring, outer_ring]))
        
        # Random with symmetry enforcement
        np.random.seed(42)
        for _ in range(8):
            pts = np.random.randn(n, 2) * 1.0
            inits.append(pts)
        
        # Structured init: hexagonal lattice pattern
        # 1 center, 6 in first ring, 9 in second ring
        center_hex = np.array([[0.0, 0.0]])
        hex_ring1_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        hex_r1 = 0.7
        hex_ring1 = np.column_stack([hex_r1*np.cos(hex_ring1_angles), 
                                     hex_r1*np.sin(hex_ring1_angles)])
        hex_ring2_angles = np.linspace(0, 2*np.pi, 9, endpoint=False) + np.pi/9
        hex_r2 = 1.4
        hex_ring2 = np.column_stack([hex_r2*np.cos(hex_ring2_angles), 
                                     hex_r2*np.sin(hex_ring2_angles)])
        inits.append(np.vstack([center_hex, hex_ring1, hex_ring2]))
        
        # Structured init: two interlocking squares (rotated 45 degrees)
        angles_sq_a = np.linspace(0, 2*np.pi, 8, endpoint=False)
        r_sq_a = 1.0
        sq_a_pts = np.column_stack([r_sq_a*np.cos(angles_sq_a), 
                                    r_sq_a*np.sin(angles_sq_a)])
        angles_sq_b = angles_sq_a + np.pi/8
        r_sq_b = 0.5
        sq_b_pts = np.column_stack([r_sq_b*np.cos(angles_sq_b), 
                                    r_sq_b*np.sin(angles_sq_b)])
        inits.append(np.vstack([sq_a_pts, sq_b_pts]))
        
        return inits

    # Generate and evaluate initial configurations
    inits = generate_initializations()
    best_points = inits[0].copy()
    best_score = score(best_points)
    
    for init in inits[1:]:
        s = score(init)
        if s > best_score:
            best_score = s
            best_points = init.copy()

    # Optimize from multiple starting points
    np.random.seed(42)
    n_trials = 60
    trial_count = 0
    
    for trial in range(n_trials):
        if time.time() > deadline - 120:
            break
        
        # Alternate between perturbations of best and random restarts
        if trial % 5 == 0:
            init = best_points + np.random.randn(n, 2) * 0.02
        elif trial % 5 == 1:
            init = best_points + np.random.randn(n, 2) * 0.08
        elif trial % 5 == 2:
            init = best_points + np.random.randn(n, 2) * 0.2
        elif trial % 5 == 3:
            # Random restart with some structure
            angle = np.random.uniform(0, 2*np.pi)
            r = np.random.uniform(0.5, 1.5)
            init = np.column_stack([r*np.cos(angle + np.random.randn(n)*0.1), 
                                    r*np.sin(angle + np.random.randn(n)*0.1)])
        else:
            # Perturb only a subset of points (partial restart)
            init = best_points.copy()
            n_perturb = np.random.randint(3, 8)
            idx = np.random.choice(n, n_perturb, replace=False)
            init[idx] += np.random.randn(n_perturb, 2) * 0.3
        
        # Use smooth objective for gradient-based optimization
        result = minimize(objective, init.flatten(), args=(50.0,), method='L-BFGS-B', 
                         options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-9})
        
        if result.fun < 0:
            # Evaluate with exact score
            trial_points = result.x.reshape((n, 2))
            trial_score = score(trial_points)
            if trial_score > best_score:
                best_score = trial_score
                best_points = trial_points.copy()
                trial_count += 1

    # Simulated annealing with adaptive step size for escaping local optima
    if time.time() < deadline - 90:
        sa_points = best_points.copy()
        sa_score = best_score
        current_temp = 0.15
        cooling_rate = 0.97
        n_sa_steps = 3000
        step_scale = 1.0
        
        for step in range(n_sa_steps):
            if time.time() > deadline - 45:
                break
            if current_temp < 1e-5:
                break
            
            # Adaptive step size: scale with temperature
            step_size = current_temp * step_scale
            perturbation = np.random.randn(n, 2) * step_size
            new_points = sa_points + perturbation
            new_score = score(new_points)
            
            # Acceptance criterion
            if new_score > sa_score:
                sa_points = new_points
                sa_score = new_score
                if sa_score > best_score:
                    best_score = sa_score
                    best_points = sa_points.copy()
                # Acceptance: increase step scale slightly
                step_scale = min(step_scale * 1.05, 2.0)
            else:
                # Metropolis criterion with scaled temperature
                delta = new_score - sa_score
                if delta < 0:
                    accept_prob = np.exp(delta / (current_temp * 0.05 + 1e-10))
                else:
                    accept_prob = 1.0
                if np.random.rand() < accept_prob:
                    sa_points = new_points
                    sa_score = new_score
                # Rejection: decrease step scale
                step_scale = max(step_scale * 0.95, 0.3)
            
            current_temp *= cooling_rate
        
        # Local optimization from SA result using smooth objective
        if sa_score > best_score * 0.95:
            result = minimize(objective, sa_points.flatten(), args=(50.0,), method='L-BFGS-B',
                             options={'maxiter': 8000, 'ftol': 1e-12, 'gtol': 1e-9})
            if result.fun < 0:
                sa_trial_points = result.x.reshape((n, 2))
                sa_trial_score = score(sa_trial_points)
                if sa_trial_score > best_score:
                    best_score = sa_trial_score
                    best_points = sa_trial_points.copy()

    # Final refinement with tighter tolerance using smooth objective
    if time.time() < deadline - 20:
        result = minimize(objective, best_points.flatten(), args=(100.0,), method='L-BFGS-B',
                         options={'maxiter': 15000, 'ftol': 1e-14, 'gtol': 1e-11})
        if result.fun < 0:
            final_points = result.x.reshape((n, 2))
            final_score = score(final_points)
            if final_score > best_score:
                best_points = final_points
                best_score = final_score
    
    # Additional Nelder-Mead refinement for fine-tuning
    if time.time() < deadline - 10:
        try:
            result_nm = minimize(lambda p: -score(p.reshape((n,2))), best_points.flatten(), 
                                method='Nelder-Mead',
                                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result_nm.fun < 0:
                nm_points = result_nm.x.reshape((n, 2))
                nm_score = score(nm_points)
                if nm_score > best_score:
                    best_points = nm_points
                    best_score = nm_score
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
