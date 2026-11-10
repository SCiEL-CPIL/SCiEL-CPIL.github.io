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
        """Calculate R = (d_min/d_max)^2 - vectorized for speed"""
        dists = pdist(points, 'euclidean')
        d_min = dists.min()
        d_max = dists.max()
        if d_max < 1e-10:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def objective(params):
        """Minimize -R"""
        points = params.reshape((n, 2))
        return -score(points)

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
        
        # Structured init: two interlocking squares (rotated 45 degrees)
        # Square 1: 8 points
        angles_s1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
        r_s1 = 0.9
        sq1_pts = np.column_stack([r_s1*np.cos(angles_s1), r_s1*np.sin(angles_s1)])
        
        # Square 2: 8 points, rotated
        angles_s2 = angles_s1 + np.pi/8
        r_s2 = 1.1
        sq2_pts = np.column_stack([r_s2*np.cos(angles_s2), r_s2*np.sin(angles_s2)])
        
        inits.append(np.vstack([sq1_pts, sq2_pts]))
        
        # Structured init: hexagonal lattice fragment
        # Row 0: 5 points
        row0_x = np.linspace(-1.5, 1.5, 5)
        row0_y = np.zeros(5)
        row0 = np.column_stack([row0_x, row0_y])
        
        # Row 1: 6 points, offset
        row1_x = np.linspace(-1.2, 1.2, 6)
        row1_y = np.full(6, 0.866)
        row1 = np.column_stack([row1_x, row1_y])
        
        # Row 2: 5 points
        row2_x = np.linspace(-1.5, 1.5, 5)
        row2_y = np.full(5, 1.732)
        row2 = np.column_stack([row2_x, row2_y])
        
        inits.append(np.vstack([row0, row1, row2]))
        
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
    n_trials = 50
    trial_count = 0
    
    for trial in range(n_trials):
        if time.time() > deadline - 90:
            break
        
        # Alternate between perturbations of best and random restarts
        if trial % 4 == 0:
            init = best_points + np.random.randn(n, 2) * 0.03
        elif trial % 4 == 1:
            init = best_points + np.random.randn(n, 2) * 0.1
        elif trial % 4 == 2:
            init = best_points + np.random.randn(n, 2) * 0.3
        else:
            # Random restart with some structure
            angle = np.random.uniform(0, 2*np.pi)
            r = np.random.uniform(0.5, 1.5)
            init = np.column_stack([r*np.cos(angle + np.random.randn(n)*0.1), 
                                    r*np.sin(angle + np.random.randn(n)*0.1)])
        
        result = minimize(objective, init.flatten(), method='L-BFGS-B', 
                         options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if result.success and result.fun < 0:
            trial_score = -result.fun
            if trial_score > best_score:
                best_score = trial_score
                best_points = result.x.reshape((n, 2))
                trial_count += 1

    # Improved simulated annealing with adaptive step size
    if time.time() < deadline - 60:
        sa_points = best_points.copy()
        sa_score = best_score
        current_temp = 0.5  # Start with larger steps
        cooling_rate = 0.995  # Slower cooling for more effective search
        n_sa_steps = 5000
        
        accept_count = 0
        total_count = 0
        
        for step in range(n_sa_steps):
            if time.time() > deadline - 30:
                break
            if current_temp < 1e-8:
                break
            
            total_count += 1
            
            # Adaptive perturbation: scale with current temperature and point spread
            point_spread = np.std(sa_points, axis=0)
            base_scale = np.max(point_spread) if np.max(point_spread) > 1e-6 else 1.0
            perturbation = np.random.randn(n, 2) * current_temp * base_scale
            new_points = sa_points + perturbation
            new_score = score(new_points)
            
            # Acceptance criterion with proper Metropolis
            if new_score >= sa_score:
                sa_points = new_points
                sa_score = new_score
                accept_count += 1
                if sa_score > best_score:
                    best_score = sa_score
                    best_points = sa_points.copy()
            else:
                # Metropolis criterion: accept with probability exp(-delta/T)
                delta = sa_score - new_score
                if delta < 1e-15:
                    sa_points = new_points
                    sa_score = new_score
                    accept_count += 1
                elif np.random.rand() < np.exp(-delta / (current_temp * best_score * 0.1 + 1e-12)):
                    sa_points = new_points
                    sa_score = new_score
                    accept_count += 1
            
            # Adaptive temperature adjustment
            if total_count % 500 == 0 and total_count > 0:
                accept_rate = accept_count / total_count
                if accept_rate > 0.5:
                    current_temp *= 0.9  # Too many accepts, cool faster
                elif accept_rate < 0.1:
                    current_temp *= 1.1  # Too few accepts, warm up
                accept_count = 0
                total_count = 0
            
            current_temp *= cooling_rate
        
        # Local optimization from SA result
        if time.time() < deadline - 20:
            result = minimize(objective, sa_points.flatten(), method='L-BFGS-B',
                             options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
            if result.success and -result.fun > best_score:
                best_score = -result.fun
                best_points = result.x.reshape((n, 2))

    # Multiple refinement passes with different methods
    if time.time() < deadline - 15:
        # Pass 1: L-BFGS-B with tight tolerance
        result = minimize(objective, best_points.flatten(), method='L-BFGS-B',
                         options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-10})
        if result.success and -result.fun > best_score:
            best_points = result.x.reshape((n, 2))
            best_score = -result.fun
        
        # Pass 2: Try Nelder-Mead for local refinement (different optimizer)
        if time.time() < deadline - 10:
            result2 = minimize(objective, best_points.flatten(), method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            if result2.success and -result2.fun > best_score:
                best_points = result2.x.reshape((n, 2))
                best_score = -result2.fun
        
        # Pass 3: Another L-BFGS-B from slightly perturbed best
        if time.time() < deadline - 5:
            perturbed = best_points + np.random.randn(n, 2) * 0.01
            result3 = minimize(objective, perturbed.flatten(), method='L-BFGS-B',
                              options={'maxiter': 5000, 'ftol': 1e-13, 'gtol': 1e-9})
            if result3.success and -result3.fun > best_score:
                best_points = result3.x.reshape((n, 2))
                best_score = -result3.fun

    return best_points

# EVOLVE-BLOCK-END
