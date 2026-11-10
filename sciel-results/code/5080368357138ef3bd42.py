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
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        ratio = d_min / d_max
        return ratio ** 2

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

    # Simulated annealing for escaping local optima
    if time.time() < deadline - 60:
        sa_points = best_points.copy()
        sa_score = best_score
        current_temp = 0.1
        cooling_rate = 0.95
        n_sa_steps = 2000
        
        for step in range(n_sa_steps):
            if time.time() > deadline - 30:
                break
            if current_temp < 1e-6:
                break
            
            # Generate perturbation
            perturbation = np.random.randn(n, 2) * current_temp
            new_points = sa_points + perturbation
            new_score = score(new_points)
            
            # Acceptance criterion
            if new_score > sa_score:
                sa_points = new_points
                sa_score = new_score
                if sa_score > best_score:
                    best_score = sa_score
                    best_points = sa_points.copy()
            else:
                # Metropolis criterion
                delta = new_score - sa_score
                if np.random.rand() < np.exp(delta / (current_temp * 0.01 + 1e-10)):
                    sa_points = new_points
                    sa_score = new_score
            
            current_temp *= cooling_rate
        
        # Local optimization from SA result
        if sa_score > best_score * 0.99:
            result = minimize(objective, sa_points.flatten(), method='L-BFGS-B',
                             options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
            if result.success and -result.fun > best_score:
                best_score = -result.fun
                best_points = result.x.reshape((n, 2))

    # Final refinement with tighter tolerance
    if time.time() < deadline - 15:
        result = minimize(objective, best_points.flatten(), method='L-BFGS-B',
                         options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-10})
        if result.success and -result.fun > best_score:
            best_points = result.x.reshape((n, 2))
            best_score = -result.fun

    return best_points

# EVOLVE-BLOCK-END
