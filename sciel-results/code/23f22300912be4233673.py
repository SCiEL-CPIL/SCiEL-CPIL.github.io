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
    deadline = start_time + 590  # Safety margin before 630s kill

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
        """Generate diverse initial configurations with known good structures"""
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
        center = np.array([[0.0, 0.0]])
        angles2 = np.linspace(0, 2*np.pi, 5, endpoint=False)
        r2 = 0.7
        ring2 = np.column_stack([r2*np.cos(angles2), r2*np.sin(angles2)])
        angles3 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
        r3 = 1.3
        ring3 = np.column_stack([r3*np.cos(angles3), r3*np.sin(angles3)])
        angles4 = np.linspace(0, 2*np.pi, 5, endpoint=False)
        r4 = 1.9
        ring4 = np.column_stack([r4*np.cos(angles4), r4*np.sin(angles4)])
        inits.append(np.vstack([center, ring2, ring3, ring4]))
        
        # Hexagonal-like packing
        center2 = np.array([[0.0, 0.0]])
        hex_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        hex_r = 0.8
        hex_ring = np.column_stack([hex_r*np.cos(hex_angles), hex_r*np.sin(hex_angles)])
        outer_angles = np.linspace(0, 2*np.pi, 9, endpoint=False)
        outer_r2 = 1.6
        outer_ring = np.column_stack([outer_r2*np.cos(outer_angles), outer_r2*np.sin(outer_angles)])
        inits.append(np.vstack([center2, hex_ring, outer_ring]))
        
        # New: 1 + 5 + 10 arrangement (center, pentagon, decagon)
        center3 = np.array([[0.0, 0.0]])
        angles5 = np.linspace(0, 2*np.pi, 5, endpoint=False)
        r5 = 0.6
        ring5 = np.column_stack([r5*np.cos(angles5), r5*np.sin(angles5)])
        angles10 = np.linspace(0, 2*np.pi, 10, endpoint=False)
        r10 = 1.4
        ring10 = np.column_stack([r10*np.cos(angles10), r10*np.sin(angles10)])
        inits.append(np.vstack([center3, ring5, ring10]))
        
        # New: 2 + 6 + 8 arrangement
        angles2a = np.array([0, np.pi])
        r2a = 0.3
        ring2a = np.column_stack([r2a*np.cos(angles2a), r2a*np.sin(angles2a)])
        angles6 = np.linspace(0, 2*np.pi, 6, endpoint=False)
        r6 = 0.9
        ring6 = np.column_stack([r6*np.cos(angles6), r6*np.sin(angles6)])
        angles8 = np.linspace(0, 2*np.pi, 8, endpoint=False)
        r8 = 1.6
        ring8 = np.column_stack([r8*np.cos(angles8), r8*np.sin(angles8)])
        inits.append(np.vstack([ring2a, ring6, ring8]))
        
        # New: 4 + 4 + 8 arrangement
        angles4a = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        r4a = 0.4
        ring4a = np.column_stack([r4a*np.cos(angles4a), r4a*np.sin(angles4a)])
        angles4b = angles4a + np.pi/4
        r4b = 1.0
        ring4b = np.column_stack([r4b*np.cos(angles4b), r4b*np.sin(angles4b)])
        angles8b = np.linspace(0, 2*np.pi, 8, endpoint=False)
        r8b = 1.7
        ring8b = np.column_stack([r8b*np.cos(angles8b), r8b*np.sin(angles8b)])
        inits.append(np.vstack([ring4a, ring4b, ring8b]))
        
        # New: 3 + 3 + 3 + 7 arrangement
        angles3a = np.linspace(0, 2*np.pi, 3, endpoint=False)
        r3a = 0.3
        ring3a = np.column_stack([r3a*np.cos(angles3a), r3a*np.sin(angles3a)])
        angles3b = angles3a + np.pi/3
        r3b = 0.8
        ring3b = np.column_stack([r3b*np.cos(angles3b), r3b*np.sin(angles3b)])
        angles3c = angles3a
        r3c = 1.3
        ring3c = np.column_stack([r3c*np.cos(angles3c), r3c*np.sin(angles3c)])
        angles7 = np.linspace(0, 2*np.pi, 7, endpoint=False)
        r7 = 1.8
        ring7 = np.column_stack([r7*np.cos(angles7), r7*np.sin(angles7)])
        inits.append(np.vstack([ring3a, ring3b, ring3c, ring7]))
        
        # Random with symmetry enforcement
        np.random.seed(42)
        for _ in range(12):
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
    n_trials = 60
    trial_count = 0
    
    for trial in range(n_trials):
        if time.time() > deadline - 120:
            break
        
        # Alternate between perturbations of best and random restarts
        if trial % 5 == 0:
            init = best_points + np.random.randn(n, 2) * 0.02
        elif trial % 5 == 1:
            init = best_points + np.random.randn(n, 2) * 0.05
        elif trial % 5 == 2:
            init = best_points + np.random.randn(n, 2) * 0.15
        elif trial % 5 == 3:
            init = best_points + np.random.randn(n, 2) * 0.4
        else:
            # Random restart with some structure
            angle = np.random.uniform(0, 2*np.pi)
            r = np.random.uniform(0.5, 1.5)
            init = np.column_stack([r*np.cos(angle + np.random.randn(n)*0.1), 
                                    r*np.sin(angle + np.random.randn(n)*0.1)])
        
        result = minimize(objective, init.flatten(), method='L-BFGS-B', 
                         options={'maxiter': 4000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if result.success and result.fun < 0:
            trial_score = -result.fun
            if trial_score > best_score:
                best_score = trial_score
                best_points = result.x.reshape((n, 2))
                trial_count += 1

    # Simulated annealing for escaping local optima
    if time.time() < deadline - 90:
        sa_points = best_points.copy()
        sa_score = best_score
        current_temp = 0.15
        cooling_rate = 0.97
        n_sa_steps = 3000
        
        for step in range(n_sa_steps):
            if time.time() > deadline - 45:
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
                             options={'maxiter': 6000, 'ftol': 1e-12, 'gtol': 1e-8})
            if result.success and -result.fun > best_score:
                best_score = -result.fun
                best_points = result.x.reshape((n, 2))

    # Final refinement with tighter tolerance
    if time.time() < deadline - 20:
        result = minimize(objective, best_points.flatten(), method='L-BFGS-B',
                         options={'maxiter': 12000, 'ftol': 1e-14, 'gtol': 1e-10})
        if result.success and -result.fun > best_score:
            best_points = result.x.reshape((n, 2))
            best_score = -result.fun

    return best_points

# EVOLVE-BLOCK-END
