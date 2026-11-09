<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
=======
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Compute minimum side length of enclosing hexagon
    def enclosing_hex_side(inner_positions, inner_angles, outer_center, outer_angle_deg):
        ocx, ocy = outer_center
        oa = np.radians(outer_angle_deg)
        
        all_verts = []
        for i in range(len(inner_positions)):
            cx, cy = inner_positions[i]
            a_rad = np.radians(inner_angles[i])
            v = hex_vertices(cx, cy, 1.0, a_rad)
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        all_verts_shifted = all_verts - np.array([ocx, ocy])
        
        min_s = 0.0
        for i in range(6):
            a_i = oa + np.radians(60 * i)
            a_ip1 = oa + np.radians(60 * (i + 1))
            e_unit = np.array([np.cos(a_ip1) - np.cos(a_i), np.sin(a_ip1) - np.sin(a_i)])
            crosses = all_verts_shifted[:, 0] * e_unit[1] - all_verts_shifted[:, 1] * e_unit[0]
            min_cross = np.min(crosses)
            required_s = -2 * min_cross / SQRT3
            if required_s > min_s:
                min_s = required_s
        
        return max(min_s, 0.0)
=======
    # Compute minimum side length of enclosing hexagon
    def enclosing_hex_side(inner_positions, inner_angles, outer_center, outer_angle_deg):
        ocx, ocy = outer_center
        oa = np.radians(outer_angle_deg)
        
        all_verts = []
        for i in range(len(inner_positions)):
            cx, cy = inner_positions[i]
            a_rad = np.radians(inner_angles[i])
            v = hex_vertices(cx, cy, 1.0, a_rad)
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        all_verts_shifted = all_verts - np.array([ocx, ocy])
        
        min_s = 0.0
        for i in range(6):
            a_i = oa + np.radians(60 * i)
            a_ip1 = oa + np.radians(60 * (i + 1))
            e_unit = np.array([np.cos(a_ip1) - np.cos(a_i), np.sin(a_ip1) - np.sin(a_i)])
            crosses = all_verts_shifted[:, 0] * e_unit[1] - all_verts_shifted[:, 1] * e_unit[0]
            min_cross = np.min(crosses)
            required_s = -2 * min_cross / SQRT3
            if required_s > min_s:
                min_s = required_s
        
        return max(min_s, 0.0)
    
    # Smooth surrogate for separation check using soft-min
    def soft_min_gap(v1, v2, beta=500):
        """Compute a smooth approximation of the minimum gap between two hexagons."""
        min_gap = np.inf
        for vi in [v1, v2]:
            for k in range(6):
                e = vi[(k+1)%6] - vi[k]
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-15:
                    n = n / norm
                proj_i = v1 @ n
                proj_j = v2 @ n
                gap = max(proj_i.min() - proj_j.max(), proj_j.min() - proj_i.max())
                if gap < min_gap:
                    min_gap = gap
        # Return soft penalty: for gap < 0, penalty is large; smooth transition
        # Use log-sum-exp style soft-min for the gap
        if min_gap < 0:
            return -min_gap  # overlap amount
        else:
            return 0.0
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Objective function: minimize outer hexagon side length with penalty for overlaps
    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        if not np.all(np.isfinite(params)):
            return 1e10
        
        # Parse parameters: 11*(x,y,angle) + outer(x,y,angle) = 36 params
        inner_params = params[:33].reshape(11, 3)
        outer_params = params[33:36]
        
        inner_positions = inner_params[:, :2]
        inner_angles = inner_params[:, 2]
        outer_center = outer_params[:2]
        outer_angle = outer_params[2]
        
        # Compute vertices of all inner hexagons
        inner_verts = all_hex_vertices(inner_positions, inner_angles)
        
        # Check separation between all pairs
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                if not hexagons_separated(inner_verts[i], inner_verts[j]):
                    # Compute minimum overlap
                    min_gap = np.inf
                    for vi in [inner_verts[i], inner_verts[j]]:
                        for k in range(6):
                            e = vi[(k+1)%6] - vi[k]
                            n = np.array([-e[1], e[0]])
                            norm = np.linalg.norm(n)
                            if norm > 1e-15:
                                n = n / norm
                            proj_i = inner_verts[i] @ n
                            proj_j = inner_verts[j] @ n
                            gap = max(proj_i.min() - proj_j.max(), proj_j.min() - proj_i.max())
                            if gap < min_gap:
                                min_gap = gap
                    # Penalty for overlap
                    if min_gap < 1e-6:
                        penalty += (1e-6 - min_gap) ** 2
        
        # Compute enclosing hexagon side
        s = enclosing_hex_side(inner_positions, inner_angles, outer_center, outer_angle)
        
        # Also ensure side >= sqrt(11)
        if s < np.sqrt(11):
            s = np.sqrt(11)
        
        return s + 1000 * penalty
=======
    # Objective function: minimize outer hexagon side length with penalty for overlaps
    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        if not np.all(np.isfinite(params)):
            return 1e10
        
        # Parse parameters: 11*(x,y,angle) + outer(x,y,angle) = 36 params
        inner_params = params[:33].reshape(11, 3)
        outer_params = params[33:36]
        
        inner_positions = inner_params[:, :2]
        inner_angles = inner_params[:, 2]
        outer_center = outer_params[:2]
        outer_angle = outer_params[2]
        
        # Compute vertices of all inner hexagons
        inner_verts = all_hex_vertices(inner_positions, inner_angles)
        
        # Check separation between all pairs
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                if not hexagons_separated(inner_verts[i], inner_verts[j]):
                    # Compute minimum overlap
                    min_gap = np.inf
                    for vi in [inner_verts[i], inner_verts[j]]:
                        for k in range(6):
                            e = vi[(k+1)%6] - vi[k]
                            n = np.array([-e[1], e[0]])
                            norm = np.linalg.norm(n)
                            if norm > 1e-15:
                                n = n / norm
                            proj_i = inner_verts[i] @ n
                            proj_j = inner_verts[j] @ n
                            gap = max(proj_i.min() - proj_j.max(), proj_j.min() - proj_i.max())
                            if gap < min_gap:
                                min_gap = gap
                    # Penalty for overlap
                    if min_gap < 1e-6:
                        penalty += (1e-6 - min_gap) ** 2
        
        # Compute enclosing hexagon side
        s = enclosing_hex_side(inner_positions, inner_angles, outer_center, outer_angle)
        
        # Also ensure side >= sqrt(11)
        if s < np.sqrt(11):
            s = np.sqrt(11)
        
        return s + 1000 * penalty
    
    # Alternative smooth objective for differential evolution
    def smooth_objective(params):
        if time.time() > deadline:
            return 1e10
        
        if not np.all(np.isfinite(params)):
            return 1e10
        
        inner_params = params[:33].reshape(11, 3)
        outer_params = params[33:36]
        
        inner_positions = inner_params[:, :2]
        inner_angles = inner_params[:, 2]
        outer_center = outer_params[:2]
        outer_angle = outer_params[2]
        
        inner_verts = all_hex_vertices(inner_positions, inner_angles)
        
        # Smooth penalty: sum of soft overlaps
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                overlap = soft_min_gap(inner_verts[i], inner_verts[j])
                if overlap > 0:
                    penalty += overlap ** 2
        
        s = enclosing_hex_side(inner_positions, inner_angles, outer_center, outer_angle)
        if s < np.sqrt(11):
            s = np.sqrt(11)
        
        return s + 500 * penalty
>>>>>>> REPLACE
<<<<<<< SEARCH
    best_params = x0.copy()
    best_val = objective(x0)
    
    # Strategy 1: Multi-start Powell optimization with different initial conditions
    np.random.seed(42)
    num_starts = 12
    for start_idx in range(num_starts):
        if time.time() > deadline - 80:
            break
        
        if start_idx == 0:
            x_start = x0.copy()
        else:
            # Perturb the initial guess
            x_start = x0.copy()
            # Add random noise to positions
            noise_scale = 0.3 * (start_idx + 1)
            for i in range(11):
                x_start[3*i] += np.random.randn() * noise_scale
                x_start[3*i+1] += np.random.randn() * noise_scale
                x_start[3*i+2] += np.random.randn() * 20  # angle in degrees
            # Keep outer center near origin
            x_start[33] = np.random.randn() * 0.3
            x_start[34] = np.random.randn() * 0.3
            x_start[35] = np.random.randn() * 20
        
        try:
            result = minimize(objective, x_start, method='Powell', 
                             options={'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-12})
            
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Strategy 2: Use Nelder-Mead for local refinement from best found
    if time.time() < deadline - 40:
        try:
            result2 = minimize(objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
            
            if result2.fun < best_val:
                best_val = result2.fun
                best_params = result2.x.copy()
        except Exception:
            pass
    
    # Strategy 3: Final high-precision polish with COBYLA
    if time.time() < deadline - 15:
        try:
            result3 = minimize(objective, best_params, method='COBYLA',
                              options={'maxiter': 2000, 'rhobeg': 1e-8, 'tol': 1e-14})
            
            if result3.fun < best_val:
                best_val = result3.fun
                best_params = result3.x.copy()
        except Exception:
            pass
=======
    best_params = x0.copy()
    best_val = objective(x0)
    
    # Strategy 1: Multi-start Powell optimization with different initial conditions
    np.random.seed(42)
    num_starts = 12
    for start_idx in range(num_starts):
        if time.time() > deadline - 80:
            break
        
        if start_idx == 0:
            x_start = x0.copy()
        else:
            # Perturb the initial guess
            x_start = x0.copy()
            # Add random noise to positions
            noise_scale = 0.3 * (start_idx + 1)
            for i in range(11):
                x_start[3*i] += np.random.randn() * noise_scale
                x_start[3*i+1] += np.random.randn() * noise_scale
                x_start[3*i+2] += np.random.randn() * 20  # angle in degrees
            # Keep outer center near origin
            x_start[33] = np.random.randn() * 0.3
            x_start[34] = np.random.randn() * 0.3
            x_start[35] = np.random.randn() * 20
        
        try:
            result = minimize(objective, x_start, method='Powell', 
                             options={'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-12})
            
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Strategy 2: Use Nelder-Mead for local refinement from best found
    if time.time() < deadline - 40:
        try:
            result2 = minimize(objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
            
            if result2.fun < best_val:
                best_val = result2.fun
                best_params = result2.x.copy()
        except Exception:
            pass
    
    # Strategy 3: Final high-precision polish with COBYLA
    if time.time() < deadline - 15:
        try:
            result3 = minimize(objective, best_params, method='COBYLA',
                              options={'maxiter': 2000, 'rhobeg': 1e-8, 'tol': 1e-14})
            
            if result3.fun < best_val:
                best_val = result3.fun
                best_params = result3.x.copy()
        except Exception:
            pass
    
    # Strategy 4: Differential evolution with smooth objective for global exploration
    if time.time() < deadline - 100:
        try:
            # Define bounds for differential evolution
            bounds = [(-5, 5), (-5, 5), (-180, 180)] * 11 + [(-1, 1), (-1, 1), (-180, 180)]
            
            de_result = differential_evolution(
                smooth_objective, 
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-8,
                polish=True,
                init='sobol'
            )
            
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Strategy 5: Final ultra-precision polish with COBYLA from best
    if time.time() < deadline - 10:
        try:
            result4 = minimize(objective, best_params, method='COBYLA',
                              options={'maxiter': 5000, 'rhobeg': 1e-10, 'tol': 1e-16})
            
            if result4.fun < best_val:
                best_val = result4.fun
                best_params = result4.x.copy()
        except Exception:
            pass
>>>>>>> REPLACE
