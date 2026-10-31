"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Check overlap: dist_sq >= (r_i + r_j)^2
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
    return True

def _post_process(pos, radii):
    """
    Shrink radii and adjust positions to ensure strict validity.
    This is a safety net.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    
    for _ in range(100):
        valid = True
        
        # 1. Ensure circles are inside the square
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-6)
                valid = False
        
        # 2. Ensure no overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Scale down both radii to just fit
                    # We want new_r_i + new_r_j = dist * 0.9999
                    # Keep ratio of radii
                    total_r = radii[i] + radii[j]
                    if total_r > 0:
                        scale = (dist * 0.9999) / total_r
                        if scale < 1.0:
                            radii[i] *= scale
                            radii[j] *= scale
                            valid = False
                    else:
                        radii[i] = 1e-6
                        radii[j] = 1e-6
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _energy(pos, radii):
    """
    Calculate the energy (penalty) for a configuration.
    Lower is better. 0 means valid.
    """
    n = len(radii)
    penalty = 0.0
    
    # Boundary violations
    for i in range(n):
        x, y, r = pos[i, 0], pos[i, 1], radii[i]
        if x - r < 0: penalty += (x - r) ** 2
        if x + r > 1: penalty += (x + r - 1) ** 2
        if y - r < 0: penalty += (y - r) ** 2
        if y + r > 1: penalty += (y + r - 1) ** 2
        
    # Overlap violations
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
                
    return penalty

def _simulated_annealing(pos, radii, deadline, seed):
    """
    Simulated Annealing to find a good configuration.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    current_energy = _energy(pos, radii)
    
    t0 = 0.05
    t_end = 0.0005
    # Estimate steps based on time
    # We will run in a loop until deadline
    step = 0
    while time.time() < deadline:
        # Cool down
        frac = min(step / 5000.0, 1.0)
        temp = t0 * (t_end / t0) ** frac
        
        # Perturb a random circle
        i = rng.randint(N)
        dx = rng.normal(0, temp * 0.05)
        dy = rng.normal(0, temp * 0.05)
        dr = rng.normal(0, temp * 0.02)
        
        new_x = pos[i, 0] + dx
        new_y = pos[i, 1] + dy
        new_r = radii[i] + dr
        
        # Clamp to reasonable bounds
        new_x = np.clip(new_x, 0.0, 1.0)
        new_y = np.clip(new_y, 0.0, 1.0)
        new_r = np.clip(new_r, 1e-4, 0.5)
        
        # Save old state
        old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
        
        # Apply new state
        pos[i, 0] = new_x
        pos[i, 1] = new_y
        radii[i] = new_r
        
        new_energy = _energy(pos, radii)
        delta_e = new_energy - current_energy
        
        # Acceptance probability
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-8)):
            current_energy = new_energy
        else:
            # Reject
            pos[i, 0] = old_x
            pos[i, 1] = old_y
            radii[i] = old_r
            
        # Check for improvement
        if step % 100 == 0:
            # Ensure validity for score calculation
            pp_pos, pp_radii = _post_process(pos, radii)
            if _is_valid(pp_pos, pp_radii):
                score = np.sum(pp_radii)
                if score > best_score:
                    best_score = score
                    best_pos = pp_pos.copy()
                    best_radii = pp_radii.copy()
                    
        step += 1
        
    return best_pos, best_radii

def _slsqp_polish(pos, radii, deadline):
    """
    Polish using SLSQP with hard constraints.
    """
    x0 = np.concatenate([pos.flatten(), radii])
    
    def objective(x):
        # Maximize sum of radii => Minimize negative sum
        return -np.sum(x[2*N:])
        
    def obj_grad(x):
        grad = np.zeros_like(x)
        # Gradient of -sum(radii)
        grad[2*N:] = -1.0
        return grad
        
    constraints = []
    
    # Boundary constraints:
    # r_i <= x_i
    # r_i <= 1 - x_i
    # r_i <= y_i
    # r_i <= 1 - y_i
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i], 
                            'jac': lambda x, i=i: np.concatenate([np.eye(2*N)[2*i], -np.eye(N)[i]])})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*i]) - x[2*N+i], 
                            'jac': lambda x, i=i: np.concatenate([-np.eye(2*N)[2*i], -np.eye(N)[i]])})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i], 
                            'jac': lambda x, i=i: np.concatenate([np.eye(2*N)[2*i+1], -np.eye(N)[i]])})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*i+1]) - x[2*N+i], 
                            'jac': lambda x, i=i: np.concatenate([-np.eye(2*N)[2*i+1], -np.eye(N)[i]])})
                            
    # Non-overlap constraints:
    # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[2*i]-x[2*j])**2 + (x[2*i+1]-x[2*j+1])**2 - (x[2*N+i]+x[2*N+j])**2,
                'jac': lambda x, i=i, j=j: np.concatenate([
                    np.concatenate([np.eye(2*N)[2*i]*2*(x[2*i]-x[2*j]), np.zeros(2*N-2), np.eye(2*N)[2*j]*(-2)*(x[2*i]-x[2*j])]), # dx part
                    np.concatenate([np.eye(2*N)[2*i+1]*2*(x[2*i+1]-x[2*j+1]), np.zeros(2*N-2), np.eye(2*N)[2*j+1]*(-2)*(x[2*i+1]-x[2*j+1])]), # dy part
                    np.concatenate([np.zeros(2*N), np.eye(N)[i]*(-2)*(x[2*N+i]+x[2*N+j]), np.zeros(N-1), np.eye(N)[j]*(-2)*(x[2*N+i]+x[2*N+j])]) # dr part
                ])
            })
            
    bounds = []
    for i in range(N):
        bounds.append((1e-4, 0.9999))
        bounds.append((1e-4, 0.9999))
    for i in range(N):
        bounds.append((1e-4, 0.5))
        
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                       constraints=constraints, jac=obj_grad,
                       options={'maxiter': 200, 'ftol': 1e-10})
        if res.success or res.fun < -np.sum(radii):
            x_opt = res.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = x_opt[2*N:]
            return pos_opt, radii_opt
    except Exception:
        pass
        
    return pos, radii

def _initial_random(seed):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.02, 0.1, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_grid():
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    # 5x5 grid
    side = 5
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            if idx < N:
                pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                radii[idx] = 0.19
    # Fill remaining if any
    for i in range(N):
        if radii[i] == 0:
            pos[i] = [0.5, 0.5]
            radii[i] = 0.01
    pos, radii = _post_process(pos, radii)
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pos = None
    best_radii = None
    best_score = 0.0
    
    # List of initializations
    inits = []
    inits.append(_initial_grid())
    for i in range(5):
        inits.append(_initial_random(i * 100))
        
    # Strategy:
    # 1. Run SA on each init for a short time
    # 2. Keep the best result
    # 3. Run SLSQP polish on the best result
    # 4. If time remains, run SLSQP on other good results
    
    candidates = []
    
    for idx, (pos, radii) in enumerate(inits):
        if time.time() > deadline - 30:
            break
            
        # SA phase
        sa_deadline = min(deadline - 30, time.time() + 20) # 20s per init max
        pos_sa, radii_sa = _simulated_annealing(pos, radii, sa_deadline, seed=idx)
        
        # Post-process SA result
        pos_pp, radii_pp = _post_process(pos_sa, radii_sa)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            candidates.append((score, pos_pp, radii_pp))
            
            if score > best_score:
                best_score = score
                best_pos = pos_pp
                best_radii = radii_pp
                
    # Sort candidates by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Polish top candidates with SLSQP
    for score, pos_c, radii_c in candidates[:3]: # Top 3
        if time.time() > deadline - 10:
            break
            
        pos_pol, radii_pol = _slsqp_polish(pos_c, radii_c, deadline)
        
        # Post-process SLSQP result
        pos_pp, radii_pp = _post_process(pos_pol, radii_pol)
        if _is_valid(pos_pp, radii_pp):
            final_score = np.sum(radii_pp)
            if final_score > best_score:
                best_score = final_score
                best_pos = pos_pp
                best_radii = radii_pp
                
    # Final safety net
    if best_pos is None:
        best_pos, best_radii = _initial_grid()
        
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback to simple grid
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.19
                    k += 1
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
        
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
