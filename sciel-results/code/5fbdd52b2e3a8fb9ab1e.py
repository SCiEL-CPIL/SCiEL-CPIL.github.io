"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # Initial guess: a reasonable grid packing
    side = 6  # 6x6 grid = 36 cells, use first 26
    positions = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append((x, y))
    positions = positions[:N]
    
    # Initial radii: small equal radii
    r_init = np.full(N, 0.08)
    
    # Build initial state vector: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.array([p[0] for p in positions])
    y0 = np.array([p[1] for p in positions])
    r0 = r_init.copy()
    
    def state_to_circles(state):
        xs = state[0:N]
        ys = state[N:2*N]
        rs = state[2*N:3*N]
        return xs, ys, rs
    
    def objective(state):
        """Minimize negative sum of radii (to maximize sum)."""
        xs, ys, rs = state_to_circles(state)
        return -np.sum(rs)
    
    def constraints(state):
        """Return list of constraint dicts for scipy.optimize.minimize."""
        xs, ys, rs = state_to_circles(state)
        cons = []
        
        # Boundary constraints: circles must be inside [0,1]^2
        # x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda s, i=i: s[i] - s[2*N+i]})
        # x_i + r_i <= 1
        cons.append({'type': 'ineq', 'fun': lambda s, i=i: 1 - s[i] - s[2*N+i]})
        # y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda s, i=i: s[N+i] - s[2*N+i]})
        # y_i + r_i <= 1
        cons.append({'type': 'ineq', 'fun': lambda s, i=i: 1 - s[N+i] - s[2*N+i]})
        
        # Pairwise non-overlap: dist(i,j) >= r_i + r_j
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda s, i=i, j=j: np.sqrt((s[i]-s[j])**2 + (s[N+i]-s[N+j])**2) - s[2*N+i] - s[2*N+j]
                })
        
        return cons
    
    # Build initial state
    state0 = np.concatenate([x0, y0, r0])
    
    # Run optimization with multiple restarts
    best_state = state0.copy()
    best_obj = objective(state0)
    
    # Try several random restarts
    np.random.seed(42)
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            state = state0.copy()
        else:
            # Random perturbation around a reasonable packing
            xs_rand = np.random.uniform(0.1, 0.9, N)
            ys_rand = np.random.uniform(0.1, 0.9, N)
            rs_rand = np.full(N, 0.07)
            state = np.concatenate([xs_rand, ys_rand, rs_rand])
        
        # Use SLSQP which handles constraints well
        try:
            res = minimize(
                objective,
                state,
                method='SLSQP',
                bounds=[(0, 1)] * (2*N) + [(0.01, 0.5)] * N,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res.success or res.fun < best_obj:
                # Verify constraints are satisfied
                xs, ys, rs = state_to_circles(res.x)
                valid = True
                # Check boundaries
                if np.any(xs - rs < -1e-9) or np.any(xs + rs > 1 + 1e-9) or \
                   np.any(ys - rs < -1e-9) or np.any(ys + rs > 1 + 1e-9):
                    valid = False
                # Check pairwise
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                            if dist < rs[i] + rs[j] - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    best_state = res.x.copy()
                    best_obj = res.fun
        except Exception as e:
            continue
    
    # Final polish: local refinement around best solution
    if time.time() < deadline - 5:
        try:
            res_polish = minimize(
                objective,
                best_state,
                method='SLSQP',
                bounds=[(0, 1)] * (2*N) + [(0.01, 0.5)] * N,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            if res_polish.success:
                xs, ys, rs = state_to_circles(res_polish.x)
                valid = True
                if np.any(xs - rs < -1e-9) or np.any(xs + rs > 1 + 1e-9) or \
                   np.any(ys - rs < -1e-9) or np.any(ys + rs > 1 + 1e-9):
                    valid = False
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                            if dist < rs[i] + rs[j] - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                if valid:
                    best_state = res_polish.x.copy()
        except Exception:
            pass
    
    # Extract final result
    xs, ys, rs = state_to_circles(best_state)
    
    # Clamp to valid range to be safe
    rs = np.clip(rs, 0.001, 0.5)
    xs = np.clip(xs, rs, 1 - rs)
    ys = np.clip(ys, rs, 1 - rs)
    
    # Final validation and adjustment
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                # Shrink radii slightly to fix
                shrink = (min_dist - dist) / 2 + 1e-6
                rs[i] = max(0.001, rs[i] - shrink)
                rs[j] = max(0.001, rs[j] - shrink)
                # Re-clamp positions
                xs[i] = np.clip(xs[i], rs[i], 1 - rs[i])
                ys[i] = np.clip(ys[i], rs[i], 1 - rs[i])
                xs[j] = np.clip(xs[j], rs[j], 1 - rs[j])
                ys[j] = np.clip(ys[j], rs[j], 1 - rs[j])
    
    result = np.column_stack([xs, ys, rs])
    return result
# EVOLVE-BLOCK-END
