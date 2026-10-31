import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s for polish and safety
    
    # Initial state: 6x6 grid, take first 26
    side = 6
    positions = []
    radii = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side
                positions.append([x, y])
                radii.append(r)
    
    state = np.array([p + [r] for p, r in zip(positions, radii)])
    
    def energy(s):
        x = s[:, 0]
        y = s[:, 1]
        r = s[:, 2]
        
        v_bound = np.maximum(0, r - x) + np.maximum(0, r - (1 - x)) + \
                  np.maximum(0, r - y) + np.maximum(0, r - (1 - y))
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, None] + r[None, :]
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        viol_overlap = np.maximum(0, r_sum - dist)
        viol_overlap = viol_overlap[mask]
        
        total_viol = np.sum(v_bound) + np.sum(viol_overlap)
        
        E = 10.0 * total_viol**2 - np.sum(r)
        return E

    best_state = state.copy()
    best_energy = energy(best_state)
    
    T = 0.1
    alpha = 0.9999
    
    while time.time() < deadline:
        new_state = state.copy()
        idx = np.random.randint(N)
        
        step_size = 0.05 * T / 0.1
        new_state[idx, 0] += np.random.normal(0, step_size)
        new_state[idx, 1] += np.random.normal(0, step_size)
        new_state[idx, 2] += np.random.normal(0, step_size)
        new_state[idx, 2] = max(0.0, new_state[idx, 2])
        
        E_old = energy(state)
        E_new = energy(new_state)
        
        if E_new < E_old or np.random.rand() < np.exp(-(E_new - E_old) / T):
            state = new_state
            if E_new < best_energy:
                best_energy = E_new
                best_state = new_state.copy()
        
        T *= alpha
        if T < 0.001:
            T = 0.001

    # Polish with scipy
    def objective(s):
        # Reshape to (N, 3)
        s = s.reshape(N, 3)
        return energy(s)
        
    # Use Nelder-Mead
    try:
        res = minimize(objective, best_state.flatten(), method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < best_energy:
            best_state = res.x.reshape(N, 3)
            best_energy = res.fun
    except Exception:
        pass

    # Ensure validity
    def is_valid(s):
        x = s[:, 0]
        y = s[:, 1]
        r = s[:, 2]
        if np.any(r < 0): return False
        if np.any(x < r) or np.any(x > 1-r) or np.any(y < r) or np.any(y > 1-r): return False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, None] + r[None, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        if np.any((r_sum - dist)[mask] > 1e-9): return False
        return True

    if not is_valid(best_state):
        scale = 1.0
        for _ in range(100):
            new_r = best_state[:, 2] * scale
            temp_state = best_state.copy()
            temp_state[:, 2] = new_r
            if is_valid(temp_state):
                best_state = temp_state
                break
            scale *= 0.95
            if scale < 0.1:
                break
                
    return best_state
