import numpy as np
import time
from scipy.optimize import minimize

N = 26
TIME_LIMIT = 110

# Pre-define constraint functions
def _make_constraints():
    cons = []
    for i in range(N):
        # x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i] - z[3*i+2]})
        # 1 - x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: 1 - z[3*i] - z[3*i+2]})
        # y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i+1] - z[3*i+2]})
        # 1 - y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: 1 - z[3*i+1] - z[3*i+2]})
        # r >= 1e-6
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i+2] - 1e-6})
    
    for i in range(N):
        for j in range(i+1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda z, i=i, j=j: (z[3*i] - z[3*j])**2 + (z[3*i+1] - z[3*j+1])**2 - (z[3*i+2] + z[3*j+2])**2
            })
    return cons

CONSTRAINTS = _make_constraints()

def _objective(z):
    return -np.sum(z[2::3])

def _make_initial_guesses():
    guesses = []
    
    # Guess 1: Hexagonal
    z = np.zeros(3*N)
    idx = 0
    r_hex = 0.09
    dy = 2 * r_hex
    dx = np.sqrt(3) * r_hex
    for j in range(6):
        for i in range(6):
            if idx >= N: break
            x = dx * (i + (j % 2) * 0.5) + r_hex
            y = dy * j + r_hex
            if x <= 1 - r_hex and y <= 1 - r_hex:
                z[3*idx] = x
                z[3*idx+1] = y
                z[3*idx+2] = r_hex
                idx += 1
    if idx < N:
        for k in range(idx, N):
            z[3*k] = 0.5
            z[3*k+1] = 0.5
            z[3*k+2] = 0.01
    guesses.append(z)
    
    # Guess 2: Corner heavy
    z = np.zeros(3*N)
    r_corner = 0.2
    corners = [(r_corner, r_corner), (1-r_corner, r_corner), (r_corner, 1-r_corner), (1-r_corner, 1-r_corner)]
    for i, (x, y) in enumerate(corners):
        z[3*i] = x
        z[3*i+1] = y
        z[3*i+2] = r_corner
    side_c = 5
    span = 0.6
    step = span / (side_c - 1)
    idx = 4
    for i in range(side_c):
        for j in range(side_c):
            if idx >= N: break
            x = 0.2 + i * step
            y = 0.2 + j * step
            z[3*idx] = x
            z[3*idx+1] = y
            z[3*idx+2] = 0.05
            idx += 1
    guesses.append(z)
    
    # Guess 3: Random
    np.random.seed(42)
    z = np.zeros(3*N)
    for i in range(N):
        z[3*i] = np.random.uniform(0.1, 0.9)
        z[3*i+1] = np.random.uniform(0.1, 0.9)
        z[3*i+2] = 0.05
    guesses.append(z)
    
    return guesses

def _is_valid(z, tol=1e-8):
    for i in range(N):
        x, y, r = z[3*i], z[3*i+1], z[3*i+2]
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dist_sq = (z[3*i] - z[3*j])**2 + (z[3*i+1] - z[3*j+1])**2
            rad_sum_sq = (z[3*i+2] + z[3*j+2])**2
            if dist_sq < rad_sum_sq - tol:
                return False
    return True

def construct():
    start_time = time.time()
    best_z = None
    best_val = -np.inf
    
    guesses = _make_initial_guesses()
    
    for guess in guesses:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
            
        try:
            res = minimize(
                _objective, 
                guess, 
                method='SLSQP', 
                bounds=[(1e-6, 1)] * (3*N),
                constraints=CONSTRAINTS,
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            
            if res.success:
                z = res.x
                if _is_valid(z):
                    val = -res.fun
                    if val > best_val:
                        best_val = val
                        best_z = z.copy()
        except Exception:
            pass

    if best_z is None:
        # Fallback
        z = np.zeros(3*N)
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx < N:
                    z[3*idx] = (i + 0.5) / 5
                    z[3*idx+1] = (j + 0.5) / 5
                    z[3*idx+2] = 0.1
                    idx += 1
        if idx < N:
            z[3*idx] = 0.5
            z[3*idx+1] = 0.5
            z[3*idx+2] = 0.01
        best_z = z
        best_val = np.sum(best_z[2::3])

    # Polish
    if time.time() - start_time < TIME_LIMIT - 5:
        try:
            res = minimize(
                _objective, 
                best_z, 
                method='SLSQP', 
                bounds=[(1e-6, 1)] * (3*N),
                constraints=CONSTRAINTS,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            if res.success:
                z = res.x
                if _is_valid(z):
                    val = -res.fun
                    if val > best_val:
                        best_val = val
                        best_z = z.copy()
        except Exception:
            pass

    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_z[3*i]
        result[i, 1] = best_z[3*i+1]
        result[i, 2] = best_z[3*i+2]
        
    return result
