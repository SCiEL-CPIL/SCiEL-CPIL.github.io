import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_score = -np.inf
    best_solution = None
    
    def penalty_objective_and_grad(v, mu):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        obj = -np.sum(r)
        grad = np.zeros(3*N)
        grad[2::3] = -1.0
        
        # Boundary penalties
        for i in range(N):
            # x < r
            diff = r[i] - x[i]
            if diff > 0:
                pen = diff**2
                obj += mu * pen
                # d(pen)/d(r_i) = 2*diff
                # d(pen)/d(x_i) = -2*diff
                grad[3*i+2] += mu * 2 * diff
                grad[3*i] -= mu * 2 * diff
                
            # x > 1-r
            diff = x[i] - (1 - r[i])
            if diff > 0:
                pen = diff**2
                obj += mu * pen
                # d(pen)/d(r_i) = -2*diff (since r is in 1-r)
                # d(pen)/d(x_i) = 2*diff
                grad[3*i+2] -= mu * 2 * diff
                grad[3*i] += mu * 2 * diff
                
            # y < r
            diff = r[i] - y[i]
            if diff > 0:
                pen = diff**2
                obj += mu * pen
                grad[3*i+2] += mu * 2 * diff
                grad[3*i+1] -= mu * 2 * diff
                
            # y > 1-r
            diff = y[i] - (1 - r[i])
            if diff > 0:
                pen = diff**2
                obj += mu * pen
                grad[3*i+2] -= mu * 2 * diff
                grad[3*i+1] += mu * 2 * diff
        
        # Pairwise penalties
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                diff = r[i] + r[j] - dist
                if diff > 0:
                    pen = diff**2
                    obj += mu * pen
                    # d(pen)/d(r_i) = 2*diff
                    # d(pen)/d(r_j) = 2*diff
                    grad[3*i+2] += mu * 2 * diff
                    grad[3*j+2] += mu * 2 * diff
                    
                    # d(pen)/d(x_i) = 2*diff * dx/dist
                    # d(pen)/d(x_j) = 2*diff * (-dx)/dist
                    # d(pen)/d(y_i) = 2*diff * dy/dist
                    # d(pen)/d(y_j) = 2*diff * (-dy)/dist
                    factor = mu * 2 * diff / dist
                    grad[3*i] += factor * dx
                    grad[3*j] -= factor * dx
                    grad[3*i+1] += factor * dy
                    grad[3*j+1] -= factor * dy
                    
        return obj, grad

    def is_valid(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        if np.any(r > x) or np.any(r > 1-x) or np.any(r > y) or np.any(r > 1-y):
            return False
            
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j] - 1e-6:
                    return False
        return True

    def get_score(v):
        r = v[2::3]
        return np.sum(r)

    def generate_initial_guesses():
        guesses = []
        np.random.seed(42)
        
        # Random
        for _ in range(15):
            x = np.random.rand(N)
            y = np.random.rand(N)
            r = np.full(N, 0.1)
            v = np.zeros(3*N)
            v[0::3] = x
            v[1::3] = y
            v[2::3] = r
            guesses.append(v)
            
        # Hexagonal
        for _ in range(5):
            v = np.zeros(3*N)
            idx = 0
            spacing = 0.18
            for i in range(5):
                for j in range(5):
                    if idx >= N: break
                    x = 0.1 + j * spacing + (i % 2) * spacing * 0.5
                    y = 0.1 + i * spacing * 0.866
                    x = np.clip(x, 0.05, 0.95)
                    y = np.clip(y, 0.05, 0.95)
                    v[3*idx] = x
                    v[3*idx+1] = y
                    v[3*idx+2] = 0.08
                    idx += 1
            guesses.append(v)
            
        return guesses

    initial_guesses = generate_initial_guesses()
    
    for v0 in initial_guesses:
        if time.time() > deadline:
            break
            
        mu = 10.0
        best_v = v0
        best_local_score = -np.inf
        
        for step in range(15):
            if time.time() > deadline:
                break
                
            def fun(v):
                return penalty_objective_and_grad(v, mu)[0]
            
            def jac(v):
                return penalty_objective_and_grad(v, mu)[1]
                
            res = minimize(fun, best_v, jac=jac, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12})
            
            if res.success or res.status == 0:
                if is_valid(res.x):
                    score = get_score(res.x)
                    if score > best_local_score:
                        best_local_score = score
                        best_v = res.x
                    # If valid, we can try to reduce mu to see if we can expand?
                    # No, if valid, penalty is 0. The optimizer pushes radii up until constraints bind.
                    # So it should be at a local optimum.
                    # We can break out of the penalty loop if no improvement in score after a few steps?
                    # Or just accept and try next restart.
                else:
                    mu *= 2.0
                    best_v = res.x
            else:
                break
                
        if is_valid(best_v):
            score = get_score(best_v)
            if score > best_score:
                best_score = score
                best_solution = best_v.copy()
                
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    x = best_solution[0::3]
    y = best_solution[1::3]
    r = best_solution[2::3]
    
    # Final validity check and shrink
    valid = True
    for i in range(N):
        if r[i] > x[i] or r[i] > 1-x[i] or r[i] > y[i] or r[i] > 1-y[i]:
            valid = False
            break
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j]:
                    valid = False
                    break
            if not valid: break
            
    if not valid:
        shrink = 1.0
        while not valid and shrink > 0.5:
            shrink *= 0.95
            r_new = r * shrink
            if np.any(r_new > x) or np.any(r_new > 1-x) or np.any(r_new > y) or np.any(r_new > 1-y):
                continue
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    if dist < r_new[i] + r_new[j]:
                        valid = False
                        break
                if not valid: break
            if valid:
                r = r_new
                break
        if not valid:
             side = int(np.ceil(np.sqrt(N)))
             cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
             return np.array(cells[:N])
             
    result = np.column_stack((x, y, r))
    return result
