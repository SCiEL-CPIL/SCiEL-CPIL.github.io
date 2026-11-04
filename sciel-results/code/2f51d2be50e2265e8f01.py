import numpy as np
import scipy.optimize as opt
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    best_val = -np.inf
    best_x = None
    
    # Initial guess: Grid
    side = int(np.ceil(np.sqrt(N)))
    xs = []
    for i in range(side):
        for j in range(side):
            if len(xs) < N:
                xs.append([(i + 0.5) / side, (j + 0.5) / side, 0.45 / side])
    
    v0 = np.zeros(3*N + 1)
    for i, (cx, cy, r) in enumerate(xs):
        v0[3*i] = cx
        v0[3*i+1] = cy
        v0[3*i+2] = r
    v0[3*N] = 1.0 # W=1, H=1
    
    # Objective and Gradient
    def obj_grad(v):
        cx = v[0:3*N:3]
        cy = v[1:3*N:3]
        r = v[2:3*N:3]
        W = v[3*N]
        H = 2.0 - W
        
        grad = np.zeros_like(v)
        val = -np.sum(r)
        grad[2:3*N:3] -= 1.0
        
        lam_c = 100.0
        for i in range(N):
            for j in range(i+1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-8:
                    d = 1e-8
                
                dist = r[i] + r[j]
                if d < dist:
                    delta = dist - d
                    val += lam_c * delta * delta
                    
                    factor = 2.0 * lam_c * delta / d
                    grad[3*i]   += factor * (-dx)
                    grad[3*i+1] += factor * (-dy)
                    grad[3*j]   += factor * (dx)
                    grad[3*j+1] += factor * (dy)
                    
                    grad[3*i+2]   += 2.0 * lam_c * delta
                    grad[3*j+2]   += 2.0 * lam_c * delta
        
        lam_b = 100.0
        for i in range(N):
            if cx[i] < r[i]:
                delta = r[i] - cx[i]
                val += lam_b * delta * delta
                grad[3*i]   += 2.0 * lam_b * delta * (-1)
                grad[3*i+2] += 2.0 * lam_b * delta * (1)
            
            if cx[i] > W - r[i]:
                delta = cx[i] - (W - r[i])
                val += lam_b * delta * delta
                grad[3*i]   += 2.0 * lam_b * delta * (1)
                grad[3*N]   += 2.0 * lam_b * delta * (-1)
                grad[3*i+2] += 2.0 * lam_b * delta * (1)
            
            if cy[i] < r[i]:
                delta = r[i] - cy[i]
                val += lam_b * delta * delta
                grad[3*i+1] += 2.0 * lam_b * delta * (-1)
                grad[3*i+2] += 2.0 * lam_b * delta * (1)
            
            if cy[i] > H - r[i]:
                delta = cy[i] - (H - r[i])
                val += lam_b * delta * delta
                grad[3*i+1] += 2.0 * lam_b * delta * (1)
                grad[3*N]   += 2.0 * lam_b * delta * (1)
                grad[3*i+2] += 2.0 * lam_b * delta * (1)
        
        if W < 0.1:
            delta = 0.1 - W
            val += 1000.0 * delta * delta
            grad[3*N] += 2000.0 * delta * (-1)
        if W > 1.9:
            delta = W - 1.9
            val += 1000.0 * delta * delta
            grad[3*N] += 2000.0 * delta * (1)
            
        for i in range(N):
            if r[i] < 0.01:
                delta = 0.01 - r[i]
                val += 1000.0 * delta * delta
                grad[3*i+2] += 2000.0 * delta * (-1)
            
        return val, grad

    def fun(v):
        val, _ = obj_grad(v)
        return val

    def jac(v):
        _, grad = obj_grad(v)
        return grad

    restarts = 0
    while time.time() < deadline:
        restarts += 1
        if restarts == 1:
            v_init = v0.copy()
        else:
            v_init = v0 + np.random.normal(0, 0.1, size=3*N+1)
            v_init[3*N] = np.clip(v_init[3*N], 0.5, 1.5)
            # Ensure radii are positive
            v_init[2:3*N:3] = np.abs(v_init[2:3*N:3]) + 0.05
        
        try:
            res = opt.minimize(fun, v_init, jac=jac, method='L-BFGS-B', 
                                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            
            cx = res.x[0:3*N:3]
            cy = res.x[1:3*N:3]
            r = res.x[2:3*N:3]
            W = res.x[3*N]
            H = 2.0 - W
            
            valid = True
            if W < 0.01 or H < 0.01:
                valid = False
            if np.any(r < 0):
                valid = False
            
            for i in range(N):
                if cx[i] < r[i] - 1e-5 or cx[i] > W - r[i] + 1e-5 or cy[i] < r[i] - 1e-5 or cy[i] > H - r[i] + 1e-5:
                    valid = False
                    break
            
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
                        if d < r[i] + r[j] - 1e-5:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                score = np.sum(r)
                if score > best_val:
                    best_val = score
                    best_x = res.x.copy()
                    
        except Exception:
            pass

    if best_x is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    cx = best_x[0:3*N:3]
    cy = best_x[1:3*N:3]
    r = best_x[2:3*N:3]
    
    out = np.zeros((N, 3))
    out[:, 0] = cx
    out[:, 1] = cy
    out[:, 2] = r
    
    return out
