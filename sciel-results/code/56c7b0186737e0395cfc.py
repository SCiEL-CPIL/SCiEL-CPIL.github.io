import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    deadline = time.time() + 600.0
    best_score = -np.inf
    best_x = None
    
    # Initial configurations
    inits = []
    
    # 1. 3x3 grid + 1 small (current best)
    s = 1.0/3.0 - 1e-6
    t = 1.0 - 3.0*s
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    idx = 0
    for r in range(3):
        for c in range(3):
            centers[idx] = [(c + 0.5)*s, (r + 0.5)*s]
            sides[idx] = s
            idx += 1
    centers[idx] = [1.0 - 0.5*t, 1.0 - 0.5*t]
    sides[idx] = t
    x0 = np.concatenate([centers.flatten(), sides])
    inits.append(x0)
    
    # 2. Random
    np.random.seed(42)
    for _ in range(5):
        cx = np.random.rand(n)
        cy = np.random.rand(n)
        s = np.random.rand(n) * 0.4 + 0.1
        x0 = np.concatenate([cx, cy, s])
        inits.append(x0)
        
    # 3. 4x4 grid
    s = 0.25
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    idx = 0
    for r in range(4):
        for c in range(4):
            if idx >= 10: break
            centers[idx] = [(c + 0.5)*s, (r + 0.5)*s]
            sides[idx] = s
            idx += 1
    x0 = np.concatenate([centers.flatten(), sides])
    inits.append(x0)
    
    beta = 50.0 # Steepness of sigmoid
    
    def penalty_and_grad(x):
        cx = x[0:n]
        cy = x[n:2*n]
        s = x[2*n:3*n]
        
        # Ensure s > 0
        s = np.maximum(s, 1e-6)
        
        obj = -np.sum(s)
        
        # Boundary penalties
        P_b = 0.0
        G_b = np.zeros_like(x)
        
        for i in range(n):
            # x bounds
            lo = s[i]/2.0
            hi = 1.0 - s[i]/2.0
            if cx[i] < lo:
                P_b += (lo - cx[i])**2
                G_b[i] += -2*(lo - cx[i]) # d/dc (lo-c)^2 = -2(lo-c)
                # d/ds term
                G_b[2*n+i] += -2*(lo - cx[i]) * 0.5
            elif cx[i] > hi:
                P_b += (cx[i] - hi)**2
                G_b[i] += 2*(cx[i] - hi)
                G_b[2*n+i] += -2*(cx[i] - hi) * 0.5
            
            # y bounds
            if cy[i] < lo:
                P_b += (lo - cy[i])**2
                G_b[n+i] += -2*(lo - cy[i])
                G_b[2*n+i] += -2*(lo - cy[i]) * 0.5
            elif cy[i] > hi:
                P_b += (cy[i] - hi)**2
                G_b[n+i] += 2*(cy[i] - hi)
                G_b[2*n+i] += -2*(cy[i] - hi) * 0.5
        
        # Overlap penalties
        P_o = 0.0
        G_o = np.zeros_like(x)
        
        for i in range(n):
            for j in range(i+1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                r_ij = (s[i] + s[j]) / 2.0
                
                u = abs(dx) - r_ij
                v = abs(dy) - r_ij
                
                # Sigmoid approximation of "violation" (1 if u<0 and v<0)
                # p_u = 1 / (1 + exp(beta * u))
                # p_v = 1 / (1 + exp(beta * v))
                # P_ij = p_u * p_v
                
                exp_u = np.exp(beta * u)
                exp_v = np.exp(beta * v)
                
                p_u = 1.0 / (1.0 + exp_u)
                p_v = 1.0 / (1.0 + exp_v)
                
                P_ij = p_u * p_v
                P_o += P_ij
                
                # Gradients
                # dP/du = -beta * exp_u / (1 + exp_u)^2 * p_v = -beta * p_u * (1 - p_u) * p_v
                # dP/dv = -beta * p_u * p_v * (1 - p_v)
                
                dP_du = -beta * p_u * (1.0 - p_u) * p_v
                dP_dv = -beta * p_u * p_v * (1.0 - p_v)
                
                # d u / d cx_i = sign(dx)
                # d u / d cx_j = -sign(dx)
                # d u / d s_i = -0.5
                # d u / d s_j = -0.5
                
                sign_dx = 1.0 if dx >= 0 else -1.0
                sign_dy = 1.0 if dy >= 0 else -1.0
                
                G_o[i] += dP_du * sign_dx
                G_o[j] += dP_du * (-sign_dx)
                G_o[n+i] += dP_dv * sign_dy
                G_o[n+j] += dP_dv * (-sign_dy)
                
                G_o[2*n+i] += dP_du * (-0.5) + dP_dv * (-0.5)
                G_o[2*n+j] += dP_du * (-0.5) + dP_dv * (-0.5)
                
        # Weighted sum
        lambda_b = 10.0
        lambda_o = 10.0
        
        total_obj = obj + lambda_b * P_b + lambda_o * P_o
        total_grad = G_b * lambda_b + G_o * lambda_o
        
        return total_obj, total_grad

    for x0 in inits:
        if time.time() > deadline - 60:
            break
            
        try:
            res = minimize(penalty_and_grad, x0, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < best_score:
                x_opt = res.x
                cx = x_opt[0:n]
                cy = x_opt[n:2*n]
                s = x_opt[2*n:3*n]
                
                # Validate and refine
                # Check if valid
                valid = True
                for i in range(n):
                    if s[i] <= 0:
                        valid = False
                        break
                    if cx[i] - s[i]/2 < -1e-7 or cx[i] + s[i]/2 > 1 + 1e-7:
                        valid = False
                        break
                    if cy[i] - s[i]/2 < -1e-7 or cy[i] + s[i]/2 > 1 + 1e-7:
                        valid = False
                        break
                
                if valid:
                    for i in range(n):
                        for j in range(i+1, n):
                            if abs(cx[i]-cx[j]) < (s[i]+s[j])/2 - 1e-7 and \
                               abs(cy[i]-cy[j]) < (s[i]+s[j])/2 - 1e-7:
                                valid = False
                                break
                        if not valid: break
                
                if valid:
                    score = np.sum(s)
                    if score > best_score:
                        best_score = score
                        best_x = x_opt.copy()
                        
        except Exception as e:
            pass

    if best_x is not None:
        cx = best_x[0:n]
        cy = best_x[n:2*n]
        s = best_x[2*n:3*n]
        centers = np.column_stack((cx, cy))
        return centers, s
    else:
        # Fallback to initial best
        return inits[0][:2*n].reshape(10,2), inits[0][2*n:]
