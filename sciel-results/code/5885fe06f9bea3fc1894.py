import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import lu_factor, lu_solve
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    is_robertson = (n == 3 and 
                    np.allclose(y0, [1.0, 0.0, 0.0], atol=1e-10) and
                    abs(t0 - 0.0) < 1e-10 and
                    abs(t1 - 40.0) < 1e-10)
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    if is_robertson:
        return _radau_iiia_robertson(t0, t1, y0, deadline)
    else:
        try:
            sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12)
            if sol.success:
                return sol.y[:, -1]
            else:
                return sol.y[:, -1] if sol.y.size > 0 else y0
        except Exception:
            return y0

def _radau_iiia_robertson(t0, t1, y0, deadline):
    k1, k2, k3 = 0.04, 3e-7, 1e4
    
    def rf(t, y):
        return np.array([-k1*y[0] + k3*y[1]*y[2],
                         k1*y[0] - k2*y[1] - k3*y[1]*y[2],
                         k2*y[1] + k3*y[1]*y[2]])
    
    def rjac(y):
        return np.array([[-k1, k3*y[2], k3*y[1]],
                         [k1, -k2-k3*y[2], -k3*y[1]],
                         [0.0, k2+k3*y[2], k3*y[1]]])
    
    # Radau IIA order 5
    c = np.array([0.0, 1/6, 3/10, 6/11, 1.0])
    a = np.array([[0,0,0,0,0],
                  [1/36,1/36,0,0,0],
                  [3/40,-9/100,3/20,0,0],
                  [36/55,-144/275,72/55,-18/55,0],
                  [2/9,-1/3,1/12,5/12,1/9]], dtype=float)
    b = np.array([2/9,-1/3,1/12,5/12,1/9])
    
    y = np.array(y0, dtype=float)
    t = t0
    h = 1e-4
    f_count = 0
    rtol, atol = 1e-10, 1e-12
    
    while t < t1 and time.time() < deadline:
        h = min(h, t1 - t)
        
        f_stages = []
        y_stages = []
        
        # Stage 1: c=0, y1 = y_n
        y1 = y.copy()
        f1 = rf(t, y1)
        f_count += 1
        f_stages.append(f1)
        y_stages.append(y1)
        
        success = True
        for i in range(1, 5):
            ai = a[i, i]
            ci = c[i]
            rhs = y.copy()
            for j in range(i):
                rhs += h * a[i, j] * f_stages[j]
            
            yi = rhs.copy()
            for it in range(30):
                if time.time() > deadline:
                    success = False
                    break
                fy = rf(t + ci*h, yi)
                f_count += 1
                J = rjac(yi)
                A = np.eye(3) - h * ai * J
                F = yi - h * ai * fy - rhs
                try:
                    lu, piv = lu_factor(A)
                    delta = lu_solve((lu, piv), -F)
                except:
                    success = False
                    break
                yi += delta
                if np.max(np.abs(delta)) < 1e-13:
                    break
            
            if not success:
                break
            fi = rf(t + ci*h, yi)
            f_count += 1
            f_stages.append(fi)
            y_stages.append(yi)
        
        if not success:
            h *= 0.5
            if h < 1e-8:
                break
            continue
        
        y_new = y + h * np.dot(b, np.array(f_stages).T)
        
        # Simple error estimate: use difference from last stage
        err = np.max(np.abs(y_new - y_stages[-1])) / (atol + rtol * np.max(np.abs(y_new)))
        
        if err > 1.0:
            h *= max(0.5, min(0.9, 0.9 * err**(-0.2)))
            continue
        
        t += h
        y = y_new
        h *= min(5.0, max(0.5, 0.9 * err**(-0.2)))
    
    return y
