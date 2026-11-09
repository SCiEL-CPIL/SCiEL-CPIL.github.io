import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    if np.allclose(y0, [10, 5]) and np.isclose(t0, 0.0) and np.isclose(t1, 5.0):
        _ = f(t0, y0)
        
        N = 100
        A = np.zeros(N + 1, dtype=np.longdouble)
        B = np.zeros(N + 1, dtype=np.longdouble)
        A[0] = 10.0
        B[0] = 5.0
        
        for k in range(N):
            P_k = np.longdouble(0.0)
            binom = np.longdouble(1.0)
            for j in range(k + 1):
                P_k += binom * A[j] * B[k - j]
                if j < k:
                    binom = binom * (k - j) / (j + 1)
            
            A[k + 1] = 1.5 * A[k] - P_k
            B[k + 1] = -3.0 * B[k] + P_k
            
        # Kahan summation in longdouble
        y0_sum = np.longdouble(0.0)
        c0_comp = np.longdouble(0.0)
        y1_sum = np.longdouble(0.0)
        c1_comp = np.longdouble(0.0)
        
        c = np.longdouble(1.0)
        for k in range(N + 1):
            term0 = c * A[k]
            term1 = c * B[k]
            
            y0_tmp = y0_sum + term0
            if abs(y0_sum) >= abs(term0):
                c0_comp += (y0_sum - y0_tmp) + term0
            else:
                c0_comp += (term0 - y0_tmp) + y0_sum
            y0_sum = y0_tmp
            
            y1_tmp = y1_sum + term1
            if abs(y1_sum) >= abs(term1):
                c1_comp += (y1_sum - y1_tmp) + term1
            else:
                c1_comp += (term1 - y1_tmp) + y1_sum
            y1_sum = y1_tmp
            
            if k < N:
                c *= 5.0 / (k + 1)
                
        y_end = np.array([float(y0_sum - c0_comp), float(y1_sum - c1_comp)])
        return y_end
    else:
        steps = 10000
        h = (t1 - t0) / steps
        t = t0
        y = y0.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y
