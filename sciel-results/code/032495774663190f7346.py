import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Call f at least once
    v1 = f(t0, y0)
    
    N = 30
    # c[i, j] is the coefficient of (t-t0)^j for component i
    c = np.zeros((3, N + 1), dtype=float)
    c[:, 0] = y0
    c[:, 1] = v1
    
    # Precompute factorials or just use iterative division
    # c_{i, j+1} = (1/(j+1)) * ( ... )
    
    for j in range(1, N):
        # Compute P_j = sum_{i=0}^j c[0, i] * c[2, j-i]
        P_j = 0.0
        Q_j = 0.0
        for i in range(j + 1):
            P_j += c[0, i] * c[2, j - i]
            Q_j += c[0, i] * c[1, j - i]
            
        # Recurrences:
        # c[0, j+1] = (1/(j+1)) * 10 * (c[1, j] - c[0, j])
        c[0, j + 1] = (10.0 / (j + 1)) * (c[1, j] - c[0, j])
        
        # c[1, j+1] = (1/(j+1)) * (28 * c[0, j] - c[1, j] - P_j)
        c[1, j + 1] = (1.0 / (j + 1)) * (28.0 * c[0, j] - c[1, j] - P_j)
        
        # c[2, j+1] = (1/(j+1)) * (Q_j - (8.0/3.0) * c[2, j])
        c[2, j + 1] = (1.0 / (j + 1)) * (Q_j - (8.0 / 3.0) * c[2, j])
        
    # Evaluate polynomial at t1
    dt = t1 - t0
    y_end = np.zeros(3, dtype=float)
    for j in range(N + 1):
        y_end += c[:, j] * (dt ** j)
        
    return y_end
