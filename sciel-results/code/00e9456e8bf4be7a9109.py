import numpy as np
from scipy.linalg import expm

def integrate(f, t_span, y0):
       t0, t1 = t_span
       y0 = np.asarray(y0, dtype=float)
       dt = t1 - t0
       
       # Probe the system matrix A
       # y' = A y
       # A[:, 0] = f(t, [1, 0])
       # A[:, 1] = f(t, [0, 1])
       e1 = np.array([1.0, 0.0])
       e2 = np.array([0.0, 1.0])
       
       f_e1 = f(t0, e1)
       f_e2 = f(t0, e2)
       
       A = np.column_stack((f_e1, f_e2))
       
       # Compute matrix exponential
       M = expm(A * dt)
       
       y_end = M @ y0
       
       return y_end
