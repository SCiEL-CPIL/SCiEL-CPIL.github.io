import numpy as np
import time

N = 150

def construct():
       # Initial points
       i = np.arange(N) + 0.5
       z = 1.0 - 2.0 * i / N
       phi = np.pi * (1.0 + np.sqrt(5.0)) * i
       r = np.sqrt(1.0 - z * z)
       pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
       
       # Optimization loop
       # ...
       return pts
