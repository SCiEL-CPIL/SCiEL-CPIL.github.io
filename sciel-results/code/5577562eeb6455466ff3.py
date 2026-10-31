import numpy as np

N = 100

def construct():
       c = (np.sqrt(5) - 1) / 2
       # First N/2 steps are c, next N/2 are 1-c
       h = np.zeros(N)
       h[:N//2] = c
       h[N//2:] = 1 - c
       return h
