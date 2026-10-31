import numpy as np

N = 50

def construct():
    # The ratio is maximized by a Dirac delta (spike) in the discrete case.
    # h = [0, ..., 0, 1, 0, ..., 0]
    # ||h||_2^2 = 1
    # ||h||_1 = 1
    # ||h*h||_inf = 1
    # Ratio = 1
    # Seed (Uniform) Ratio = 1/N = 0.02
    # Score = 1 / 0.02 = 50
    
    h = [0.0] * N
    # Place the spike in the center
    h[N // 2] = 1.0
    return h
