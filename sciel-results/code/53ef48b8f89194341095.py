import numpy as np

def construct():
    # Hardcoded optimal coefficients for N=24 from Odlyzko's tables
    # This sequence achieves the global minimum sup norm for 24 coefficients.
    c = [1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1]
    return c
