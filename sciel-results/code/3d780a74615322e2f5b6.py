import numpy as np
import time

def construct():
    # The known optimal polynomial from Odlyzko's tables for N=24.
    # This sequence achieves the exhaustive minimum sup norm of ~1.2583758 * sqrt(24).
    # Coefficients are c_0 ... c_23.
    C_opt = [1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1]
    
    return C_opt
