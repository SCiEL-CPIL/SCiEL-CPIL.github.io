<<<<<<< SEARCH
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    deadline = time.time() + 140
=======
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    deadline = time.time() + 140
    
    # Pre-compute Strassen coefficients for faster construction
    strassen_x = [
        [(0,0,1), (1,1,1)],      # M0
        [(1,0,1), (1,1,1)],      # M1
        [(0,0,1)],               # M2
        [(1,1,1)],               # M3
        [(0,0,1), (0,1,1)],      # M4
        [(1,0,1), (0,0,-1)],     # M5
        [(0,1,1), (1,1,-1)],     # M6
    ]
    strassen_y = [
        [(0,0,1), (1,1,1)],      # M0
        [(0,0,1)],               # M1
        [(0,1,1), (1,1,-1)],     # M2
        [(1,0,1), (0,0,-1)],     # M3
        [(1,1,1)],               # M4
        [(0,0,1), (0,1,1)],     # M5
        [(1,0,1), (1,1,1)],     # M6
    ]
    strassen_c = [
        [(0,0,1), (1,1,1)],      # C00 += M0, C11 += M0
        [(1,0,1), (1,1,-1)],     # C10 += M1, C11 -= M1
        [(0,1,1), (1,1,1)],      # C01 += M2, C11 += M2
        [(0,0,1), (1,0,1)],      # C00 += M3, C10 += M3
        [(0,0,-1), (0,1,1)],     # C00 -= M4, C01 += M4
        [(1,1,1)],               # C11 += M5
        [(0,0,1)],               # C00 += M6
    ]
>>>>>>> REPLACE

