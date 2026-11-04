<<<<<<< SEARCH
import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-9):
=======
import numpy as np
import time

DIMENSIONS = [4, 4, 8]
_DEADLINE = time.time() + 110  # safety margin before 120s limit

def _time_left():
    return _DEADLINE - time.time()

def _verify(U, V, W, n, m, p, tol=1e-9):
>>>>>>> REPLACE

