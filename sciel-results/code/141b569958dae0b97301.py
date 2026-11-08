<<<<<<< SEARCH
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
=======
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 120.0

def _is_timeout():
    return (time.time() - _START_TIME) > _TIME_LIMIT

DIMENSIONS = [4, 4, 8]

def construct():
    if _is_timeout():
        # Fallback to a simple valid decomposition if we ran out of time (unlikely)
        # Rank 128 naive decomposition
        U = [[0]*128 for _ in range(16)]
        V = [[0]*128 for _ in range(32)]
        W = [[0]*128 for _ in range(32)]
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    r = i*32 + j*8 + k
                    U[i*4+j][r] = 1
                    V[j*8+k][r] = 1
                    W[k*4+i][r] = 1
        return {"U": U, "V": V, "W": W}

    n, m, p = DIMENSIONS
>>>>>>> REPLACE

Now for the type casting.

