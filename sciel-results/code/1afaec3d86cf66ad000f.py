<<<<<<< SEARCH
import numpy as np

def construct():
    n, m, p = 2, 4, 6
=======
import numpy as np
import time
from fractions import Fraction
import itertools

def construct():
    _start = time.time()
    _DEADLINE = 140.0  # safety margin before 150s kill

    def _time_left():
        return _DEADLINE - (time.time() - _start)

    n, m, p = 2, 4, 6
>>>>>>> REPLACE

