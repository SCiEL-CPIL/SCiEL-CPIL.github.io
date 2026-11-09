<<<<<<< SEARCH
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
import sys

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _time_left():
    return _DEADLINE - time.time()
=======
import time
import numpy as np
from scipy.optimize import minimize
import sys

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _time_left():
    return _DEADLINE - time.time()
>>>>>>> REPLACE

