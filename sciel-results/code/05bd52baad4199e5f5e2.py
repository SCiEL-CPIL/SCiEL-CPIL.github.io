<<<<<<< SEARCH
"""Rewritten program for large reciprocal-subsum-free sets."""
import time
from fractions import Fraction
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)
=======
"""Rewritten program for large reciprocal-subsum-free sets."""
import time
import random
from fractions import Fraction
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)
>>>>>>> REPLACE

