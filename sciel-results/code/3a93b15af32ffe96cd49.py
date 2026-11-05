<<<<<<< SEARCH
import numpy as np
import itertools
import time
import random
import sys

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1()
    
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4:
            return tetracode
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(w) for w in itertools.product(range(q), repeat=n)]
=======
import numpy as np
import itertools
import time
import random
import sys

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1()
    
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4:
            return tetracode
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(w) for w in itertools.product(range(q), repeat=n)]
>>>>>>> REPLACE

