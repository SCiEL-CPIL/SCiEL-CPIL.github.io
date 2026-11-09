<<<<<<< SEARCH
"""
Combined approach:
1. Structural idea from Inspiration 1: Use smooth, differentiable overlap penalties (SAT-based) and
   scipy.optimize.minimize (L-BFGS-B) to find tight packings. This is much more effective than
   discrete simulated annealing for this continuous problem.
2. Refinement machinery from Current Program:
   - Robust validity checking (though we rely on the optimizer to find valid states, we verify at the end).
   - Multi-start strategy with diverse topologies (Lattice, Random, Circular).
   - Explicit deadline handling.

Key improvements:
- The objective function minimizes the outer side length directly, using a smooth penalty for overlaps
  and containment violations. This allows the gradient-based optimizer to navigate the configuration space.
- We define the outer hexagon center and rotation as variables to be optimized as well, allowing the
  inner hexagons to be arranged optimally relative to the container.
"""

import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11
DEADLINE = 150.0
START_TIME = time.time()

def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)
=======
"""
Force-Directed Relaxation with Simulated Annealing.
This approach models the packing problem as a physical system where hexagons exert repulsive forces
when they overlap and attractive forces to stay within a compact region. The outer hexagon size is
minimized by treating it as a dynamic boundary that penalizes expansion.
"""

import math
import time
import random
import numpy as np

N = 11
DEADLINE = 150.0
START_TIME = time.time()

def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)
>>>>>>> REPLACE

