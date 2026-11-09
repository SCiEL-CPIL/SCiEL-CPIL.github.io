"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from fractions import Fraction
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

DENOMS = list(range(2, 49))
N = len(DENOMS)

L = 1
for d in DENOMS:
    L = lcm(L, d)

WEIGHTS = [L // d for d in DENOMS]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Objective: Maximize sum of weights
    # c_i = -weight_i
    c = np.array([-w for w in WEIGHTS], dtype=float)
    
    # Constraint: sum(w_i * x_i) <= L - 1
    # A_ub = [w_1, w_2, ..., w_n]
    A_ub = np.array(WEIGHTS, dtype=float)
    b_ub = L - 1
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: x_i must be integer (0 or 1)
    integrality = np.ones(N, dtype=int)
    
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A_ub, ub=b_ub),
            integrality=integrality,
            bounds=bounds
        )
    except Exception:
        # Fallback to greedy if MILP fails
        best_sum = 0
        best_indices = []
        for i, w in enumerate(WEIGHTS):
            if best_sum + w < L:
                best_sum += w
                best_indices.append(i)
        if not best_indices:
            best_indices = [0]
        return best_indices

    if res.status == 0 and res.x is not None:
        # Extract indices where x_i == 1
        indices = [int(i) for i in range(N) if res.x[i] > 0.5]
        if indices:
            return indices
    
    # Fallback if MILP returns no solution or infeasible
    best_sum = 0
    best_indices = []
    for i, w in enumerate(WEIGHTS):
        if best_sum + w < L:
            best_sum += w
            best_indices.append(i)
    
    if not best_indices:
        best_indices = [0]
        
    return best_indices
# EVOLVE-BLOCK-END
