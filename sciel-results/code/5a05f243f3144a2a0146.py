"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 256
    # Weights: reciprocal of each number (use Fraction for exactness)
    weights = [Fraction(1, i) for i in range(1, N + 1)]
    
    # Build conflict graph: two numbers conflict if they can form a 3-AP with a third number in [1,N]
    # a, b, c form a 3-AP if a+c = 2b (a < b < c)
    # Two numbers x, y conflict if there exists z in [1,N] such that {x,y,z} forms a 3-AP
    
    # For efficiency, precompute conflicts using bitsets
    # Vertex i represents number i+1 (0-indexed internally)
    num_vertices = N
    
    # conflicts[i] is a bitset of vertices that conflict with vertex i
    conflicts = [0] * num_vertices
    
    for i in range(num_vertices):
        x = i + 1
        for j in range(num_vertices):
            if i == j:
                continue
            y = j + 1
            # Check if x and y can be part of a 3-AP with some z in [1,N]
            conflict = False
            # x is first, y is second: z = 2y - x, need z > y and z <= N
            z = 2 * y - x
            if y < z <= N:
                conflict = True
            # y is first, x is second: z = 2x - y, need z > x and z <= N
            z = 2 * x - y
            if x < z <= N:
                conflict = True
            # x is first, y is last (or vice versa): z = (x+y)/2
            s = x + y
            if s % 2 == 0:
                z = s // 2
                if 1 <= z <= N and z != x and z != y:
                    conflict = True
            
            if conflict:
                conflicts[i] |= (1 << j)
    
    # Greedy solution first for initial lower bound
    greedy_set = set()
    greedy_weight = Fraction(0, 1)
    order = sorted(range(num_vertices), key=lambda i: -float(weights[i]))
    for i in order:
        can_add = True
        for j in greedy_set:
            if (conflicts[j] >> i) & 1:
                can_add = False
                break
        if can_add:
            greedy_set.add(i)
            greedy_weight += weights[i]
    
    best_set = set(greedy_set)
    best_weight = greedy_weight
    
    # Formulate as MILP: maximize sum of weights subject to no 3-AP
    # Variables: x_i in {0,1} for each i in [1, N]
    # Objective: maximize sum of (1/i) * x_i
    # Constraints: for each 3-AP (a, b, c), x_a + x_b + x_c <= 2
    
    # Collect all 3-APs
    ap_list = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            c = 2 * b - a
            if c > b and c <= N:
                ap_list.append((a - 1, b - 1, c - 1))  # 0-indexed
    
    n_constraints = len(ap_list)
    
    # Objective: maximize sum of weights
    c_obj = np.array([-float(weights[i]) for i in range(num_vertices)])  # negative for minimization
    
    # Constraints: A @ x <= b
    # Each row: x_a + x_b + x_c <= 2
    A = np.zeros((n_constraints, num_vertices))
    for idx, (a, b, c) in enumerate(ap_list):
        A[idx, a] = 1
        A[idx, b] = 1
        A[idx, c] = 1
    b_vec = np.ones(n_constraints) * 2
    
    # Bounds: 0 <= x_i <= 1
    lb = np.zeros(num_vertices)
    ub = np.ones(num_vertices)
    
    # Integrality: all variables are integer
    integrality = np.ones(num_vertices)
    
    # Time limit for solver
    time_remaining = deadline - time.time() - 5.0  # 5s buffer
    if time_remaining < 1.0:
        time_remaining = 1.0
    
    try:
        res = milp(
            c=c_obj,
            constraints=LinearConstraint(A, ub=b_vec),
            integrality=integrality,
            bounds=Bounds(lb, ub),
            options={"time_limit": time_remaining, "mip_rel_gap": 0.0}
        )
        
        if res.status == 0:  # optimal
            # Extract solution
            x = res.x
            milp_set = set()
            for i in range(num_vertices):
                if x[i] > 0.5:
                    milp_set.add(i)
            
            if len(milp_set) > 0:
                milp_weight = sum(weights[i] for i in milp_set)
                if milp_weight > best_weight:
                    best_weight = milp_weight
                    best_set = milp_set
        elif res.status == 2:  # locally optimal
            x = res.x
            milp_set = set()
            for i in range(num_vertices):
                if x[i] > 0.5:
                    milp_set.add(i)
            
            if len(milp_set) > 0:
                milp_weight = sum(weights[i] for i in milp_set)
                if milp_weight > best_weight:
                    best_weight = milp_weight
                    best_set = milp_set
    except Exception as e:
        pass
    
    # Extract best set
    result = sorted([i + 1 for i in best_set])
    
    if not result:
        # Fallback to greedy
        result = sorted([i + 1 for i in greedy_set])
    
    return result
# EVOLVE-BLOCK-END
