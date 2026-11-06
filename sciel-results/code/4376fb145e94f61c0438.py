"""Solve: factor 12! into 1..12 DISTINCT integers > 12, minimize the LARGEST factor."""
# EVOLVE-BLOCK-START
import time
from math import factorial
from scipy.optimize import milp, LinearConstraint, Bounds

def _all_divisors(n):
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            if i > 12:
                divs.append(i)
            if i != n // i and n // i > 12:
                divs.append(n // i)
        i += 1
    return sorted(divs)

def construct():
    deadline = time.time() + 80  # safety margin under 90s kill
    target = factorial(12)
    divs = _all_divisors(target)
    n = len(divs)

    # Precompute log2 of target and each divisor for log-space product constraint
    import math
    log_target = math.log2(target)
    log_divs = [math.log2(d) for d in divs]

    # Binary selection variables x_i in {0,1}
    # Product constraint (in log space): sum(x_i * log_divs_i) == log_target
    # Distinctness is automatic since each divisor appears once.
    # Count constraint: 1 <= sum(x_i) <= 12
    # Minimize largest: introduce t >= each selected divisor.
    #   t - x_i * d_i >= 0  is NOT linear in the right way.
    # Instead: minimize t subject to t >= d_i * x_i for all i.
    #   t - d_i * x_i >= 0  for all i  (if x_i=1 then t>=d_i; if x_i=0 then t>=0, ok)
    # This is linear! c = [0,...,0,1] (last var is t)

    # Variables: x_0..x_{n-1}, t (index n)
    # c: minimize t
    c = [0.0] * n + [1.0]

    # Constraints:
    # 1) sum(x_i * log_divs_i) == log_target
    # 2) sum(x_i) >= 1
    # 3) sum(x_i) <= 12
    # 4) t - d_i * x_i >= 0  for all i  ->  t >= d_i * x_i

    A_eq = [[log_divs[i] for i in range(n)] + [0.0]]
    b_eq = [log_target]

    A_ub = []
    b_ub = []
    # sum(x_i) <= 12
    A_ub.append([1.0] * n + [0.0])
    b_ub.append(12.0)
    # sum(x_i) >= 1  ->  -sum(x_i) <= -1
    A_ub.append([-1.0] * n + [0.0])
    b_ub.append(-1.0)
    # t - d_i * x_i >= 0  ->  -t + d_i * x_i <= 0  for each i
    for i in range(n):
        row = [0.0] * n + [-1.0]
        row[i] = divs[i]
        A_ub.append(row)
        b_ub.append(0.0)

    integrality = [1] * n + [0]  # x_i integer, t continuous
    lb = [0.0] * n + [0.0]
    ub = [1.0] * n + [float('inf')]

    cons = LinearConstraint(A_ub, -float('inf'), b_ub)
    cons_eq = LinearConstraint(A_eq, b_eq, b_eq)
    bounds = Bounds(lb, ub)

    res = milp(c=c, constraints=[cons, cons_eq], integrality=integrality, bounds=bounds)

    if time.time() > deadline:
        # Fallback: return single factor (valid but suboptimal)
        return [target]

    if res.status != 0:
        # Fallback
        return [target]

    x = res.x
    selected = [divs[i] for i in range(n) if x[i] > 0.5]
    if not selected:
        return [target]
    # Verify product
    prod = 1
    for v in selected:
        prod *= v
    if prod != target:
        # Numerical issue; try rounding
        return [target]
    return selected
# EVOLVE-BLOCK-END
