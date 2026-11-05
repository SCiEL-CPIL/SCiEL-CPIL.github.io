import time
import random
import numpy as np
try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_SCIPY_MILP = True
except ImportError:
    HAS_SCIPY_MILP = False

def construct():
    """
    Constructs a subset of [1, 256] with no non-constant 3-term arithmetic progression
    that maximizes the sum of reciprocals.
    
    Strategy:
    1. Formulate as Integer Linear Programming (ILP).
    2. Solve using scipy.optimize.milp (HiGHS).
    3. Fallback to heuristic if ILP fails or is unavailable.
    """
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Weights: 1/i
    weights = np.array([0.0] + [1.0 / i for i in range(1, n + 1)])
    # Objective is to maximize sum(weights[i] * x[i])
    # milp minimizes, so we minimize -sum(weights[i] * x[i])
    c = -weights[1:]  # length 256
    
    # Generate constraints: x_a + x_b + x_c <= 2 for all 3-APs
    # We need the constraint matrix A and upper bounds b
    # A is sparse? scipy.optimize.milp accepts dense or sparse.
    # 16512 rows, 256 cols. Dense is 16512 * 256 * 8 bytes ~ 33 MB. Acceptable.
    
    ap_list = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c_val = a + 2 * d
            # Indices in 0-based array for variables x_1..x_256 are 0..255
            # a, b, c_val are 1-based
            ap_list.append((a - 1, b - 1, c_val - 1))
    
    m = len(ap_list)
    A = np.zeros((m, n))
    for i, (u, v, w) in enumerate(ap_list):
        A[i, u] = 1.0
        A[i, v] = 1.0
        A[i, w] = 1.0
    
    # Constraints: A x <= 2
    # LinearConstraint(A, lb, ub)
    # lb = -inf, ub = 2
    lincons = LinearConstraint(A, lb=-np.inf, ub=2)
    bounds = Bounds(lb=0, ub=1)
    
    best_result = None
    
    if HAS_SCIPY_MILP:
        try:
            res = milp(c=c, constraints=lincons, integrality=np.ones(n, dtype=int), bounds=bounds)
            if res.status == 0:
                # res.x is the optimal solution
                sol = res.x
                # Convert to set of 1-based indices
                result_set = [int(i + 1) for i in range(n) if sol[i] > 0.5]
                if len(result_set) > 0:
                    best_result = result_set
            else:
                # If status is not 0, check if it found a feasible solution?
                # status 2: Optimal solution found.
                # status 3: Feasible solution found but not optimal?
                # Actually, HiGHS status: 0=Optimal, 1=Infeasible, 2=Unbounded, etc.
                # If status is 0, we are good.
                pass
        except Exception as e:
            pass # Fallback to heuristic
    
    if best_result is None:
        # Fallback: Simple Greedy + Local Search
        # This is the original logic, simplified for fallback
        in_set = [False] * (n + 1)
        block_count = [0] * (n + 1)
        set_list = []
        
        def add(x):
            if in_set[x] or block_count[x] > 0:
                return
            in_set[x] = True
            set_list.append(x)
            for z in set_list[:-1]:
                if (x + z) % 2 == 0:
                    m_id = (x + z) // 2
                    if 1 <= m_id <= n and m_id != x and m_id != z:
                        block_count[m_id] += 1
                y = 2 * z - x
                if 1 <= y <= n and y != x and y != z:
                    block_count[y] += 1
                y = 2 * x - z
                if 1 <= y <= n and y != x and y != z:
                    block_count[y] += 1

        def remove(x):
            if not in_set[x]:
                return
            in_set[x] = False
            set_list.remove(x)
            for z in set_list:
                if (x + z) % 2 == 0:
                    m_id = (x + z) // 2
                    if 1 <= m_id <= n and m_id != x and m_id != z:
                        block_count[m_id] -= 1
                y = 2 * z - x
                if 1 <= y <= n and y != x and y != z:
                    block_count[y] -= 1
                y = 2 * x - z
                if 1 <= y <= n and y != x and y != z:
                    block_count[y] -= 1

        # Greedy fill
        for x in range(1, n + 1):
            if not in_set[x] and block_count[x] == 0:
                add(x)
        
        # Simple local search
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            current_sum = sum(1.0 / i for i in set_list)
            S = set_list[:]
            for x in S:
                if time.time() > deadline - 5.0:
                    break
                remove(x)
                best_y = -1
                best_gain = 0.0
                for y in range(1, n + 1):
                    if not in_set[y] and block_count[y] == 0 and y != x:
                        gain = 1.0 / y
                        if gain > 1.0 / x:
                            if best_y == -1 or gain > 1.0 / best_y:
                                best_y = y
                                best_gain = gain
                if best_y != -1:
                    add(best_y)
                    if sum(1.0 / i for i in set_list) > current_sum + 1e-10:
                        improved = True
                        break
                else:
                    add(x)
            if not improved:
                # Greedy fill again
                for x in range(1, n + 1):
                    if not in_set[x] and block_count[x] == 0:
                        add(x)
        
        best_result = set_list[:]

    best_result.sort()
    
    # Final validation
    assert len(best_result) > 0
    s_set = set(best_result)
    for i, a in enumerate(best_result):
        for b in best_result[i+1:]:
            if (a + b) % 2 == 0:
                mid = (a + b) // 2
                if mid != a and mid != b and mid in s_set:
                    raise AssertionError("3-AP found")
    
    return best_result
