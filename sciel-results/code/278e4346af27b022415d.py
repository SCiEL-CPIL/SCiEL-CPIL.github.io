import time
import random
import numpy as np
try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_SCIPY_MILP = True
except ImportError:
    HAS_SCIPY_MILP = False

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
        
    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True
        
    def solve_with_milp(S):
        if not HAS_SCIPY_MILP:
            return None
            
        N = len(S)
        val_to_idx = {v: i for i, v in enumerate(S)}
        
        A_rows = []
        b_rhs = []
        
        for i in range(N):
            for j in range(i, N):
                s = S[i] + S[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    row = np.zeros(N)
                    row[i] = 1.0
                    row[j] = 1.0
                    row[k] = 1.0
                    if i == j:
                        row[i] = 2.0
                    A_rows.append(row)
                    b_rhs.append(2.0)
                    
        if not A_rows:
            return [x for x in S]
            
        A = np.array(A_rows)
        b = np.array(b_rhs)
        
        c = -np.ones(N) # Minimize -sum(x) to maximize sum(x)
        
        constraints = LinearConstraint(A, b, 2.0) # A x <= 2 => A x - 2 <= 0? 
        # LinearConstraint(A, lb, ub) means lb <= A x <= ub
        # We want A x <= 2. So lb = -inf, ub = 2.
        constraints = LinearConstraint(A, -np.inf, b)
        
        integrality = np.ones(N)
        bounds = Bounds(0, 1)
        
        try:
            res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.status == 0:
                # res.x is the solution
                sol = res.x
                subset = [S[i] for i in range(N) if sol[i] > 0.5]
                return subset
            else:
                return None
        except Exception:
            return None
            
    def solve_with_heuristic(S, time_limit):
        # Fallback: Greedy + Local Search
        # ... (similar to current program but optimized)
        pass

    # Main logic
    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for size, seed in instances:
        S = instance(size, seed)
        
        # Try MILP first
        # MILP should be very fast for these sizes
        result = solve_with_milp(S)
        
        if result is None or not is_sum_free(result):
            # Fallback
            # Use a simple greedy
            result = [x for x in S if x % 2 != 0]
            
        results.append(result)
        
    return results
