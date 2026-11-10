"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction

# EVOLVE-BLOCK-START

N = 256

def _is_3ap_free(S):
    """Check if set S has no 3-term arithmetic progression."""
    S_set = set(S)
    for a in S:
        for b in S:
            if a < b:
                c = 2 * b - a
                if c in S_set:
                    return False
    return True

def _reciprocal_sum(S):
    """Compute exact reciprocal sum as Fraction."""
    total = Fraction(0, 1)
    for x in S:
        total += Fraction(1, x)
    return total

def _greedy_construct():
    """Greedy construction: add elements in order, skip if creates 3-AP."""
    selected = []
    selected_set = set()
    for x in range(1, N + 1):
        # Check if adding x creates a 3-AP
        creates_ap = False
        for a in selected:
            for b in selected:
                if a < b:
                    c = 2 * b - a
                    if c == x:
                        creates_ap = True
                        break
            if creates_ap:
                break
        if not creates_ap:
            selected.append(x)
            selected_set.add(x)
    return selected

def _greedy_by_weight():
    """Greedy construction: sort by 1/x descending (i.e., x ascending), add if no 3-AP."""
    selected = []
    selected_set = set()
    # Process in order of increasing x (highest reciprocal weight first)
    for x in range(1, N + 1):
        creates_ap = False
        for a in selected:
            for b in selected:
                if a < b:
                    c = 2 * b - a
                    if c == x:
                        creates_ap = True
                        break
            if creates_ap:
                break
        if not creates_ap:
            selected.append(x)
            selected_set.add(x)
    return selected

def _local_search(initial_set, time_limit=30):
    """Local search to improve reciprocal sum by swapping elements."""
    import random
    
    start_time = time.time()
    
    # Convert to set for quick lookup
    S = set(initial_set)
    current_weight = sum(Fraction(1, x) for x in S)
    best_weight = current_weight
    best_set = set(S)
    
    # Precompute for each x, the pairs (a,b) such that 2b-a = x or 2a-x = b or x+... 
    # Actually, let's think about it differently.
    # When we remove an element, we free up some constraints.
    # When we add an element, we need to check it doesn't form a 3-AP.
    
    def can_add(x, S_current):
        """Check if x can be added to S_current without creating a 3-AP."""
        for a in S_current:
            for b in S_current:
                if a < b:
                    c = 2 * b - a
                    if c == x:
                        return False
        return True
    
    def check_all(S_current):
        """Verify S_current is 3-AP free."""
        return _is_3ap_free(list(S_current))
    
    # Local search: try removing one element and adding another
    # We want to maximize sum of 1/x, so prefer smaller numbers
    # But removing a small number loses a lot, so maybe we should remove larger numbers
    
    for iteration in range(5000):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Try each element in the set for removal
        for u in list(S):
            # Try removing u
            S_without_u = S - {u}
            
            # Try adding each element not in S
            for v in range(1, N + 1):
                if v in S_without_u:
                    continue
                if can_add(v, S_without_u):
                    S_candidate = S_without_u | {v}
                    # Verify it's valid
                    if check_all(S_candidate):
                        w_candidate = current_weight - Fraction(1, u) + Fraction(1, v)
                        if w_candidate > current_weight:
                            S = S_candidate
                            current_weight = w_candidate
                            if current_weight > best_weight:
                                best_weight = current_weight
                                best_set = set(S)
                            improved = True
                            break
            if improved:
                break
        
        if not improved:
            break
    
    return list(best_set)

def _two_phase_local_search(initial_set, time_limit=45):
    """Two-phase local search with more aggressive swaps."""
    import random
    
    start_time = time.time()
    
    S = set(initial_set)
    current_weight = sum(Fraction(1, x) for x in S)
    best_weight = current_weight
    best_set = set(S)
    
    def can_add(x, S_current):
        for a in S_current:
            for b in S_current:
                if a < b:
                    c = 2 * b - a
                    if c == x:
                        return False
        return True
    
    for iteration in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Randomly pick an element to remove
        u = random.choice(list(S))
        S_without_u = S - {u}
        
        # Try to add the best possible element
        best_v = None
        best_gain = Fraction(0, 1)
        for v in range(1, N + 1):
            if v in S_without_u:
                continue
            if can_add(v, S_without_u):
                gain = Fraction(1, v) - Fraction(1, u)
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
        
        if best_v is not None and best_gain > 0:
            S = S_without_u | {best_v}
            current_weight += best_gain
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(S)
            improved = True
        
        if not improved:
            break
    
    return list(best_set)

def _simulated_annealing(initial_set, time_limit=40):
    """Simulated annealing to escape local optima."""
    import random
    import math
    
    start_time = time.time()
    
    S = set(initial_set)
    current_weight = sum(Fraction(1, x) for x in S)
    best_weight = current_weight
    best_set = set(S)
    
    T = 0.5  # Initial temperature
    cooling_rate = 0.995
    
    def can_add(x, S_current):
        for a in S_current:
            for b in S_current:
                if a < b:
                    c = 2 * b - a
                    if c == x:
                        return False
        return True
    
    while time.time() - start_time < time_limit:
        T *= cooling_rate
        if T < 1e-8:
            T = 1e-8
        
        # Propose a random move: remove one, add one
        u = random.choice(list(S))
        S_without_u = S - {u}
        
        candidates = []
        for v in range(1, N + 1):
            if v in S_without_u:
                continue
            if can_add(v, S_without_u):
                candidates.append(v)
        
        if not candidates:
            continue
        
        v = random.choice(candidates)
        S_new = S_without_u | {v}
        
        # Verify validity
        if not _is_3ap_free(list(S_new)):
            continue
        
        delta = (Fraction(1, v) - Fraction(1, u))
        delta_float = float(delta)
        
        # Accept if improvement, or with probability exp(delta/T)
        if delta_float >= 0 or random.random() < math.exp(delta_float / T):
            S = S_new
            current_weight += delta
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(S)
    
    return list(best_set)

def _milp_solve(time_limit=60):
    """Try to solve using ILP via scipy.optimize.milp."""
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
    except ImportError:
        return None
    
    # Variables: x_i for i in 1..N, binary
    # Objective: maximize sum(1/i * x_i)
    # Constraints: for each 3-AP (a,b,c), x_a + x_b + x_c <= 2
    
    # Find all 3-APs
    constraints_list = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            c = 2 * b - a
            if c <= N and c > b:
                # This is a valid 3-AP
                row = np.zeros(N)
                row[a-1] = 1
                row[b-1] = 1
                row[c-1] = 1
                constraints_list.append(row)
    
    if not constraints_list:
        return list(range(1, N+1))
    
    A = np.array(constraints_list)
    
    # Objective: minimize -sum(1/i * x_i)
    c_obj = -np.array([1.0/i for i in range(1, N+1)])
    
    # Constraints: A @ x <= 2
    ub = np.ones(A.shape[0]) * 2
    lb = np.full(A.shape[0], -np.inf)
    
    # Bounds on x: 0 or 1
    lb_vars = np.zeros(N)
    ub_vars = np.ones(N)
    
    integrality = np.ones(N)
    
    try:
        res = milp(
            c=c_obj,
            constraints=LinearConstraint(A, lb, ub),
            integrality=integrality,
            bounds=Bounds(lb_vars, ub_vars),
            options={"time_limit": time_limit}
        )
        
        if res.success or res.status == 11:  # 11 means time limit reached with feasible solution
            x_sol = res.x
            selected = [i+1 for i in range(N) if x_sol[i] > 0.5]
            if selected:
                return selected
    except Exception as e:
        pass
    
    return None

def _refine_with_local_search(S, time_limit=20):
    """Final refinement pass with local search."""
    S_set = set(S)
    
    # Try to add any missing elements
    for x in range(1, N+1):
        if x not in S_set:
            can_add = True
            for a in S_set:
                for b in S_set:
                    if a < b:
                        c = 2*b - a
                        if c == x:
                            can_add = False
                            break
                if not can_add:
                    break
            if can_add:
                S_set.add(x)
    
    # Try to swap
    start = time.time()
    for _ in range(200):
        if time.time() - start > time_limit:
            break
        improved = False
        for u in list(S_set):
            S_wo_u = S_set - {u}
            for v in range(1, N+1):
                if v in S_wo_u:
                    continue
                can_add = True
                for a in S_wo_u:
                    for b in S_wo_u:
                        if a < b:
                            c = 2*b - a
                            if c == v:
                                can_add = False
                                break
                    if not can_add:
                        break
                if can_add:
                    # Check if swap improves
                    if 1.0/v > 1.0/u:
                        S_set = S_wo_u | {v}
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    return list(S_set)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Phase 1: Get initial solution via greedy
    initial = _greedy_construct()
    best_set = initial
    best_weight = _reciprocal_sum(initial)
    
    # Phase 2: Try local search
    if time.time() < deadline - 10:
        ls_result = _local_search(initial, time_limit=15)
        if ls_result:
            ls_weight = _reciprocal_sum(ls_result)
            if ls_weight > best_weight:
                best_set = ls_result
                best_weight = ls_weight
    
    # Phase 3: Try simulated annealing from best so far
    if time.time() < deadline - 10:
        sa_result = _simulated_annealing(best_set, time_limit=30)
        if sa_result:
            sa_weight = _reciprocal_sum(sa_result)
            if sa_weight > best_weight:
                best_set = sa_result
                best_weight = sa_weight
    
    # Phase 4: Final refinement
    if time.time() < deadline - 2:
        refined = _refine_with_local_search(best_set, time_limit=5)
        if refined:
            ref_weight = _reciprocal_sum(refined)
            if ref_weight > best_weight:
                best_set = refined
                best_weight = ref_weight
    
    # Verify final result
    if not _is_3ap_free(best_set):
        # Fallback to greedy
        best_set = _greedy_construct()
    
    return best_set

# EVOLVE-BLOCK-END
