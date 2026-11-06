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
        
    def solve_with_milp(S, time_budget):
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
        
        constraints = LinearConstraint(A, -np.inf, b)
        
        integrality = np.ones(N)
        bounds = Bounds(0, 1)
        
        try:
            res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds,
                       options={'time_limit': time_budget})
            if res.status == 0:
                sol = res.x
                subset = [S[i] for i in range(N) if sol[i] > 0.5]
                return subset
            else:
                return None
        except Exception:
            return None
            
    def solve_with_heuristic(S, time_limit):
        # Greedy + Local Search
        S_set = set(S)
        # Start with all odd numbers (sum of two odds is even, so odd+odd != odd)
        # Actually, sum of two odds is even, so if we only take odds, no two odds sum to an odd.
        # But we need to check if any even is in the set. If we only take odds, it's sum-free.
        # Wait, a+b=c. If a,b are odd, a+b is even. c is odd. So a+b != c.
        # So the set of all odd numbers in S is sum-free.
        
        # Better heuristic: Greedy by value, then local search
        # Sort by absolute value descending? Or try multiple orderings.
        
        def greedy(order):
            S_local = set()
            for x in order:
                # Check if adding x creates a conflict
                # Conflict if there exist a, b in S_local such that a+b=x or a+x=b or x+b=a
                # Since S_local is sum-free, we just need to check:
                # 1. x+x in S_local
                # 2. For each a in S_local, a+x in S_local
                can_add = True
                if x + x in S_local:
                    can_add = False
                else:
                    for a in S_local:
                        if a + x in S_local:
                            can_add = False
                            break
                if can_add:
                    S_local.add(x)
            return S_local
        
        # Try a few different orderings
        best = set()
        
        # Ordering 1: All odds first, then evens
        odds = [x for x in S if x % 2 != 0]
        evens = [x for x in S if x % 2 == 0]
        order1 = odds + evens
        sol1 = greedy(order1)
        if len(sol1) > len(best):
            best = sol1
        
        # Ordering 2: Sorted by absolute value descending
        order2 = sorted(S, key=lambda x: -abs(x))
        sol2 = greedy(order2)
        if len(sol2) > len(best):
            best = sol2
        
        # Ordering 3: Sorted by value
        order3 = sorted(S)
        sol3 = greedy(order3)
        if len(sol3) > len(best):
            best = sol3
        
        # Local search: try to swap elements
        # We'll do a simple hill climbing
        current = list(best)
        current_set = set(current)
        
        def is_sum_free_set(S_set_local):
            for a in S_set_local:
                for b in S_set_local:
                    if a + b in S_set_local:
                        return False
            return True
        
        # Local search
        improved = True
        while improved:
            improved = False
            for i in range(len(current)):
                for j in range(len(S)):
                    if S[j] in current_set:
                        continue
                    # Try swapping current[i] with S[j]
                    new_set = current_set - {current[i]} | {S[j]}
                    if is_sum_free_set(new_set):
                        # Check if this is an improvement (it's the same size, so no improvement in size)
                        # But maybe it enables further improvements?
                        # For now, let's just accept if size is same or larger
                        # Actually, swapping keeps size same. Let's look for additions first.
                        pass
            
            # Try adding elements
            for x in S:
                if x in current_set:
                    continue
                # Check if adding x keeps sum-free
                can_add = True
                if x + x in current_set:
                    can_add = False
                else:
                    for a in current_set:
                        if a + x in current_set:
                            can_add = False
                            break
                if can_add:
                    current_set.add(x)
                    current.append(x)
                    improved = True
                    break
            
            if time.time() > start_time + time_limit:
                break
        
        return list(current_set)

    # Main logic
    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for size, seed in instances:
        S = instance(size, seed)
        
        # Time budget for this instance
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining <= 0:
            # Not enough time, use quick heuristic
            result = [x for x in S if x % 2 != 0]
            results.append(result)
            continue
        
        # Allocate time: 70% to MILP, 30% to heuristic
        milp_time = remaining * 0.7
        heuristic_time = remaining * 0.3
        
        # Try MILP first
        result = solve_with_milp(S, milp_time)
        
        if result is None or not is_sum_free(result):
            # Fallback to heuristic
            result = solve_with_heuristic(S, heuristic_time)
            if not is_sum_free(result):
                # Last resort: odds only
                result = [x for x in S if x % 2 != 0]
        else:
            # MILP succeeded, but try to improve with heuristic if we have time
            elapsed2 = time.time() - start_time
            remaining2 = DEADLINE - elapsed2
            if remaining2 > 5:  # At least 5 seconds left
                heuristic_result = solve_with_heuristic(S, min(remaining2, 10))
                if is_sum_free(heuristic_result) and len(heuristic_result) > len(result):
                    result = heuristic_result
            
        results.append(result)
        
        # Check deadline
        if time.time() > DEADLINE - 2:
            # Not enough time for remaining instances
            for s, sd in instances[instances.index((size, seed)) + 1:]:
                S_rem = instance(s, sd)
                results.append([x for x in S_rem if x % 2 != 0])
            break
        
    return results
