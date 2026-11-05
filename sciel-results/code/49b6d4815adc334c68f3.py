import time
import random
import numpy as np
import scipy.optimize as opt

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S):
        S_set = set(S)
        for a in S_set:
            for b in S_set:
                if a + b in S_set:
                    return False
        return True

    def can_add(x, S_set):
        for a in S_set:
            if (x - a) in S_set:
                return False
        for a in S_set:
            if (a + x) in S_set:
                return False
        if (2 * x) in S_set:
            return False
        return True

    def solve_heuristic(S, time_limit):
        n = len(S)
        S_set = set(S)
        
        def greedy_sorted(order):
            result = set()
            for x in order:
                if can_add(x, result):
                    result.add(x)
            return result

        # Multiple initializations
        inits = []
        inits.append(greedy_sorted(S))
        inits.append(greedy_sorted(list(reversed(S))))
        odds = [x for x in S if x % 2 != 0]
        inits.append(set(odds))
        
        # Suffix approach
        suffix = S
        for k in range(1, n+1):
            if 2 * S[n-k] > S[n-1]:
                suffix = S[n-k:]
            else:
                break
        inits.append(set(suffix))
        
        for _ in range(10):
            shuffled = list(S)
            random.shuffle(shuffled)
            inits.append(greedy_sorted(shuffled))
        
        best = max(inits, key=len)
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            improved = False
            best_copy = set(best)
            
            # Try removing each element
            for x in list(best_copy):
                trial = set(best_copy)
                trial.discard(x)
                
                # Try to add elements from S \ trial
                for y in S:
                    if y in trial:
                        continue
                    if can_add(y, trial):
                        trial.add(y)
                
                if len(trial) > len(best):
                    best = trial
                    improved = True
                    break
            
            if not improved:
                # Try random restarts with perturbation
                found = False
                for _ in range(20):
                    if time.time() - start_time > time_limit:
                        break
                    perturbed = set(best)
                    for _ in range(random.randint(1, 3)):
                        if perturbed:
                            perturbed.discard(random.choice(list(perturbed)))
                    remaining = [x for x in S if x not in perturbed]
                    random.shuffle(remaining)
                    for y in remaining:
                        if can_add(y, perturbed):
                            perturbed.add(y)
                    if len(perturbed) > len(best):
                        best = perturbed
                        found = True
                        break
                if found:
                    continue
                else:
                    # Try pair removals if still stuck and time permits
                    elements = list(best)
                    if len(elements) > 10:
                        for i in range(len(elements)):
                            if time.time() - start_time > time_limit:
                                break
                            for j in range(i+1, len(elements)):
                                trial = set(best)
                                trial.discard(elements[i])
                                trial.discard(elements[j])
                                for y in S:
                                    if y in trial:
                                        continue
                                    if can_add(y, trial):
                                        trial.add(y)
                                if len(trial) > len(best):
                                    best = trial
                                    improved = True
                                    break
                            if improved:
                                break
                    if not improved:
                        break
        
        return list(best)

    def solve_ilp(S, time_limit):
        n = len(S)
        S_set = set(S)
        index = {v: i for i, v in enumerate(S)}
        
        # Find all constraints x_a + x_b + x_c <= 2
        constraints = []
        # To avoid duplicates, we can just generate all i, j
        # Since a+b=c and b+a=c are the same, we can restrict i<=j?
        # No, if a != b, (a,b,c) and (b,a,c) are distinct variable indices but same constraint.
        # The constraint is x_i + x_j + x_k <= 2.
        # It is symmetric in i, j. So we can just iterate i, j and add the constraint.
        # We should avoid adding the exact same row twice if possible, but it's not critical for correctness, just speed.
        
        for i, a in enumerate(S):
            for j, b in enumerate(S):
                c = a + b
                if c in S_set:
                    k = index[c]
                    # Constraint: x_i + x_j + x_k <= 2
                    # We can store as (i, j, k)
                    constraints.append((i, j, k))
        
        if not constraints:
            return list(S)
        
        # Build matrix A and vector b
        # A is sparse, use scipy.sparse
        from scipy.sparse import lil_matrix
        m = len(constraints)
        A = lil_matrix((m, n))
        b = np.ones(m) * 2
        
        for row, (i, j, k) in enumerate(constraints):
            A[row, i] = 1
            A[row, j] = 1
            A[row, k] = 1
        
        # Objective: maximize sum(x) -> minimize -sum(x)
        c = -np.ones(n)
        
        # Integrality: 1 for all
        integrality = np.ones(n)
        
        # Bounds: (0, 1)
        bounds = [(0, 1)] * n
        
        # Solve
        try:
            # milp requires scipy >= 1.7
            res = opt.milp(c, constraints=opt.LinearConstraint(A, lb=0, ub=b), 
                           integrality=integrality, bounds=bounds, 
                           options={'time_limit': time_limit, 'mip_rel_gap': 0.001})
            
            if res.status in [0, 1, 2, 3, 4, 5, 6]: # Optimal or feasible found
                if res.x is not None:
                    sol = []
                    for i, val in enumerate(res.x):
                        if val > 0.5:
                            sol.append(S[i])
                    # Verify
                    if is_sum_free(sol):
                        return sol
            # If no solution or invalid, fall through
        except Exception:
            pass
            
        return None

    instances = [
        instance(60, 801),
        instance(120, 802),
        instance(240, 803)
    ]
    
    results = []
    total_time = 115  # Safety margin for 120s limit
    start = time.time()
    
    for i, S in enumerate(instances):
        elapsed = time.time() - start
        remaining_time = total_time - elapsed - 5
        
        if remaining_time <= 0:
            # Fallback
            results.append([x for x in S if x % 2 != 0])
            continue
        
        # Allocate time
        # ILP is powerful but can be slow. Heuristic is fast.
        # Try ILP first with a generous time limit, fallback to heuristic if it fails or is too slow.
        # Actually, we can run heuristic first to get a good incumbent?
        # scipy.milp doesn't take initial solution.
        # Let's give ILP most of the time, and heuristic the rest if ILP fails.
        
        time_for_ilp = remaining_time * 0.7
        time_for_heuristic = remaining_time * 0.3
        
        best_sol = None
        
        # Try ILP
        if time_for_ilp > 1:
            ilp_sol = solve_ilp(S, time_for_ilp)
            if ilp_sol is not None:
                best_sol = ilp_sol
        
        # If ILP failed or we have time left, try Heuristic
        elapsed2 = time.time() - start
        remaining_time2 = total_time - elapsed2 - 5
        
        if best_sol is None or remaining_time2 > 1:
            heur_sol = solve_heuristic(S, remaining_time2)
            if best_sol is None or len(heur_sol) > len(best_sol):
                best_sol = heur_sol
        
        if best_sol is None:
            # Absolute fallback
            best_sol = [x for x in S if x % 2 != 0]
            
        results.append(best_sol)
    
    return results
