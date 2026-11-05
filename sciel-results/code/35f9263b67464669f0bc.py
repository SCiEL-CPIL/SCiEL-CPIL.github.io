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

    def solve_walhsat(S, time_limit):
        """WalkSAT-style local search for sum-free subset."""
        n = len(S)
        S_set = set(S)
        
        # Build conflict clauses: for each a, b where a+b is in S,
        # we have constraint x_a + x_b + x_{a+b} <= 2
        # This is a 3-CNF clause: NOT(x_a AND x_b AND x_{a+b})
        clauses = []
        seen = set()
        for i, a in enumerate(S):
            for j, b in enumerate(S):
                c = a + b
                if c in S_set:
                    k = S.index(c) if c in S_set else -1
                    if k >= 0:
                        # Normalize clause (sort indices)
                        clause = tuple(sorted([i, j, k]))
                        if clause not in seen:
                            seen.add(clause)
                            clauses.append(clause)
        
        if not clauses:
            return list(S)
        
        # Initialize with greedy
        def greedy_init():
            result = set()
            for x in S:
                if can_add(x, result):
                    result.add(x)
            return result
        
        best_set = greedy_init()
        best_size = len(best_set)
        
        # Convert set to boolean array
        def set_to_bool(S_set):
            return np.array([1 if x in S_set else 0 for x in S])
        
        def bool_to_set(x):
            return [S[i] for i in range(n) if x[i] > 0.5]
        
        # Evaluate number of violated clauses
        def violated_clauses(x):
            count = 0
            for clause in clauses:
                if all(x[i] > 0.5 for i in clause):
                    count += 1
            return count
        
        # Weighted clause restart
        def walhsat_search(x, time_limit, rng):
            start_time = time.time()
            x = x.copy()
            current_violated = violated_clauses(x)
            best_x = x.copy()
            best_violated = current_violated
            
            # Track clause weights for restarts
            clause_weights = np.ones(len(clauses))
            
            while time.time() - start_time < time_limit:
                if current_violated == 0:
                    # Found valid solution
                    sol = bool_to_set(x)
                    if len(sol) > len(best_x_to_set(best_x)):
                        best_x = x.copy()
                        best_violated = 0
                    break
                
                # Select a random violated clause
                violated_idx = [i for i, clause in enumerate(clauses) if all(x[j] > 0.5 for j in clause)]
                if not violated_idx:
                    break
                
                # Weighted random selection
                weights = clause_weights[violated_idx]
                total_w = weights.sum()
                if total_w == 0:
                    weights = np.ones(len(violated_idx))
                    total_w = len(violated_idx)
                
                probs = weights / total_w
                selected = rng.choice(len(violated_idx), p=probs)
                clause = clauses[violated_idx[selected]]
                
                # Flip a variable in the clause with probability
                # Higher probability for variables that appear in more violated clauses
                var_counts = np.zeros(n)
                for idx in violated_idx:
                    for var in clauses[idx]:
                        var_counts[var] += 1
                
                # Choose which variable to flip
                flip_probs = var_counts[[i for i in clause]]
                if flip_probs.sum() == 0:
                    flip_probs = np.ones(3)
                flip_probs = flip_probs / flip_probs.sum()
                
                flip_choice = rng.choice(3, p=flip_probs)
                var_to_flip = clause[flip_choice]
                
                # Try flipping
                x[var_to_flip] = 1 - x[var_to_flip]
                new_violated = violated_clauses(x)
                
                if new_violated < current_violated:
                    current_violated = new_violated
                    if current_violated < best_violated:
                        best_violated = current_violated
                        best_x = x.copy()
                    # Decrease weight of this clause
                    clause_weights[violated_idx[selected]] *= 0.95
                else:
                    # Revert
                    x[var_to_flip] = 1 - x[var_to_flip]
                    # Increase weight of this clause
                    clause_weights[violated_idx[selected]] *= 1.05
                
                # Periodic restart with perturbation
                if rng.random() < 0.01:
                    # Perturb current best
                    x = best_x.copy()
                    for _ in range(rng.randint(1, 3)):
                        if rng.random() < 0.5:
                            idx = rng.randint(0, n-1)
                            x[idx] = 1 - x[idx]
                    current_violated = violated_clauses(x)
                    clause_weights = np.ones(len(clauses))
            
            return best_x, best_violated
        
        def best_x_to_set(x):
            return bool_to_set(x)
        
        # Run WalkSAT with multiple restarts
        start_time = time.time()
        rng = random.Random(42)
        
        while time.time() - start_time < time_limit:
            # Start from greedy or random
            if rng.random() < 0.5:
                initial_set = greedy_init()
            else:
                # Random initialization
                x_init = np.zeros(n)
                for i in range(n):
                    x_init[i] = 1 if rng.random() < 0.5 else 0
                initial_set = bool_to_set(x_init)
            
            x_init = set_to_bool(set(initial_set))
            x_result, viol = walhsat_search(x_init, min(time_limit - (time.time() - start_time), 2.0), rng)
            
            sol = bool_to_set(x_result)
            if len(sol) > best_size:
                best_size = len(sol)
                best_set = set(sol)
            
            if viol == 0:
                break
        
        return list(best_set)

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
        # WalkSAT is a good complement for local search.
        
        time_for_ilp = remaining_time * 0.4
        time_for_walhsat = remaining_time * 0.3
        time_for_heuristic = remaining_time * 0.3
        
        best_sol = None
        
        # Try ILP
        if time_for_ilp > 1:
            ilp_sol = solve_ilp(S, time_for_ilp)
            if ilp_sol is not None:
                best_sol = ilp_sol
        
        # Try WalkSAT
        elapsed2 = time.time() - start
        remaining_time2 = total_time - elapsed2 - 5
        if time_for_walhsat > 1 and remaining_time2 > 1:
            walhsat_sol = solve_walhsat(S, time_for_walhsat)
            if best_sol is None or len(walhsat_sol) > len(best_sol):
                best_sol = walhsat_sol
        
        # If we have time left, try Heuristic
        elapsed3 = time.time() - start
        remaining_time3 = total_time - elapsed3 - 5
        
        if best_sol is None or remaining_time3 > 1:
            heur_sol = solve_heuristic(S, remaining_time3)
            if best_sol is None or len(heur_sol) > len(best_sol):
                best_sol = heur_sol
        
        if best_sol is None:
            # Absolute fallback
            best_sol = [x for x in S if x % 2 != 0]
            
        results.append(best_sol)
    
    return results
