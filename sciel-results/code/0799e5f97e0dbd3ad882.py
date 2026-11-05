"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def solve_sum_free(vals, seed=42, time_limit=30.0):
        n = len(vals)
        if n == 0:
            return []
        
        val_to_idx = {v: i for i, v in enumerate(vals)}
        constraints = []
        for i in range(n):
            for j in range(i, n):
                s = vals[i] + vals[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    constraints.append((i, j, k))
                    
        var_to_cons = [[] for _ in range(n)]
        for c_idx, (i, j, k) in enumerate(constraints):
            var_to_cons[i].append(c_idx)
            if j != i:
                var_to_cons[j].append(c_idx)
            if k != i and k != j:
                var_to_cons[k].append(c_idx)
            # Note: if i==j, var_to_cons[i] gets one entry for this constraint?
            # No, the constraint is on indices i, j, k.
            # If i==j, the variables involved are i and k.
            # My loop above:
            # var_to_cons[i].append(c_idx)
            # if j != i: ... (False)
            # if k != i and k != j: ... (True if k!=i)
            # So var_to_cons[i] gets 1, var_to_cons[k] gets 1. Correct.
            
        sol = [1 if v % 2 != 0 else 0 for v in vals]
        
        viol_count = [0] * n
        total_violations = 0
        for c_idx, (i, j, k) in enumerate(constraints):
            if sol[i] and sol[j] and sol[k]:
                total_violations += 1
                viol_count[i] += 1
                viol_count[j] += 1
                viol_count[k] += 1
        
        best_sol = sol[:]
        best_score = sum(sol)
        
        rng = random.Random(seed)
        start_time = time.time()
        
        max_no_improve = 2000
        no_improve_count = 0
        temperature = 0.1
        
        for _ in range(500000):
            if time.time() - start_time > time_limit:
                break
                
            if total_violations == 0:
                # Try to add
                added = False
                for _ in range(10): # Try a few random 0s
                    zeros = [i for i in range(n) if sol[i] == 0]
                    if not zeros: break
                    x = rng.choice(zeros)
                    can_add = True
                    for c_idx in var_to_cons[x]:
                        i, j, k = constraints[c_idx]
                        if (x == i and sol[j] and sol[k]) or \
                           (x == j and sol[i] and sol[k]) or \
                           (x == k and sol[i] and sol[j]):
                            can_add = False
                            break
                    if can_add:
                        sol[x] = 1
                        added = True
                        break
                if not added:
                    # Flip a 1 to 0
                    ones = [i for i in range(n) if sol[i]]
                    if not ones: break
                    x = rng.choice(ones)
                    sol[x] = 0
                continue
            
            # Violations exist
            if rng.random() < temperature:
                x = rng.randint(0, n-1)
            else:
                max_v = max(viol_count)
                if max_v == 0:
                    x = rng.randint(0, n-1)
                else:
                    candidates = [i for i in range(n) if viol_count[i] == max_v]
                    x = rng.choice(candidates)
            
            old_val = sol[x]
            new_val = 1 - old_val
            sol[x] = new_val
            
            for c_idx in var_to_cons[x]:
                i, j, k = constraints[c_idx]
                if old_val == 0 and new_val == 1:
                    if (x == i and sol[j] and sol[k]) or \
                       (x == j and sol[i] and sol[k]) or \
                       (x == k and sol[i] and sol[j]):
                        total_violations += 1
                        viol_count[i] += 1
                        viol_count[j] += 1
                        viol_count[k] += 1
                elif old_val == 1 and new_val == 0:
                    if (x == i and sol[j] and sol[k]) or \
                       (x == j and sol[i] and sol[k]) or \
                       (x == k and sol[i] and sol[j]):
                        total_violations -= 1
                        viol_count[i] -= 1
                        viol_count[j] -= 1
                        viol_count[k] -= 1
            
            cur_score = sum(sol)
            if total_violations == 0 and cur_score > best_score:
                best_sol = sol[:]
                best_score = cur_score
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    sol = best_sol[:]
                    for _ in range(n // 10):
                        x = rng.randint(0, n-1)
                        sol[x] = 1 - sol[x]
                    viol_count = [0] * n
                    total_violations = 0
                    for c_idx, (i, j, k) in enumerate(constraints):
                        if sol[i] and sol[j] and sol[k]:
                            total_violations += 1
                            viol_count[i] += 1
                            viol_count[j] += 1
                            viol_count[k] += 1
                    no_improve_count = 0

        return [vals[i] for i in range(n) if best_sol[i]]

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        subset = solve_sum_free(vals, seed=seed)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
