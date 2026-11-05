import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if len(S) < 2:
            return True
        Ss = set(S)
        for a in S:
            for b in S:
                if a + b in Ss:
                    return False
        return True

    def can_add_fast(x, S_set):
        for z in S_set:
            if (x + z) in S_set:
                return False
            if (x - z) in S_set:
                return False
        return True

    def make_greedy(arr, key):
        chosen = []
        chosen_set = set()
        for x in sorted(arr, key=key):
            if can_add_fast(x, chosen_set):
                chosen.append(x)
                chosen_set.add(x)
        # Add-back passes
        all_set = set(arr)
        changed = True
        passes = 0
        while changed and passes < 8:
            changed = False
            passes += 1
            excluded = [x for x in arr if x not in chosen_set]
            for x in excluded:
                if can_add_fast(x, chosen_set):
                    chosen.append(x)
                    chosen_set.add(x)
                    changed = True
        return chosen

    def ilp_max_sum_free(arr, initial_subset, time_budget):
        """Use scipy milp to find a large sum-free subset.
        Variables: x_i in {0,1} for each element in arr.
        Maximize sum x_i.
        Constraint: for each triple (i,j,k) where arr[i]+arr[j]==arr[k],
                    x_i + x_j + x_k <= 2.
        """
        n = len(arr)
        if n == 0:
            return []
        
        # Build conflict triples
        idx_map = {v: i for i, v in enumerate(arr)}
        triples = set()
        for i in range(n):
            for j in range(i, n):
                s = arr[i] + arr[j]
                if s in idx_map:
                    k = idx_map[s]
                    # Sort to canonical form
                    t = tuple(sorted((i, j, k)))
                    triples.add(t)
        
        num_triples = len(triples)
        
        # Objective: maximize sum x_i
        c = np.full(n, -1.0)  # minimize -sum x_i
        
        # Constraints: for each triple, x_i + x_j + x_k <= 2
        if num_triples == 0:
            return list(arr)
        
        A = np.zeros((num_triples, n))
        for r, (i, j, k) in enumerate(triples):
            A[r, i] = 1.0
            A[r, j] = 1.0
            A[r, k] = 1.0
        
        ub = np.full(num_triples, 2.0)
        lb = np.full(num_triples, -np.inf)
        
        constraints = LinearConstraint(A, lb, ub)
        integrality = np.ones(n)
        bounds = Bounds(0, 1)
        
        try:
            res = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                options={'time_limit': max(0.5, time_budget - 0.5),
                         'mip_rel_gap': 0.0}
            )
            if res.status in (0, 2, 3):  # optimal, or feasible found
                x = res.x
                if x is not None:
                    subset = [arr[i] for i in range(n) if x[i] > 0.5]
                    if is_sum_free(subset):
                        return subset
                    else:
                        # Fallback: filter to valid
                        valid = []
                        vs = set()
                        for v in subset:
                            if can_add_fast(v, vs):
                                valid.append(v)
                                vs.add(v)
                        return valid
        except Exception:
            pass
        
        return list(initial_subset)

    def local_search_add(arr, initial, time_budget):
        current = list(initial)
        current_set = set(current)
        arr_set = set(arr)
        not_in = list(arr_set - current_set)
        best = current
        best_len = len(current)
        
        rng = random.Random(999)
        iterations = 0
        
        def compute_conflicts(not_in_list, cur_set):
            cc = {}
            for x in not_in_list:
                cnt = 0
                for z in cur_set:
                    if (x + z) in cur_set or (x - z) in cur_set:
                        cnt += 1
                cc[x] = cnt
            return cc
        
        conflict_count = compute_conflicts(not_in, current_set)
        
        while time_budget() > 0.3 and iterations < 50000:
            iterations += 1
            if not not_in:
                break
            
            zero_conflict = [x for x in not_in if conflict_count.get(x, 0) == 0]
            
            if zero_conflict:
                x = min(zero_conflict, key=lambda v: abs(v))
            else:
                x = rng.choice(not_in)
            
            if can_add_fast(x, current_set):
                current.append(x)
                current_set.add(x)
                not_in.remove(x)
                if len(current) > best_len:
                    best = list(current)
                    best_len = len(current)
                for y in not_in:
                    if (x + y) in current_set or (x - y) in current_set or (y - x) in current_set:
                        conflict_count[y] = conflict_count.get(y, 0) + 1
        return best

    def local_search_swap(arr, initial, time_budget):
        current = list(initial)
        current_set = set(current)
        arr_set = set(arr)
        not_in = list(arr_set - current_set)
        best = current
        best_len = len(current)
        
        rng = random.Random(777)
        iterations = 0
        while time_budget() > 0.3 and iterations < 20000:
            iterations += 1
            if not not_in or not current:
                break
            x = rng.choice(not_in)
            if can_add_fast(x, current_set):
                current.append(x)
                current_set.add(x)
                not_in.remove(x)
                if len(current) > best_len:
                    best = list(current)
                    best_len = len(current)
                continue
            idx = rng.randrange(len(current))
            removed = current[idx]
            current.pop(idx)
            current_set.discard(removed)
            if can_add_fast(x, current_set):
                current.append(x)
                current_set.add(x)
                not_in.remove(x)
                if len(current) > best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                current.append(removed)
                current_set.add(removed)
        return best

    def solve_instance(arr):
        n = len(arr)
        best = []
        
        # 1. Structural candidates
        pos = [x for x in arr if x > 0]
        if pos:
            pos_sorted = sorted(pos, reverse=True)
            cand = pos_sorted[:len(pos_sorted)//2 + 1]
            if is_sum_free(cand):
                if len(cand) > len(best):
                    best = cand
            else:
                cand = make_greedy(pos, lambda x: -x)
                if len(cand) > len(best):
                    best = cand
        
        neg = [x for x in arr if x < 0]
        if neg:
            neg_sorted = sorted(neg)
            cand = neg_sorted[:len(neg_sorted)//2 + 1]
            if is_sum_free(cand):
                if len(cand) > len(best):
                    best = cand
            else:
                cand = make_greedy(neg, lambda x: x)
                if len(cand) > len(best):
                    best = cand
        
        odds = [x for x in arr if x % 2 != 0]
        if odds and len(odds) > len(best):
            best = odds
        
        # 2. Greedy with diverse orderings
        orderings = [
            lambda x: (abs(x), x),
            lambda x: x,
            lambda x: -x,
            lambda x: (abs(x), -x),
            lambda x: (x, abs(x)),
            lambda x: (-x, abs(x)),
        ]
        for key in orderings:
            cand = make_greedy(arr, key)
            if len(cand) > len(best):
                best = cand
        
        # 3. Local search improvement
        if time_left() > 5:
            improved = local_search_add(arr, best, time_left)
            if len(improved) > len(best):
                best = improved
        
        # 4. ILP refinement (the key improvement)
        if time_left() > 3:
            # Give ILP most of the remaining time
            ilp_budget = time_left() - 1.0
            ilp_result = ilp_max_sum_free(arr, best, ilp_budget)
            if len(ilp_result) > len(best):
                best = ilp_result
        
        # 5. Final local search polish after ILP
        if time_left() > 2:
            improved = local_search_add(arr, best, time_left)
            if len(improved) > len(best):
                best = improved
            improved = local_search_swap(arr, best, time_left)
            if len(improved) > len(best):
                best = improved
        
        # Final safety check
        if not is_sum_free(best):
            # Rebuild valid set
            valid = []
            vs = set()
            for v in sorted(best, key=abs):
                if can_add_fast(v, vs):
                    valid.append(v)
                    vs.add(v)
            best = valid
        
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        results.append(solve_instance(S))
    
    return results
