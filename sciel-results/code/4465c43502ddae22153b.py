import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_instance(values, budget):
        n = len(values)
        if n <= 2:
            return list(values)

        v_set = set(values)
        val_to_idx = {v: i for i, v in enumerate(values)}
        rng = random.Random(42)

        # Precompute: for each pair (i, j), whether values[i] + values[j] is in values
        # This creates a "sum conflict" - if i and j are both in S, and values[i]+values[j] is also in S, violation
        # Actually, we need: i, j, k are in S and values[i] + values[j] = values[k]
        # So conflict is a triple, not a pair. But for local search, we can check:
        # Adding k to S is safe iff for all a in S: a+k not in S, and 2k not in S.
        # Equivalently: for all a in S, (a+k) not in val_to_idx OR (a+k) not in S.
        
        # Precompute for each value k, the set of values a such that a+k is in values
        # And the set of values a such that a-k is in values
        # And check 2k
        
        # For efficiency, precompute:
        # plus_partners[k_idx] = set of a_idx such that values[a_idx] + values[k_idx] in values
        # minus_partners[k_idx] = set of a_idx such that values[a_idx] - values[k_idx] in values
        # double_check[k_idx] = True if 2*values[k_idx] in values
        
        plus_partners = [set() for _ in range(n)]
        minus_partners = [set() for _ in range(n)]
        double_in_set = [False] * n
        
        for i in range(n):
            vi = values[i]
            # 2*vi in values?
            if 2 * vi in val_to_idx:
                double_in_set[i] = True
            # For each j, check if vi + values[j] in values
            # And if values[j] + vi in values (same)
            # And if vi - values[j] in values
            # And if values[j] - vi in values
            
        # Better: for each pair (i, j), compute sum = values[i] + values[j]
        # If sum in val_to_idx, let k = val_to_idx[sum]
        # Then (i, j, k) is a sum triple
        # For local search, when considering adding k to S, we need to check
        # if there exist a, b in S such that a + b = k (i.e., k is the sum)
        # or a + k = b or b + k = a (i.e., k is one of the addends)
        
        # Precompute for each k:
        # summands_of[k] = set of (a, b) such that values[a] + values[b] = values[k]
        # addend_conflicts[k] = set of a such that values[a] + values[k] in values OR values[k] + values[a] in values
        #                      (which is the same)
        
        summands_of = [set() for _ in range(n)]  # pairs (a_idx, b_idx) where val[a]+val[b]=val[k]
        addend_conflicts = [set() for _ in range(n)]  # a_idx such that val[a]+val[k] in values
        
        for i in range(n):
            for j in range(i, n):
                s = values[i] + values[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    if k != i and k != j:
                        summands_of[k].add((i, j))
                        # i and j are addends that sum to k
                        addend_conflicts[k].add(i)
                        addend_conflicts[k].add(j)
                        # Also, i conflicts with k as addend: val[i] + val[k] might be in values
                        # But that's a different triple. Let's handle all cases.
        
        # Actually, let's think differently. For a set S to be sum-free:
        # For all a, b in S: a + b not in S.
        # When we add a new element x to S, we need:
        # 1. For all a in S: a + x not in S
        # 2. For all a in S: x + a not in S  (same as 1)
        # 3. x + x not in S
        # 
        # Condition 1 & 2: for all a in S, a + x not in S.
        # This means: for all a in S, if a + x is in values, then a + x not in S.
        # Equivalently: (S ∩ (x - S)) = empty, where x - S = {x - a : a in S}
        # And x + x not in S.
        
        # Precompute for each x (index k):
        # partners_plus[k] = set of a_idx such that values[a_idx] + values[k] in values
        #   (i.e., a + x is some value in the instance)
        # partners_minus[k] = set of a_idx such that values[a_idx] - values[k] in values
        #   (i.e., a - x is some value, so a = x + (a-x), meaning x + (a-x) = a, so if a-x in S and a in S, violation)
        #   Actually: we need a + x not in S. If a + x = c for some c in values, then c not in S.
        #   So: for each a in S, if a + x in values, then that value's index must not be in S.
        
        # Let me precompute:
        # for each k, conflict_addends[k] = set of a_idx such that values[a_idx] + values[k] in values
        #   When adding k, for each a in S ∩ conflict_addends[k], we need values[a]+values[k] not in S.
        #   But values[a]+values[k] is a specific value. Let c = val_to_idx[values[a]+values[k]].
        #   Then we need c not in S.
        # So the condition is: for all a in S, if a in conflict_addends[k], then c(a,k) not in S.
        # This is complex. Let's just do the O(|S|) check but with early termination.
        
        # For the local search, the key insight from the inspiration is to use
        # a conflict graph where nodes are elements and edges represent incompatibility.
        # But sum-free is a 3-uniform constraint, not pairwise.
        
        # Alternative approach: Use the fact that for random sets, the modular
        # structure gives a good lower bound. Let's focus on:
        # 1. Strong modular initialization
        # 2. Greedy add with multiple random orderings
        # 3. Local search with remove-and-readd moves
        
        # Modular initialization
        best_solution = []
        
        for m in [2, 3, 4, 5, 6, 7, 8, 10, 12]:
            counts = [0] * m
            for v in values:
                counts[v % m] += 1
            
            best_residue_set = set()
            max_residue_size = 0
            
            # For small m, brute force all residue subsets
            if m <= 12:
                for mask in range(1 << m):
                    residues = set()
                    valid = True
                    for r in range(m):
                        if mask & (1 << r):
                            residues.add(r)
                    if not residues:
                        continue
                    # Check sum-free in Z_m
                    for a in residues:
                        for b in residues:
                            if (a + b) % m in residues:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        cnt = sum(counts[r] for r in residues)
                        if cnt > max_residue_size:
                            max_residue_size = cnt
                            best_residue_set = residues
            
            if best_residue_set:
                candidate = [v for v in values if v % m in best_residue_set]
                if len(candidate) > len(best_solution):
                    best_solution = candidate
        
        # Also try specific patterns
        odds = [v for v in values if v % 2 != 0]
        if len(odds) > len(best_solution):
            best_solution = odds
        
        # Start with best modular solution
        S = set(best_solution)
        
        def can_add(val, current_set):
            """Check if val can be added to current_set maintaining sum-free."""
            # Check: for all a in current_set, a + val not in current_set
            # And 2*val not in current_set
            for a in current_set:
                if a + val in current_set:
                    return False
            if 2 * val in current_set:
                return False
            return True
        
        # Greedy add with multiple random orderings
        def greedy_add(start_set, values_list, rng_instance, max_rounds=5):
            S_local = set(start_set)
            remaining = [v for v in values_list if v not in S_local]
            
            for _ in range(max_rounds):
                rng_instance.shuffle(remaining)
                added_any = False
                new_remaining = []
                for v in remaining:
                    if can_add(v, S_local):
                        S_local.add(v)
                        added_any = True
                    else:
                        new_remaining.append(v)
                remaining = new_remaining
                if not added_any:
                    break
            return S_local
        
        # Run greedy add with different random seeds
        best_S = set(S)
        best_size = len(S)
        
        for trial in range(10):
            trial_rng = random.Random(1000 + trial)
            S_trial = greedy_add(S, values, trial_rng, max_rounds=3)
            if len(S_trial) > best_size:
                best_size = len(S_trial)
                best_S = S_trial
        
        # Local search: remove one, try to add multiple
        ls_deadline = start_time + budget
        current_S = set(best_S)
        
        for ls_round in range(200):
            if time.time() > ls_deadline:
                break
            
            # Try removing a random element and re-adding greedily
            if not current_S:
                break
            
            # Pick a few random elements to remove
            remove_count = min(3, len(current_S) // 10 + 1)
            to_remove = random.sample(list(current_S), remove_count)
            
            S_trial = set(current_S)
            for v in to_remove:
                S_trial.discard(v)
            
            # Greedy add from trial
            remaining = [v for v in values if v not in S_trial]
            random.shuffle(remaining)
            for v in remaining:
                if can_add(v, S_trial):
                    S_trial.add(v)
            
            if len(S_trial) > len(current_S):
                current_S = S_trial
                if len(current_S) > best_size:
                    best_size = len(current_S)
                    best_S = set(current_S)
        
        # Final greedy polish
        remaining = [v for v in values if v not in best_S]
        random.shuffle(remaining)
        for v in remaining:
            if can_add(v, best_S):
                best_S.add(v)
        
        result = list(best_S)
        if not is_sum_free(result):
            result = [x for x in values if x % 2 != 0]
        return result

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elapsed = time.time() - start_time
        remaining_time = DEADLINE - elapsed
        if remaining_time < 15:
            vals = instance(size, seed)
            res = [x for x in vals if x % 2 != 0]
            results.append(res)
            continue

        # Allocate budget based on size
        if size == 60:
            budget = min(20, remaining_time - 30)
        elif size == 120:
            budget = min(40, remaining_time - 30)
        else:
            budget = min(70, remaining_time - 15)
        
        budget = max(budget, 5)
        
        vals = instance(size, seed)
        res = solve_instance(vals, budget)
        
        if not is_sum_free(res):
            res = [x for x in vals if x % 2 != 0]
        
        results.append(res)

    return results
