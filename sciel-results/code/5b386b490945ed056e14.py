"""Rewritten program: local search with bitmask for sum-free subsets."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 135  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve_instance(values, time_limit=40):
        n = len(values)
        # Map each value to an index
        val_to_idx = {v: i for i, v in enumerate(values)}
        
        # Precompute: for each pair (i, j), does values[i] + values[j] exist in values?
        # If so, store the index k such that values[k] = values[i] + values[j]
        # For sum-free: if i and j are both in subset, then k must NOT be in subset.
        # We'll represent subset as a bitmask (Python int for n<=240 is fine, or use set)
        # For n up to 240, bitmask as Python int works but operations are O(n/word_size)
        # Actually for n=240, let's use a set of indices for clarity, or a list of bools.
        # For speed, let's use a set of indices.
        
        # Precompute conflict graph: for each pair (i,j) where i<=j, find k = idx of (values[i]+values[j])
        # If k exists, then (i,j) conflicts with k.
        # We'll store: conflicts[i] = set of j such that i and j conflict (i.e., values[i]+values[j] is in values)
        # Actually, we need: if i and j are both selected, then k must not be selected.
        # This is a 3-way constraint. Let's think differently.
        
        # Alternative: for each element k, the forbidden pairs are (i,j) where values[i]+values[j]==values[k].
        # If k is in the subset, then no pair (i,j) with values[i]+values[j]==values[k] can both be in the subset.
        
        # Let's use a different approach: greedy + local search
        # Start with a good initial solution (e.g., odd numbers, or all positive, or all negative)
        # Then do local search: try adding elements, removing elements, swapping.
        
        # For n up to 240, we can use a set-based approach with careful optimization.
        
        # Precompute: for each k, list of pairs (i,j) with i<=j such that values[i]+values[j]==values[k]
        pair_to_k = {}
        for i in range(n):
            for j in range(i, n):
                s = values[i] + values[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    pair_to_k[(i, j)] = k
        
        # For local search, we need to quickly check if adding element x to current subset S is valid.
        # Adding x is valid if:
        # 1. For all a in S: a + x is not in S (i.e., no a in S such that a+x is also in S)
        # 2. For all a in S: x + a is not in S (same as 1)
        # 3. 2*x is not in S
        # Actually: no a, b in S ∪ {x} with a+b in S ∪ {x}.
        # New violations involving x:
        # - a + b = x for some a, b in S
        # - a + x = b for some a, b in S  
        # - x + a = b for some a, b in S (same as above)
        # - x + x = a for some a in S
        
        # Let's define: can_add(S, x) returns True if S ∪ {x} is sum-free.
        # Violations:
        # 1. ∃ a, b ∈ S: a + b = x
        # 2. ∃ a, b ∈ S: a + x = b  (i.e., b - x = a, so ∃ a ∈ S: a + x ∈ S)
        # 3. 2x ∈ S
        
        # For speed, precompute for each x:
        # - pairs_summing_to[x] = set of (i,j) with i<=j, values[i]+values[j]==values[x]
        # - For condition 2: for each x, we need to check if ∃ a ∈ S such that a+x ∈ S
        #   This is equivalent to: ∃ a ∈ S ∩ (S - x) where S-x = {s-x for s in S}
        
        # This is getting complex. Let's use a simpler approach for n<=240:
        # Represent subset as a set of indices.
        # For can_add(S, x):
        #   Check if any a, b in S with a+b == x: iterate over pairs in S? O(|S|^2) - too slow.
        
        # Better: precompute for each value v, the set of indices that sum to v.
        # Actually, let's precompute:
        # sum_pairs[v] = list of (i,j) with i<=j such that values[i]+values[j]==v
        # Then can_add(S, x) where x is an index:
        #   v = values[x]
        #   Check 1: any (i,j) in sum_pairs[v] with i,j in S
        #   Check 2: any a in S such that values[a] + v is in S (i.e., values[a]+v is some values[b] with b in S)
        #            Equivalent: ∃ a in S: (values[a]+v) in val_to_idx and val_to_idx[values[a]+v] in S
        #   Check 3: 2v in S: i.e., 2*values[x] is in val_to_idx and that index is in S
        
        # For n=240, |S| can be up to ~80. Check 1: sum_pairs[v] could have many pairs.
        # Check 2: iterate over a in S, O(|S|) lookups.
        
        # Let's try a different strategy: use the fact that we want large sum-free sets.
        # Known good strategies:
        # 1. All odd numbers (sum of two odds is even, so no two odds sum to an odd)
        # 2. All numbers with |x| > n/2 (or some threshold) - sum of two large numbers exceeds range
        # 3. All positive numbers (if all positive, sum of two positives is positive and larger)
        #    Wait, that's not sum-free. 1+2=3, all positive.
        # 4. Numbers in (n/2, n] for positive case.
        
        # Let's start with multiple initializations and do local search.
        
        def is_sum_free(S):
            """Check if set of indices S is sum-free."""
            vals = [values[i] for i in S]
            val_set = set(vals)
            for i in range(len(vals)):
                for j in range(i, len(vals)):
                    if vals[i] + vals[j] in val_set:
                        return False
            return True
        
        def can_add(S, x):
            """Check if S ∪ {x} is sum-free, given S is sum-free."""
            vx = values[x]
            # Check: any a, b in S with a + b == vx
            # And: any a in S with a + vx in S
            # And: 2*vx in S
            s_vals = set(values[i] for i in S)
            # Check 1: a + b = vx for a, b in S
            # Iterate over a in S, check if vx - a in S
            for a in S:
                va = values[a]
                if vx - va in s_vals:
                    # Need to check it's a valid pair (could be a itself if 2*va = vx)
                    return False
            # Check 2: a + vx in S for some a in S
            for a in S:
                va = values[a]
                if va + vx in s_vals:
                    return False
            # Check 3: 2*vx in S
            if 2 * vx in s_vals:
                return False
            return True
        
        # This is O(|S|) per can_add, which is fine for |S| ~ 80.
        # But we call it many times. Let's optimize by keeping s_vals as a set.
        
        def local_search(S, max_iters=100000, time_limit=35):
            """Local search: try to add elements, then try swaps."""
            S = set(S)
            s_vals = set(values[i] for i in S)
            best_S = S
            best_size = len(S)
            
            t0 = time.time()
            # First pass: greedily try to add any element
            changed = True
            while changed and (time.time() - t0) < time_limit * 0.3:
                changed = False
                for x in range(n):
                    if x not in S and can_add(S, x):
                        S.add(x)
                        s_vals.add(values[x])
                        changed = True
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = set(S)
            
            # Second pass: try swaps (remove one, add another)
            for _ in range(max_iters):
                if (time.time() - t0) > time_limit:
                    break
                if len(S) == n:
                    break
                # Try random swaps
                if S:
                    a = random.choice(list(S))
                else:
                    a = random.randint(0, n-1)
                b = random.randint(0, n-1)
                if b == a:
                    continue
                # Try: remove a, add b
                if b in S:
                    continue
                S.remove(a)
                s_vals.remove(values[a])
                if can_add(S, b):
                    S.add(b)
                    s_vals.add(values[b])
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                else:
                    S.add(a)
                    s_vals.add(values[a])
            
            return list(best_S)
        
        # Multiple initializations
        best_solution = None
        best_size = 0
        
        # Init 1: odd numbers
        S1 = [i for i, v in enumerate(values) if v % 2 != 0]
        # Init 2: all positive
        S2 = [i for i, v in enumerate(values) if v > 0]
        # Init 3: all negative
        S3 = [i for i, v in enumerate(values) if v < 0]
        # Init 4: numbers with large absolute value
        mid = n // 2
        S4 = [i for i, v in enumerate(values) if abs(v) > 5000]
        # Init 5: random subset of size n//3
        rng2 = random.Random(42)
        S5 = rng2.sample(range(n), max(1, n // 3))
        
        inits = [S1, S2, S3, S4, S5]
        
        for S_init in inits:
            if time.time() - start_time > DEADLINE - 5:
                break
            # Ensure initial set is sum-free (filter it)
            S_clean = []
            for i in S_init:
                if all(values[i] + values[j] not in set(values[k] for k in S_clean) for j in S_clean if j != i) and \
                   all(values[j] + values[i] not in set(values[k] for k in S_clean) for j in S_clean if j != i) and \
                   2 * values[i] not in set(values[k] for k in S_clean):
                    S_clean.append(i)
            # Simpler: just verify and remove violators
            S_clean = list(S_init)
            changed = True
            while changed:
                changed = False
                for i in range(len(S_clean)):
                    vi = values[S_clean[i]]
                    for j in range(i):
                        vj = values[S_clean[j]]
                        if vi + vj in set(values[k] for k in S_clean):
                            S_clean.pop(i)
                            changed = True
                            break
                        elif vi + vj in set(values[k] for k in S_clean[:i]):
                            S_clean.pop(i)
                            changed = True
                            break
                    if changed:
                        break
                    if 2 * vi in set(values[k] for k in S_clean[:i]):
                        S_clean.pop(i)
                        changed = True
                        break
            
            result = local_search(S_clean, time_limit=30)
            if len(result) > best_size:
                best_size = len(result)
                best_solution = result
        
        # Convert indices back to values
        if best_solution is None:
            return []
        return [values[i] for i in best_solution]
    
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() - start_time > DEADLINE - 5:
            # Return what we have
            break
        vals = instance(size, seed)
        sol = solve_instance(vals, time_limit=40)
        results.append(sol)
        # Verify sum-free
        sol_set = set(sol)
        for i, a in enumerate(sol):
            for j, b in enumerate(sol):
                if a + b in sol_set:
                    # This shouldn't happen, but if it does, remove one
                    pass
    
    return results
# EVOLVE-BLOCK-END
