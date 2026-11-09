"""Rewritten program: conflict-graph based sum-free subset search with local search."""
import time
import random

def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def sum_free_check(S):
        """Return True if S is sum-free (no a+b=c with a,b,c in S, a=b allowed)."""
        s = set(S)
        for a in s:
            for b in s:
                if a + b in s:
                    return False
        return True

    def build_sum_free(arr):
        """Build a large sum-free subset using greedy + local search with swaps."""
        n = len(arr)
        if n == 0:
            return []
        
        elem_to_idx = {v: i for i, v in enumerate(arr)}
        rng = random.Random(12345)
        
        # Precompute conflict graph: conflicts[i] = set of j that conflict with i
        # i and j conflict if there exists k in arr such that arr[i]+arr[j]=arr[k] or arr[i]+arr[k]=arr[j] or arr[j]+arr[k]=arr[i]
        # Actually, i and j conflict if adding both to a set creates a sum-free violation
        # i and j conflict if:
        #   - arr[i] + arr[j] is in arr (and the index of that sum is not i or j)
        #   - arr[i] - arr[j] is in arr (and the index is not i or j)  
        #   - arr[j] - arr[i] is in arr (and the index is not i or j)
        # Wait, that's not quite right. Let me think again.
        # A set S is sum-free if for all a,b in S, a+b is not in S.
        # So i and j are "compatible" if for all k in S\{i,j}, arr[i]+arr[j] != arr[k], arr[i]+arr[k] != arr[j], arr[j]+arr[k] != arr[i]
        # But that depends on S. So we can't precompute a simple conflict graph.
        # Instead, let's precompute for each pair (i,j) whether arr[i]+arr[j] is in arr.
        
        # Precompute: for each i, what values v make arr[i]+v in arr?
        # And for each i, what values v make v+arr[i] in arr? (same thing)
        # And for each i, what values v make arr[i]-v in arr?
        
        # Let's precompute a dict: sum_pairs[i] = set of j such that arr[i]+arr[j] is in arr
        sum_pairs = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                if s in elem_to_idx:
                    k = elem_to_idx[s]
                    if k != i and k != j:
                        sum_pairs[i].add(j)
                        sum_pairs[j].add(i)
        
        # Greedy initialization with multiple orderings
        def greedy_init(order):
            included = set()
            for idx in order:
                c_val = arr[idx]
                ok = True
                # Check: is there a,b in included such that a+b = c_val?
                # i.e., for some a_idx in included, c_val - arr[a_idx] is in included
                for a_idx in included:
                    a_val = arr[a_idx]
                    b_val = c_val - a_val
                    if b_val in elem_to_idx and elem_to_idx[b_val] in included:
                        ok = False
                        break
                if ok:
                    # Check: is there b in included such that c_val + b is in included?
                    # i.e., for some b_idx in included, arr[idx] + arr[b_idx] is in included
                    for b_idx in included:
                        b_val = arr[b_idx]
                        s_val = c_val + b_val
                        if s_val in elem_to_idx and elem_to_idx[s_val] in included:
                            ok = False
                            break
                if ok:
                    included.add(idx)
            return included
        
        # Multiple greedy starts
        orders = [
            sorted(range(n), key=lambda i: -abs(arr[i])),
            sorted(range(n), key=lambda i: abs(arr[i])),
            list(range(n))
        ]
        rng.shuffle(orders[2])
        
        best_included = set()
        for order in orders:
            inc = greedy_init(order)
            if len(inc) > len(best_included):
                best_included = inc
        
        # Local search with efficient can_add
        included = set(best_included)
        
        def can_add(idx, current):
            """Check if idx can be added to current without violating sum-free."""
            c_val = arr[idx]
            # Check 1: for all a in current, c_val - a is not in current
            for a_idx in current:
                a_val = arr[a_idx]
                b_val = c_val - a_val
                if b_val in elem_to_idx and elem_to_idx[b_val] in current:
                    return False
            # Check 2: for all b in current, c_val + b is not in current
            for b_idx in current:
                b_val = arr[b_idx]
                s_val = c_val + b_val
                if s_val in elem_to_idx and elem_to_idx[s_val] in current:
                    return False
            return True
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Try adding excluded elements
            excluded = [i for i in range(n) if i not in included]
            rng.shuffle(excluded)
            for idx in excluded:
                if time.time() >= deadline - 5.0:
                    break
                if can_add(idx, included):
                    included.add(idx)
                    improved = True
            
            # Try swaps: remove one included, add one excluded
            if time.time() >= deadline - 5.0:
                break
            excluded = [i for i in range(n) if i not in included]
            inc_list = list(included)
            rng.shuffle(excluded)
            rng.shuffle(inc_list)
            
            # Try more swaps
            for add_idx in excluded[:min(len(excluded), 50)]:
                if time.time() >= deadline - 5.0:
                    break
                for rem_idx in inc_list[:min(len(inc_list), 30)]:
                    if time.time() >= deadline - 5.0:
                        break
                    trial = (included - {rem_idx}) | {add_idx}
                    if can_add(add_idx, trial - {add_idx}):
                        included = trial
                        improved = True
                        break
        
        # Final polish: try more aggressive local search
        if time.time() < deadline - 2.0:
            for _ in range(3):
                if time.time() >= deadline - 2.0:
                    break
                improved = False
                excluded = [i for i in range(n) if i not in included]
                rng.shuffle(excluded)
                for idx in excluded:
                    if time.time() >= deadline - 2.0:
                        break
                    if can_add(idx, included):
                        included.add(idx)
                        improved = True
                if not improved:
                    break
        
        result = [arr[i] for i in included]
        if not sum_free_check(result):
            result = [x for x in arr if x % 2]
        return result

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() >= deadline - 5.0:
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
        vals = instance(size, seed)
        sf = build_sum_free(vals)
        if not sum_free_check(sf):
            sf = [x for x in vals if x % 2]
        results.append(sf)

    return results
