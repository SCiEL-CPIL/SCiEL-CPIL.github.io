"""Sum-free subset maximization via greedy + local search + exact B&B for small n."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c for a,b,c in S (a=b allowed)."""
        S_set = set(S)
        for a in S_set:
            for b in S_set:
                if a + b in S_set:
                    return False
        return True

    def greedy_build(elements, order):
        """Greedy: add elements in given order if it keeps the set sum-free."""
        S = []
        S_set = set()
        for x in order:
            ok = True
            if 2 * x in S_set:
                ok = False
            if ok:
                for a in S:
                    if a + x in S_set:
                        ok = False
                        break
            if ok:
                S.append(x)
                S_set.add(x)
        return S

    def local_search(S, elements, max_iters=5000, timeout=10.0):
        """Try to improve S by swapping out one element and in another."""
        S_set = set(S)
        S_list = list(S)
        available = [x for x in elements if x not in S_set]
        best_size = len(S_list)
        no_improve_count = 0
        t0 = time.time()

        while max_iters > 0 and no_improve_count < 200:
            max_iters -= 1
            if time.time() - t0 > timeout:
                break
            # Try random swap: remove r from S, add a from available
            r_idx = random.randrange(len(S_list))
            r = S_list[r_idx]
            a = random.choice(available)
            if a == r:
                continue
            # Tentative: remove r, add a
            S_set.remove(r)
            S_list.pop(r_idx)
            # Check if adding a keeps sum-free
            ok = True
            if 2 * a in S_set:
                ok = False
            if ok:
                for b in S_list:
                    if b + a in S_set:
                        ok = False
                        break
            if not ok:
                # Revert
                S_list.append(r)
                S_set.add(r)
                continue
            # Add a
            S_list.append(a)
            S_set.add(a)
            best_size = len(S_list)
            no_improve_count = 0
            # Update available
            if a in available:
                available.remove(a)
            available.append(r)
        return S_list

    def exact_bb(vals, time_limit=5.0):
        """
        Exact branch-and-bound for maximum sum-free subset.
        Only feasible for small n (<= 30).
        State: (current_set, remaining_indices)
        Uses a simple DFS with pruning.
        """
        n = len(vals)
        if n > 30:
            return None
        
        # Precompute conflicts: for each element i, which pairs (j,k) in vals sum to vals[i]?
        # Actually, we need: for a set S, x is addable if no a,b in S with a+b=x,
        # and no a in S with x+a in S, and 2x not in S.
        
        # We'll use a bitmask for remaining candidates if n <= 60? No, n<=30 so bitmask is fine.
        # But Python ints can be arbitrary precision, so bitmask up to n=30 is trivial.
        
        best_size = 0
        best_set = set()
        t0 = time.time()
        
        # Sort vals to help pruning? Maybe by absolute value or something.
        # Let's try sorting by "centrality" or just natural order.
        
        def dfs(current_set, remaining_mask, depth):
            nonlocal best_size, best_set
            if time.time() - t0 > time_limit:
                return
            
            # Upper bound: current_size + popcount(remaining_mask)
            curr_size = len(current_set)
            if curr_size + bin(remaining_mask).count('1') <= best_size:
                return
            
            if remaining_mask == 0:
                if curr_size > best_size:
                    best_size = curr_size
                    best_set = set(current_set)
                return
            
            # Try adding an element from remaining
            # Heuristic: try elements that are "safe" first?
            # For simplicity, iterate through remaining elements.
            
            # Find the lowest set bit
            # We'll iterate through all bits in remaining_mask
            
            r = remaining_mask
            while r:
                if time.time() - t0 > time_limit:
                    return
                
                # Get lowest set bit
                lsb = r & (-r)
                idx = lsb.bit_length() - 1
                x = vals[idx]
                
                # Check if x can be added to current_set
                ok = True
                if 2 * x in current_set:
                    ok = False
                if ok:
                    for a in current_set:
                        if a + x in current_set:
                            ok = False
                            break
                
                if ok:
                    # Add x
                    new_set = current_set | {x}
                    # New remaining mask: remove x and any elements that conflict with x
                    # An element y conflicts with x if:
                    # 1. x + y in current_set (so y+x = c in S)
                    # 2. y + x in current_set (same)
                    # 3. x + x = 2x, and if 2x is in remaining, it conflicts? 
                    #    Actually, if we add x, then for any future y, we need y+x not in new_set.
                    #    So y is forbidden if y+x is in new_set (which is current_set union {x}).
                    #    Also, if x+y is in new_set.
                    #    So forbidden y: y such that y+x in new_set OR x+y in new_set.
                    #    Since addition is commutative, this is just: y such that y+x in new_set.
                    #    Also, y such that x+y = 2x? No, that's specific.
                    #    More generally, if we have a in new_set, then y = a - x is forbidden if y+x=a.
                    #    Also, y = x - a is forbidden if y+x=a? No, y+x=a means y=a-x.
                    #    So for each a in new_set, y = a - x is forbidden.
                    #    Also, if 2x is in new_set, then x is forbidden? No, x is already added.
                    #    But if we consider adding x, then 2x becomes a "sum" of x+x.
                    #    So any future y where y+x = 2x, i.e., y=x, is already removed.
                    #    And any y where x+y = 2x, same.
                    #    So the forbidden set is: {a - x for a in new_set} union {x + a for a in new_set}?
                    #    Wait. If we add x, then for any future element y, we must ensure:
                    #    - y + x not in new_set
                    #    - y + a not in new_set for any a in current_set (but this was already handled?)
                    #    Actually, the condition is: no a,b,c in S such that a+b=c.
                    #    When we add x, new triples involving x:
                    #    - x + a = c for a,c in current_set -> x is bad (checked above)
                    #    - x + x = c for c in current_set -> x is bad (checked above: 2x in current_set)
                    #    - a + x = c for a,c in current_set -> same as x+a=c
                    #    For future elements y:
                    #    - x + y = c for c in new_set -> y is forbidden if x+y in new_set
                    #    - y + x = c for c in new_set -> same
                    #    - y + a = c for a,c in new_set -> this is independent of x, already handled by recursion
                    #    - a + y = c for a,c in new_set -> same
                    #    So the new forbidden elements are those y such that x+y in new_set.
                    #    new_set = current_set union {x}
                    #    So forbidden y: {c - x for c in new_set}
                    #    Note: if c=x, then y=0, but 0 is not in vals (nonzero integers).
                    #    If c=2x, then y=x, which is already removed.
                    
                    forbidden_mask = 0
                    for c in new_set:
                        y = c - x
                        if y in vals:
                            # Find index of y
                            # We need a mapping from value to index
                            pass
                    # This is getting complicated with value-to-index mapping.
                    # Let's use a different approach: just check each remaining element.
                    
                    new_remaining = remaining_mask & ~lsb
                    # For each element in new_remaining, check if it conflicts with x
                    # An element y conflicts with x if x+y in new_set or y+x in new_set
                    # Since x+y = y+x, just check x+y in new_set.
                    
                    # Iterate through new_remaining
                    r2 = new_remaining
                    while r2:
                        lsb2 = r2 & (-r2)
                        idx2 = lsb2.bit_length() - 1
                        y = vals[idx2]
                        if x + y in new_set or y + x in new_set:
                            new_remaining &= ~lsb2
                        r2 &= r2 - 1
                    
                    dfs(new_set, new_remaining, depth + 1)
                
                # Option: don't add x
                # But we need to be careful: if we skip x, we still need to ensure
                # that x doesn't conflict with future elements? No, if x is not in S,
                # it doesn't create conflicts.
                
                r &= r - 1  # remove lowest set bit
        
        dfs(set(), (1 << n) - 1, 0)
        
        return list(best_set)

    def solve_instance(elements, time_budget=30.0):
        """Solve one instance: try multiple greedy orderings, then local search."""
        n = len(elements)
        best = []
        best_size = 0

        # For small instances, try exact B&B first
        if n <= 25:
            t0 = time.time()
            exact_sol = exact_bb(elements, time_limit=min(10.0, time_budget))
            if exact_sol is not None:
                best = exact_sol
                best_size = len(best)
                if time.time() - t0 < time_budget:
                    # Try to improve with local search (in case exact wasn't optimal due to time)
                    improved = local_search(best, elements, max_iters=5000, timeout=min(5.0, time_budget - (time.time() - t0)))
                    if len(improved) > best_size:
                        best = improved
                        best_size = len(improved)
                return best

        # Multiple greedy orderings
        orderings = []
        # Natural order
        orderings.append(list(elements))
        # Reverse
        orderings.append(list(reversed(elements)))
        # By absolute value
        orderings.append(sorted(elements, key=abs))
        orderings.append(sorted(elements, key=abs, reverse=True))
        # By value
        orderings.append(sorted(elements))
        # Random shuffles
        rng = random.Random(42)
        for _ in range(5):
            o = list(elements)
            rng.shuffle(o)
            orderings.append(o)

        for order in orderings:
            if time.time() - start_time > DEADLINE:
                break
            S = greedy_build(elements, order)
            if len(S) > best_size:
                best_size = len(S)
                best = list(S)

        # Local search from best greedy result
        if time.time() - start_time < DEADLINE - 5:
            improved = local_search(best, elements, max_iters=20000, timeout=min(time_budget, DEADLINE - time.time() - 5))
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)

        # Multiple local search restarts from different greedy solutions
        for order in orderings[:3]:
            if time.time() - start_time > DEADLINE - 5:
                break
            S = greedy_build(elements, order)
            improved = local_search(S, elements, max_iters=10000, timeout=min(10.0, DEADLINE - time.time() - 5))
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)

        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() - start_time > DEADLINE:
            results.append([])
            continue
        elems = instance(size, seed)
        # Allocate more time to larger instances
        budget = 40.0 if size <= 60 else (30.0 if size <= 120 else 25.0)
        sol = solve_instance(elems, time_budget=budget)
        # Verify sum-free
        if not is_sum_free(sol):
            # Fallback to odd numbers
            sol = [x for x in elems if x % 2 != 0]
        results.append(sol)

    return results
# EVOLVE-BLOCK-END
