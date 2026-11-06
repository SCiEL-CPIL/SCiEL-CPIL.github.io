"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin from 90s limit
    
    def try_greedy(order):
        """
        Greedily builds a valid set.
        Returns the set of selected integers.
        """
        selected = []
        n = 0
        
        # dp_all[k] is a bitmask where bit s is set if sum s is achievable with exactly k elements from selected
        # Max sum for k elements is k * 512. Max k is 48. Max sum ~ 24576.
        # Python ints are arbitrary precision, so we don't need to mask explicitly, 
        # but we should be mindful of memory. 
        # dp_all[0] = 1 (sum 0)
        dp_all = [1]
        
        # dp_without[a][k] is bitmask of sums achievable with k elements from selected \ {a}
        dp_without = {}
        
        for x in order:
            if time.time() - start_time > deadline:
                break
            if n >= 48:
                break
            
            # Check 1: Is x the mean of any subset of selected of size >= 2?
            # Need: exists k >= 2, subset of selected with sum = k*x
            valid = True
            # k can be at most n.
            # Sum of k elements is at most k * 512.
            # k * x <= k * 512 is always true for x <= 512.
            # But we only care if bit is set.
            for k in range(2, n + 1):
                target = k * x
                if k < len(dp_all):
                    if (dp_all[k] >> target) & 1:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Check 2: For each a in selected, is a the mean of a subset involving x?
            # a is mean of U U {x} where U subset of selected \ {a, x}
            # sum(U) + x = k * a  => sum(U) = k * a - x
            # |U| = k - 1
            # So we need to check if (k*a - x) is achievable with k-1 elements from selected \ {a}.
            # This is stored in dp_without[a][k-1].
            for a in selected:
                if not valid:
                    break
                dp_w = dp_without[a]
                # k is the size of the subset {x} U U.
                # |U| = k - 1.
                # k can range from 1 (subset {x}, size 1, but task says subset of >= 2 OTHER elements.
                # Wait, "subset of at least two OTHER elements".
                # If the subset is {x, u1}, size 2. Then k=2?
                # Mean = (x + u1) / 2 = a => x + u1 = 2a.
                # Here U = {u1}, size 1. k-1 = 1.
                # So k starts from 2?
                # If subset is {x}, size 1. Not allowed (>= 2).
                # So the subset containing x has size m >= 2.
                # m = 1 (x) + |U|. So |U| >= 1.
                # So k in the formula sum(U) = k*a - x corresponds to the size of the *entire* subset?
                # Let's re-verify the definition.
                # "No element may be the exact arithmetic mean of any subset of at least two OTHER elements."
                # Let S be the set. Let a in S.
                # Let T be a subset of S \ {a} with |T| >= 2.
                # a is mean of T iff sum(T) = |T| * a.
                
                # Case 1: T does not contain x.
                # This is checked by dp_all (which includes x? No, dp_all is updated *after* x is added).
                # In the current loop, dp_all is for `selected` (which does NOT contain x yet).
                # So Check 1: Is x the mean of T where T subset of selected?
                # sum(T) = |T| * x.
                # This is what `Check 1` does. Correct.
                
                # Case 2: T contains x.
                # T = {x} U U, where U subset of selected \ {x}.
                # Since x is not in selected yet, U subset of selected.
                # Also, the condition is "T subset of OTHER elements".
                # For element a, T must be subset of S \ {a}.
                # If we are adding x, the new set is S' = S U {x}.
                # We need to ensure no a in S' is mean of T subset of S' \ {a}, |T|>=2.
                
                # Sub-case 2a: a = x.
                # Checked in Check 1. (x is mean of T subset of S).
                
                # Sub-case 2b: a in S (a != x).
                # T subset of S' \ {a} = (S \ {a}) U {x}.
                # T can be:
                #   1. Subset of S \ {a} (does not contain x).
                #      This was already valid for S. Since a was not mean of any subset of S \ {a},
                #      it is still not mean of any subset of S \ {a}. So no new violations here.
                #   2. Contains x. T = {x} U U, where U subset of S \ {a}.
                #      Condition: |T| >= 2. Since x is in T, and x != a, this is fine.
                #      But we need |T| >= 2.
                #      If U is empty, T={x}, |T|=1. Not allowed.
                #      So |U| >= 1.
                #      sum(T) = x + sum(U) = |T| * a = (1 + |U|) * a.
                #      Let k = |T|. Then k >= 2.
                #      sum(U) = k * a - x.
                #      |U| = k - 1.
                #      We need to check if sum(U) is achievable by k-1 elements from S \ {a}.
                #      This is exactly dp_without[a][k-1].
                
                # So the loop over k should be for k >= 2.
                # k-1 >= 1.
                for k in range(2, n + 1):
                    target = k * a - x
                    if target < 0:
                        continue
                    # Max sum of k-1 elements is (k-1)*512.
                    if target > (k - 1) * 512:
                        continue
                    # Check dp_without[a][k-1]
                    if k - 1 < len(dp_w):
                        if (dp_w[k - 1] >> target) & 1:
                            valid = False
                            break
            
            if not valid:
                continue
            
            # Add x to selected
            selected.append(x)
            
            # Update dp_all
            # new_dp_all[k] = dp_all[k] | (dp_all[k-1] << x)
            new_dp_all = [0] * (n + 2)
            new_dp_all[0] = 1
            for k in range(1, n + 2):
                val = 0
                if k <= n:
                    val |= dp_all[k]
                if k - 1 >= 0 and k - 1 <= n:
                    val |= (dp_all[k - 1] << x)
                new_dp_all[k] = val
            dp_all = new_dp_all
            
            # Update dp_without
            new_dp_without = {}
            # For existing elements a in selected[:-1]
            for a in selected[:-1]:
                old_dp_w = dp_without[a]
                # new_dp_w[k] = old_dp_w[k] | (old_dp_w[k-1] << x)
                # Note: old_dp_w is for S_old \ {a}. New set is S_old U {x}.
                # New set without a is (S_old \ {a}) U {x}.
                # So we can form sums using x or not.
                new_list = [0] * (n + 2)
                new_list[0] = 1
                for k in range(1, n + 2):
                    val = 0
                    if k <= n - 1:
                        val |= old_dp_w[k]
                    if k - 1 >= 0 and k - 1 <= n - 1:
                        val |= (old_dp_w[k - 1] << x)
                    new_list[k] = val
                new_dp_without[a] = new_list
            
            # For new element x, dp_without[x] is the old dp_all (which is S_old)
            # S_old \ {x} = S_old.
            # So dp_without[x] = dp_all (before update? No, dp_all was updated in place? 
            # No, we created new_dp_all. The old dp_all is still available as dp_all variable?
            # Wait, I did `dp_all = new_dp_all`. So old dp_all is lost if I don't save it.
            # I need the old dp_all for dp_without[x].
            # Let's save it before overwriting.
            
            # Correction: I should save old_dp_all = dp_all before the update loop.
            
            dp_without = new_dp_without
            n += 1
        
        return selected

    # --- Optimized Greedy with Pre-saved DP ---
    def try_greedy_fast(order):
        selected = []
        n = 0
        dp_all = [1]
        dp_without = {}
        
        for x in order:
            if time.time() - start_time > deadline:
                break
            if n >= 48:
                break
            
            valid = True
            # Check 1: x is mean of subset of selected
            for k in range(2, n + 1):
                target = k * x
                if k < len(dp_all) and ((dp_all[k] >> target) & 1):
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 2: a in selected is mean of subset involving x
            for a in selected:
                if not valid:
                    break
                dp_w = dp_without[a]
                for k in range(2, n + 1):
                    target = k * a - x
                    if target < 0:
                        continue
                    if target > (k - 1) * 512:
                        continue
                    if k - 1 < len(dp_w) and ((dp_w[k - 1] >> target) & 1):
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Save old dp_all for new element's dp_without
            old_dp_all = dp_all
            
            selected.append(x)
            
            # Update dp_all
            new_dp_all = [0] * (n + 2)
            new_dp_all[0] = 1
            for k in range(1, n + 2):
                val = 0
                if k <= n:
                    val |= old_dp_all[k]
                if k - 1 >= 0 and k - 1 <= n:
                    val |= (old_dp_all[k - 1] << x)
                new_dp_all[k] = val
            dp_all = new_dp_all
            
            # Update dp_without
            new_dp_without = {}
            for a in selected[:-1]:
                old_dp_w = dp_without[a]
                new_list = [0] * (n + 2)
                new_list[0] = 1
                for k in range(1, n + 2):
                    val = 0
                    if k <= n - 1:
                        val |= old_dp_w[k]
                    if k - 1 >= 0 and k - 1 <= n - 1:
                        val |= (old_dp_w[k - 1] << x)
                    new_list[k] = val
                new_dp_without[a] = new_list
            
            # For new element x, dp_without[x] is old_dp_all
            # We need to pad old_dp_all to length n+2 if necessary?
            # old_dp_all has length n+1 (indices 0..n).
            # new_dp_without[x] should have indices 0..n+1.
            # old_dp_all[k] is valid for k <= n.
            # For k = n+1, it should be 0.
            new_dp_without[x] = old_dp_all + [0]
            
            dp_without = new_dp_without
            n += 1
        
        return selected

    best_set = []
    best_len = 0
    
    # Strategy 1: Ascending
    res = try_greedy_fast(range(1, 513))
    if len(res) > best_len:
        best_set = res
        best_len = len(res)
    
    # Strategy 2: Descending
    if time.time() - start_time < deadline - 5:
        res = try_greedy_fast(range(512, 0, -1))
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
    
    # Strategy 3: Spread
    if time.time() - start_time < deadline - 5:
        candidates = list(range(1, 513))
        spread = []
        lo, hi = 0, len(candidates) - 1
        while lo <= hi:
            spread.append(candidates[lo])
            if lo != hi:
                spread.append(candidates[hi])
            lo += 1
            hi -= 1
        res = try_greedy_fast(spread)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
    
    # Strategy 4: Modular Orders
    for mod in [7, 11, 13, 17, 19, 23, 29]:
        if time.time() - start_time > deadline - 3:
            break
        buckets = [[] for _ in range(mod)]
        for x in range(1, 513):
            buckets[x % mod].append(x)
        order = []
        max_len = max(len(b) for b in buckets) if buckets else 0
        for i in range(max_len):
            for b in buckets:
                if i < len(b):
                    order.append(b[i])
        res = try_greedy_fast(order)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            if best_len >= 48:
                break
    
    # Strategy 5: Randomized Restarts
    # Use a fast shuffle
    for i in range(50):
        if time.time() - start_time > deadline - 2:
            break
        random.seed(i * 12345 + 67890)
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        res = try_greedy_fast(candidates)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            if best_len >= 48:
                break
    
    # Strategy 6: Local Search / Swap
    # If we haven't reached 48, try to improve best_set
    if best_len < 48 and time.time() - start_time < deadline - 5:
        current_set = set(best_set)
        # Identify candidates not in set
        candidates = [x for x in range(1, 513) if x not in current_set]
        
        # Try to swap elements
        # This is complex to implement efficiently with DP.
        # Instead, we can try to "re-greedy" with a perturbed order starting from best_set.
        # Or simply try more random restarts.
        # Given the time, more random restarts is safer.
        for i in range(100):
            if time.time() - start_time > deadline - 1:
                break
            random.seed(99999 + i)
            candidates = list(range(1, 513))
            random.shuffle(candidates)
            res = try_greedy_fast(candidates)
            if len(res) > best_len:
                best_set = res
                best_len = len(res)
                if best_len >= 48:
                    break
    
    return best_set

# EVOLVE-BLOCK-END
