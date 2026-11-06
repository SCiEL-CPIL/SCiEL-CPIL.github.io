import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin from 90s limit
    
    # Bitmask for DP sums. Max sum is 48 * 512 = 24576.
    # We use a mask to prevent integers from growing indefinitely,
    # although Python handles big ints, masking might help slightly with bit-length operations.
    MAX_SUM = 25000
    MASK = (1 << MAX_SUM) - 1

    def try_add_incremental(selected, x, dp_all, dp_without):
        """
        Attempts to add x to the set `selected`.
        Uses incremental DP state to check validity efficiently.
        
        Returns:
        (success, new_selected, new_dp_all, new_dp_without)
        """
        n = len(selected)
        
        # Check 1: Is x the mean of any subset of `selected` of size >= 2?
        # We need to check if k*x is a subset sum of `selected` with exactly k elements.
        # dp_all[k] contains sums of k elements from `selected`.
        # We only need to check k up to n (since we can't pick more than n elements).
        for k in range(2, n + 1):
            target = k * x
            if target > MAX_SUM:
                break
            if k < len(dp_all) and (dp_all[k] >> target) & 1:
                return False, None, None, None
        
        # Check 2: For each a in selected, is a the mean of a subset involving x?
        # The subset T must be from (selected U {x}) \ {a}.
        # If T contains x, T = {x} U U, where U is a subset of selected \ {a}.
        # Condition: sum(T) = |T| * a.
        # x + sum(U) = (1 + |U|) * a.
        # Let k = |T|. Then k >= 2 (since |T| >= 2).
        # sum(U) = k * a - x.
        # |U| = k - 1.
        # We need to check if (k*a - x) is a subset sum of (selected \ {a}) with exactly k-1 elements.
        # This information is in dp_without[a][k-1].
        
        for a in selected:
            dp_w = dp_without[a]
            # k ranges from 2 to n (max size of T is 1 + (n-1) = n)
            for k in range(2, n + 1):
                target = k * a - x
                if target < 0:
                    continue
                # Max sum of k-1 elements is (k-1)*512. If target exceeds this, impossible.
                if target > (k - 1) * 512:
                    continue
                if k - 1 < len(dp_w):
                    if (dp_w[k - 1] >> target) & 1:
                        return False, None, None, None
        
        # Valid. Update structures.
        new_selected = selected + [x]
        
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
        new_dp_all = [v & MASK for v in new_dp_all]
        
        # Update dp_without
        new_dp_without = {}
        # For existing elements a in selected[:-1]
        for a in selected:
            old_dp_w = dp_without[a]
            # new_dp_w[k] = old_dp_w[k] | (old_dp_w[k-1] << x)
            new_list = [0] * (n + 2)
            new_list[0] = 1
            for k in range(1, n + 2):
                val = 0
                if k <= n - 1:
                    val |= old_dp_w[k]
                if k - 1 >= 0 and k - 1 <= n - 1:
                    val |= (old_dp_w[k - 1] << x)
                new_list[k] = val
            new_dp_without[a] = [v & MASK for v in new_list]
        
        # For the new element x, dp_without[x] is the state of `selected` (which is the old dp_all)
        # Because selected \ {x} is just selected.
        # Note: dp_all currently holds the state of `selected`.
        # We need to copy it.
        new_dp_without[x] = list(dp_all)
        
        return True, new_selected, new_dp_all, new_dp_without

    def remove_element(selected, x, dp_all, dp_without):
        """
        Removes x from selected and updates DP structures.
        This is more expensive than adding, but necessary for swaps.
        We will rebuild DP from scratch for simplicity and reliability in the swap logic,
        or implement inverse updates. Rebuilding is O(n^2 * 512) which might be too slow.
        Let's try to implement inverse updates or just rebuild if n is small enough?
        Actually, for swaps, we remove one and add one.
        If we remove x, we can reconstruct DP from the remaining elements.
        Given the time constraints, rebuilding DP for the remaining n-1 elements
        takes O((n-1)^2 * 512) bit ops. With n=40, 1600 * 512 ~ 800k ops.
        This is fast enough in Python if done a few times per second.
        But we do many swaps.
        
        Alternative: Keep the list of elements and rebuild DP only when a swap is *successful*?
        No, we need valid state to check the *next* addition.
        
        Let's implement a helper to build DP from a list.
        """
        if x not in selected:
            return None, None, None
        
        new_selected = [v for v in selected if v != x]
        n_new = len(new_selected)
        
        # Rebuild dp_all
        dp_all_new = [0] * (n_new + 1)
        dp_all_new[0] = 1
        for v in new_selected:
            for k in range(n_new, 0, -1):
                dp_all_new[k] |= (dp_all_new[k-1] << v)
            # Mask
            for k in range(n_new + 1):
                dp_all_new[k] &= MASK
        
        # Rebuild dp_without
        dp_without_new = {}
        for a in new_selected:
            # dp_without[a] is DP of new_selected \ {a}
            elems = [v for v in new_selected if v != a]
            dp_w = [0] * (n_new) # Max size n_new-1
            dp_w[0] = 1
            for v in elems:
                for k in range(len(elems), 0, -1):
                    if k < len(dp_w):
                        dp_w[k] |= (dp_w[k-1] << v)
                for k in range(len(dp_w)):
                    dp_w[k] &= MASK
            # Pad to same size for consistency? Or just keep variable length?
            # The add function checks `if k-1 < len(dp_w)`. So variable length is fine.
            dp_without_new[a] = dp_w
            
        return new_selected, dp_all_new, dp_without_new

    def local_search(selected, dp_all, dp_without, deadline):
        n = len(selected)
        all_vals = list(range(1, 513))
        
        while time.time() < deadline:
            if n >= 48:
                break
                
            added = False
            
            # Try to add a random element
            candidates = [x for x in all_vals if x not in selected]
            if not candidates:
                break
                
            random.shuffle(candidates)
            
            # Try a few candidates
            for x in candidates[:20]:
                success, new_sel, new_dp_all, new_dp_w = try_add_incremental(selected, x, dp_all, dp_without)
                if success:
                    selected = new_sel
                    dp_all = new_dp_all
                    dp_without = new_dp_w
                    n += 1
                    added = True
                    break
            
            if added:
                continue
            
            # If no addition, try swaps
            # Pick a random element to remove
            if not selected:
                break
                
            remove_idx = random.randint(0, len(selected) - 1)
            remove_x = selected[remove_idx]
            
            # Remove it
            new_selected, new_dp_all, new_dp_w = remove_element(selected, remove_x, dp_all, dp_without)
            
            if new_selected is None:
                continue
                
            # Try to add a random element to the reduced set
            candidates = [x for x in all_vals if x not in new_selected and x != remove_x]
            random.shuffle(candidates)
            
            swap_success = False
            for x in candidates[:30]:
                success, final_sel, final_dp_all, final_dp_w = try_add_incremental(new_selected, x, new_dp_all, new_dp_w)
                if success:
                    selected = final_sel
                    dp_all = final_dp_all
                    dp_without = final_dp_w
                    # n remains the same
                    swap_success = True
                    break
                    
            if not swap_success:
                # Revert: We lost the state. We must rebuild from `new_selected` (which is selected without remove_x)
                # Wait, if swap failed, we want to restore the original state.
                # We don't have the original dp_all/dp_without easily accessible without rebuilding.
                # Rebuilding from scratch for the original set is expensive but safer than tracking history.
                # However, we can just try to re-add remove_x?
                # try_add_incremental(new_selected, remove_x, new_dp_all, new_dp_w)
                # This should restore the exact same state if the logic is deterministic.
                success, restored_sel, restored_dp_all, restored_dp_w = try_add_incremental(new_selected, remove_x, new_dp_all, new_dp_w)
                if success:
                    selected = restored_sel
                    dp_all = restored_dp_all
                    dp_without = restored_dp_w
                else:
                    # This should never happen if the set was valid.
                    # If it does, we have a bug. Let's rebuild from scratch.
                    dp_all, dp_without = build_dp_from_scratch(selected)
                    
        return selected, dp_all, dp_without

    def build_dp_from_scratch(selected):
        n = len(selected)
        dp_all = [0] * (n + 1)
        dp_all[0] = 1
        for v in selected:
            for k in range(n, 0, -1):
                dp_all[k] |= (dp_all[k-1] << v)
            for k in range(n + 1):
                dp_all[k] &= MASK
                
        dp_without = {}
        for a in selected:
            elems = [v for v in selected if v != a]
            dp_w = [0] * (n)
            dp_w[0] = 1
            for v in elems:
                for k in range(n-1, 0, -1):
                    if k < len(dp_w):
                        dp_w[k] |= (dp_w[k-1] << v)
                for k in range(len(dp_w)):
                    dp_w[k] &= MASK
            dp_without[a] = dp_w
        return dp_all, dp_without

    best_size = 0
    best_set = []
    
    # Multiple restarts
    while time.time() < deadline:
        # Start with an empty set
        selected = []
        dp_all = [1]
        dp_without = {}
        
        # Greedy initial construction
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if len(selected) >= 48:
                break
                
            success, new_sel, new_dp_all, new_dp_w = try_add_incremental(selected, x, dp_all, dp_without)
            if success:
                selected = new_sel
                dp_all = new_dp_all
                dp_without = new_dp_w
        
        # Local search to improve
        selected, dp_all, dp_without = local_search(selected, dp_all, dp_without, deadline)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = list(selected)
            if best_size == 48:
                break

    return best_set

