"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import copy

def construct():
    start_time = time.time()
    deadline = 55.0  # 5s safety margin
    
    best_set = []
    
    def try_greedy(order):
        selected = []
        # dp_all[k] is a bitmask of sums achievable with k elements from selected
        # dp_without[a_idx][k] is a bitmask of sums achievable with k elements from selected excluding selected[a_idx]
        # We store dp_without as a dict mapping value to list of bitmasks
        dp_all = [0]  # dp_all[0] = 1 (sum 0 with 0 elements) -> bit 0 set
        dp_all[0] = 1
        dp_without = {} # key: value, val: list of bitmasks
        
        # To support removal/lookup by value, we track indices or just values since distinct
        # But we need to update dp_without for existing elements when adding a new one.
        
        for x in order:
            if time.time() - start_time > deadline:
                break
            
            n = len(selected)
            
            # 1. Check if x is the mean of any subset of selected of size >= 2
            # Need sum = k*x for k in 2..n
            valid = True
            for k in range(2, n + 1):
                target = k * x
                if target >= 0:
                    # Check if bit 'target' is set in dp_all[k]
                    if (dp_all[k] >> target) & 1:
                        valid = False
                        break
            
            if not valid:
                continue
                
            # 2. Check if any a in selected is the mean of a subset involving x
            # For each a in selected:
            #   Need subset U of selected \ {a} of size k >= 1
            #   sum(U) = (k+1)*a - x
            for a in selected:
                # Get dp for selected \ {a}
                # dp_without[a] is a list of bitmasks
                # k ranges from 1 to n-1 (since U is subset of S\{a}, max size n-1)
                # But wait, if k=n, U would be S\{a} which has size n-1. So k max is n-1.
                # Actually, k is |U|. U subset of S\{a}. Max |U| = n-1.
                for k in range(1, n):
                    target = (k + 1) * a - x
                    if target < 0:
                        continue
                    # Check if target is achievable with k elements from S\{a}
                    if (dp_without[a][k] >> target) & 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Valid, add x
            selected.append(x)
            
            # Update dp_all
            # dp_all_new[k] = dp_all[k] | (dp_all[k-1] << x)
            new_dp_all = [0] * (n + 2)
            new_dp_all[0] = 1
            for k in range(1, n + 2):
                # dp_all[k] (old) is for size k
                # (dp_all[k-1] << x) is for size k using x
                val = dp_all[k] if k <= n else 0
                if k - 1 >= 0 and k - 1 <= n:
                    val |= (dp_all[k-1] << x)
                new_dp_all[k] = val
            dp_all = new_dp_all
            
            # Update dp_without
            new_dp_without = {}
            # For existing elements a
            for a in selected[:-1]: # all except x
                old_dp_w = dp_without[a]
                # new dp_w is for (S \ {a}) U {x}
                # size k: old_dp_w[k] | (old_dp_w[k-1] << x)
                new_list = [0] * (n + 2)
                new_list[0] = 1
                for k in range(1, n + 2):
                    val = old_dp_w[k] if k <= n - 1 else 0 # old size max n-1
                    if k - 1 >= 0 and k - 1 <= n - 1:
                        val |= (old_dp_w[k-1] << x)
                    new_list[k] = val
                new_dp_without[a] = new_list
            
            # For new element x
            # dp_without[x] is DP for S (the set before adding x)
            # So it is just a copy of the OLD dp_all
            new_dp_without[x] = copy.deepcopy(dp_all[:n+1]) # careful with length
            # Ensure length is consistent, though we only access up to current max size
            # Let's pad it to n+2 for safety
            while len(new_dp_without[x]) < n + 2:
                new_dp_without[x].append(0)
            
            dp_without = new_dp_without
            
            if len(selected) >= 48:
                break
        
        return selected

    # Try multiple strategies to maximize cardinality
    # 1. Ascending
    res1 = try_greedy(range(1, 513))
    if len(res1) > len(best_set):
        best_set = res1
        
    # 2. Descending
    if time.time() - start_time < deadline - 10:
        res2 = try_greedy(range(512, 0, -1))
        if len(res2) > len(best_set):
            best_set = res2
            
    # 3. Randomized shuffles if time permits
    if time.time() - start_time < deadline - 5:
        for _ in range(10):
            if time.time() - start_time > deadline - 2:
                break
            candidates = list(range(1, 513))
            random.shuffle(candidates)
            res_r = try_greedy(candidates)
            if len(res_r) > len(best_set):
                best_set = res_r
                if len(best_set) == 48:
                    break

    return best_set

# EVOLVE-BLOCK-END
