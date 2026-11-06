"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin from 90s limit
    
    best_set = []
    best_len = 0
    
    MAX_SUM = 24576  # 48 * 512
    
    def compute_conflict_density(x, dp_all, n):
        """Count how many subset-sum targets k*x (k>=2) are currently achievable."""
        count = 0
        for k in range(2, n + 1):
            target = k * x
            if target > MAX_SUM:
                break
            if (dp_all[k] >> target) & 1:
                count += 1
        return count
    
    def try_greedy_conflict_density(initial_order, max_conflicts=0):
        """
        Greedy selection using conflict density heuristic.
        At each step, pick the candidate with the fewest conflict targets.
        If max_conflicts > 0, only consider candidates with conflict count <= max_conflicts.
        """
        selected = []
        n = 0
        dp_all = [1]  # dp_all[0] = 1
        dp_without = {}
        remaining = set(initial_order)
        
        while remaining and n < 48:
            if time.time() - start_time > deadline:
                break
            
            # Score each remaining candidate by conflict density
            best_candidate = None
            best_conflicts = float('inf')
            best_valid = False
            
            for x in remaining:
                # Check 1: Is x the mean of any subset of 'selected' of size >= 2?
                valid = True
                conflict_count = 0
                for k in range(2, n + 1):
                    target = k * x
                    if target > MAX_SUM:
                        break
                    if (dp_all[k] >> target) & 1:
                        conflict_count += 1
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check 2: For each a in selected, is a the mean of a subset involving x?
                for a in selected:
                    dp_w = dp_without[a]
                    for k in range(1, n):
                        target = (k + 1) * a - x
                        if target < 0 or target > MAX_SUM:
                            continue
                        if (dp_w[k] >> target) & 1:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # This candidate is valid; use conflict_count as heuristic score
                # (lower is better - fewer potential future conflicts)
                if conflict_count < best_conflicts:
                    best_conflicts = conflict_count
                    best_candidate = x
                    best_valid = True
            
            if not best_valid:
                break
            
            x = best_candidate
            remaining.discard(x)
            
            # Add x to selected
            selected.append(x)
            
            # Save old state for x's exclusion set
            old_dp_all = dp_all
            
            # Update dp_all
            new_dp_all = [0] * (n + 2)
            new_dp_all[0] = 1
            for k in range(1, n + 2):
                val = old_dp_all[k] if k <= n else 0
                if k - 1 >= 0 and k - 1 <= n:
                    val |= (old_dp_all[k-1] << x)
                new_dp_all[k] = val
            dp_all = new_dp_all
            
            # Update dp_without for existing elements
            new_dp_without = {}
            for a in selected[:-1]:
                old_dp_w = dp_without[a]
                new_list = [0] * (n + 2)
                new_list[0] = 1
                for k in range(1, n + 2):
                    val = old_dp_w[k] if k <= n - 1 else 0
                    if k - 1 >= 0 and k - 1 <= n - 1:
                        val |= (old_dp_w[k-1] << x)
                    new_list[k] = val
                new_dp_without[a] = new_list
            
            # For x, the set of others is the old selected set.
            new_dp_without[x] = old_dp_all + [0] * max(0, (n + 1) - len(old_dp_all))
            
            dp_without = new_dp_without
            n += 1
        
        return selected
    
    # Try multiple initial orderings with the conflict density heuristic
    orderings = []
    
    # 1. Ascending
    orderings.append(list(range(1, 513)))
    
    # 2. Descending
    orderings.append(list(range(512, 0, -1)))
    
    # 3. Spread (low-high alternation)
    spread = []
    lo, hi = 1, 512
    while lo <= hi:
        spread.append(lo)
        if lo != hi:
            spread.append(hi)
        lo += 1
        hi -= 1
    orderings.append(spread)
    
    # 4. Reverse spread
    rev_spread = []
    lo, hi = 1, 512
    while lo <= hi:
        rev_spread.append(hi)
        if lo != hi:
            rev_spread.append(lo)
        lo += 1
        hi -= 1
    orderings.append(rev_spread)
    
    # 5. Random shuffles
    for i in range(20):
        random.seed(i * 1000)
        cand = list(range(1, 513))
        random.shuffle(cand)
        orderings.append(cand)
    
    for order in orderings:
        if time.time() - start_time > deadline:
            break
        res = try_greedy_conflict_density(order)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            if best_len >= 48:
                break
    
    return best_set

# EVOLVE-BLOCK-END
