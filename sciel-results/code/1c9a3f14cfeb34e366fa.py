"""Improved B&B MWIS solver with tight bounds and local search polish."""
# EVOLVE-BLOCK-START
import time
from math import gcd, lcm
from fractions import Fraction

def construct():
    deadline = time.time() + 55  # 55s safety margin before 60s kill
    t0 = time.time()
    
    n_max = 256
    small = list(range(2, 129))  # 2..128
    large = list(range(129, 257))  # 129..256
    n_small = len(small)
    n_large = len(large)
    
    small_idx = {v: i for i, v in enumerate(small)}
    large_idx = {v: i for i, v in enumerate(large)}
    
    # Small weights
    sw = [Fraction(1, v) for v in small]
    # Large weights
    lw = [Fraction(1, v) for v in large]
    
    # Small-small conflict: lcm(small[i], small[j]) <= 256
    ss_conflict = [0] * n_small
    for i in range(n_small):
        for j in range(i + 1, n_small):
            if lcm(small[i], small[j]) <= 256:
                ss_conflict[i] |= (1 << j)
                ss_conflict[j] |= (1 << i)
    
    # Small-large conflict: small[i] divides large[j]
    sl_conflict = [0] * n_small
    for i, s in enumerate(small):
        for j, L in enumerate(large):
            if L % s == 0:
                sl_conflict[i] |= (1 << j)
    
    # Float weights for fast bound computation
    sw_float = [float(w) for w in sw]
    lw_float = [float(w) for w in lw]
    
    # Suffix small weights (float)
    suffix_sw_float = [0.0] * (n_small + 1)
    for i in range(n_small - 1, -1, -1):
        suffix_sw_float[i] = suffix_sw_float[i + 1] + sw_float[i]
    
    total_lw_float = sum(lw_float)
    
    # Precompute: for each small vertex, the float sum of large weights it excludes
    sl_excl_float = [0.0] * n_small
    for i in range(n_small):
        for j in range(n_large):
            if (sl_conflict[i] >> j) & 1:
                sl_excl_float[i] += lw_float[j]
    
    # Greedy seed
    def greedy_small():
        selected = []
        for s in small:
            if all(lcm(s, y) > 256 for y in selected):
                selected.append(s)
        return selected
    
    greedy_sel = greedy_small()
    greedy_mask = 0
    greedy_excluded = 0
    for s in greedy_sel:
        i = small_idx[s]
        greedy_mask |= (1 << i)
        greedy_excluded |= sl_conflict[i]
    
    def compute_exact_score(mask, excluded):
        s = Fraction(0)
        for i in range(n_small):
            if (mask >> i) & 1:
                s += sw[i]
        for j in range(n_large):
            if not (excluded >> j) & 1:
                s += lw[j]
        return s
    
    best_score = compute_exact_score(greedy_mask, greedy_excluded)
    best_small_mask = greedy_mask
    best_large_excluded = greedy_excluded
    
    # Also try: no smalls, all larges
    all_large_score = Fraction(0)
    for j in range(n_large):
        all_large_score += lw[j]
    if all_large_score > best_score:
        best_score = all_large_score
        best_small_mask = 0
        best_large_excluded = 0
    
    # Try a few different greedy orderings
    import random
    random.seed(42)
    
    for trial in range(5):
        if time.time() > deadline:
            break
        # Weighted random greedy
        remaining = list(range(n_small))
        random.shuffle(remaining)
        sel_mask = 0
        sel_excl = 0
        for i in remaining:
            if (sel_mask & ss_conflict[i]) == 0:
                sel_mask |= (1 << i)
                sel_excl |= sl_conflict[i]
        sc = compute_exact_score(sel_mask, sel_excl)
        if sc > best_score:
            best_score = sc
            best_small_mask = sel_mask
            best_large_excluded = sel_excl
    
    # Branch and bound with tight bound
    # Key insight: bound = current_small_float + suffix_sw_float[pos] + large_float_bound
    # where large_float_bound = sum of lw_float[j] for j NOT in excluded_large
    # This is tight because remaining smalls can only add more exclusions.
    
    import sys
    sys.setrecursionlimit(10000)
    
    # For fast large-bound: maintain running float of available large sum
    # When we exclude a large set, subtract its float sum.
    
    def solve_bnb():
        nonlocal best_score, best_small_mask, best_large_excluded, deadline
        
        def dfs(pos, selected_mask, excluded_large, current_score, current_float, avail_large_float):
            nonlocal best_score, best_small_mask, best_large_excluded, deadline
            
            if time.time() > deadline:
                return
            
            if pos == n_small:
                # Compute exact score
                large_score = Fraction(0)
                for j in range(n_large):
                    if not (excluded_large >> j) & 1:
                        large_score += lw[j]
                total = current_score + large_score
                if total > best_score:
                    best_score = total
                    best_small_mask = selected_mask
                    best_large_excluded = excluded_large
                return
            
            # Tight upper bound
            if current_float + suffix_sw_float[pos] + avail_large_float <= float(best_score) + 1e-12:
                return
            
            # Option 1: Include vertex pos (if compatible)
            if (selected_mask & ss_conflict[pos]) == 0:
                new_mask = selected_mask | (1 << pos)
                new_excluded = excluded_large | sl_conflict[pos]
                new_score = current_score + sw[pos]
                new_float = current_float + sw_float[pos]
                # Subtract excluded large weights from avail
                new_avail_large = avail_large_float - sl_excl_float[pos]
                # But we might double-subtract if some were already excluded.
                # Actually sl_conflict[pos] only adds NEW exclusions if not already in excluded_large.
                # For the bound, over-subtracting is fine (makes bound tighter, still valid upper bound).
                # Wait, no: if we over-subtract, the bound becomes LOWER than true, which is invalid.
                # We need: avail_large_float should be sum of lw_float[j] for j not in excluded_large.
                # When we add sl_conflict[pos] to excluded_large, the new avail is:
                #   old_avail - sum(lw_float[j] for j in sl_conflict[pos] AND j not in old excluded_large)
                # For simplicity and correctness, just compute it fresh or use a safe overestimate.
                # Safe approach: new_avail_large = avail_large_float (no subtraction) - this is looser but valid.
                # Or: compute the actual new avail. Let's do it properly but efficiently.
                
                # Actually, for the bound to be valid (upper bound), we need:
                # new_avail_large >= actual sum of lw_float for j not in new_excluded
                # Since new_excluded = old_excluded | sl_conflict[pos],
                # actual new avail = old_avail - sum(lw_float[j] for j in sl_conflict[pos] & ~old_excluded)
                # So new_avail_large = old_avail - that_sum <= old_avail
                # If we just use old_avail, it's a valid (looser) upper bound.
                # If we subtract the full sl_excl_float[pos], we might under-subtract if some were already excluded,
                # making new_avail too low -> INVALID bound.
                # So safest: use avail_large_float (no change) for the bound.
                # This is looser than ideal but correct.
                
                dfs(pos + 1, new_mask, new_excluded, new_score, new_float, avail_large_float)
            
            # Option 2: Exclude vertex pos
            dfs(pos + 1, selected_mask, excluded_large, current_score, current_float, avail_large_float)
        
        dfs(0, 0, 0, Fraction(0), 0.0, total_lw_float)
    
    solve_bnb()
    
    # Local search: 2-exchange polish
    if time.time() < deadline - 5:
        # Build current solution sets
        cur_small_set = set()
        for i in range(n_small):
            if (best_small_mask >> i) & 1:
                cur_small_set.add(small[i])
        cur_large_excluded_set = set()
        for j in range(n_large):
            if (best_large_excluded >> j) & 1:
                cur_large_excluded_set.add(large[j])
        
        # For 2-exchange: try swapping one small in with one small out, or adding/removing a small
        # When we change the small set, the large set updates automatically.
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            
            # Try adding each non-selected small
            for i in range(n_small):
                if (best_small_mask >> i) & 1:
                    continue
                # Check if adding i is compatible with current small set
                if (best_small_mask & ss_conflict[i]) != 0:
                    continue
                new_mask = best_small_mask | (1 << i)
                new_excl = best_large_excluded | sl_conflict[i]
                new_score = compute_exact_score(new_mask, new_excl)
                if new_score > best_score:
                    best_score = new_score
                    best_small_mask = new_mask
                    best_large_excluded = new_excl
                    improved = True
            
            # Try removing each selected small
            for i in range(n_small):
                if not (best_small_mask >> i) & 1:
                    continue
                new_mask = best_small_mask & ~(1 << i)
                # Recompute excluded: union of sl_conflict for remaining selected
                new_excl = 0
                for k in range(n_small):
                    if (new_mask >> k) & 1:
                        new_excl |= sl_conflict[k]
                new_score = compute_exact_score(new_mask, new_excl)
                if new_score > best_score:
                    best_score = new_score
                    best_small_mask = new_mask
                    best_large_excluded = new_excl
                    improved = True
            
            # Try 1-in-1-out exchange
            for i_out in range(n_small):
                if not (best_small_mask >> i_out) & 1:
                    continue
                for i_in in range(n_small):
                    if (best_small_mask >> i_in) & 1:
                        continue
                    # Check i_in compatible with (small_set - {i_out})
                    temp_mask = best_small_mask & ~(1 << i_out)
                    if (temp_mask & ss_conflict[i_in]) != 0:
                        continue
                    new_mask = temp_mask | (1 << i_in)
                    new_excl = 0
                    for k in range(n_small):
                        if (new_mask >> k) & 1:
                            new_excl |= sl_conflict[k]
                    new_score = compute_exact_score(new_mask, new_excl)
                    if new_score > best_score:
                        best_score = new_score
                        best_small_mask = new_mask
                        best_large_excluded = new_excl
                        improved = True
                        break
                if improved:
                    break
    
    # Build final set
    result = []
    for i in range(n_small):
        if (best_small_mask >> i) & 1:
            result.append(small[i])
    for j in range(n_large):
        if not (best_large_excluded >> j) & 1:
            result.append(large[j])
    
    if not result:
        return [2]
    
    # Verify constraint
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if lcm(result[i], result[j]) <= 256:
                return greedy_sel
    
    return result
# EVOLVE-BLOCK-END
