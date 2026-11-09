"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # 1. Generate all 2,3,5-smooth numbers up to 128
    smooth_set = set()
    for a in range(0, 8):
        val_a = 2 ** a
        for b in range(0, 5):
            val_ab = val_a * (3 ** b)
            if val_ab > 128:
                break
            for c in range(0, 4):
                val_abc = val_ab * (5 ** c)
                if val_abc > 128:
                    break
                smooth_set.add(val_abc)
    
    smooth = sorted(smooth_set)
    n_smooth = len(smooth)
    val_to_idx = {v: i for i, v in enumerate(smooth)}
    
    # 2. Precompute incompatibility masks (bitmask of indices incompatible with each index)
    incompatible_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if smooth[j] % smooth[i] == 0 or smooth[i] % smooth[j] == 0:
                incompatible_mask[i] |= (1 << j)
                incompatible_mask[j] |= (1 << i)
    
    # 3. Precompute candidates for each target, sorted descending for better pruning
    candidates = [[] for _ in range(129)]
    for n in range(1, 129):
        for idx, val in enumerate(smooth):
            if val <= n:
                candidates[n].append((idx, val))
            else:
                break
        candidates[n].sort(key=lambda x: -x[1])
    
    # 4. Find minimal antichain representation for each target
    answers = [None] * 128
    
    def find_antichain(n, max_k=30):
        """Find minimum-k antichain of smooth numbers summing to n."""
        cand = candidates[n]
        if not cand:
            return None
        
        n_cand = len(cand)
        cand_vals = [v for _, v in cand]
        cand_idxs = [i for i, _ in cand]
        
        for k in range(1, min(max_k + 1, n_cand + 1)):
            if time.time() > DEADLINE:
                return None
            
            if k == 1:
                for val in cand_vals:
                    if val == n:
                        return [val]
                continue
            
            suffix_sum = [0] * (n_cand + 1)
            for i in range(n_cand - 1, -1, -1):
                suffix_sum[i] = suffix_sum[i + 1] + cand_vals[i]
            
            best = [None]
            
            def backtrack(start, remaining, depth, chosen_vals, chosen_mask):
                if best[0] is not None:
                    return
                if depth == k:
                    if remaining == 0:
                        best[0] = list(chosen_vals)
                    return
                
                if remaining < 0:
                    return
                
                if remaining > suffix_sum[start]:
                    return
                
                if (k - depth) > (n_cand - start):
                    return
                
                for pos in range(start, n_cand):
                    if best[0] is not None:
                        return
                    v = cand_vals[pos]
                    if v > remaining:
                        break
                    
                    idx = cand_idxs[pos]
                    if chosen_mask & incompatible_mask[idx]:
                        continue
                    
                    new_mask = chosen_mask | (1 << idx)
                    chosen_vals.append(v)
                    backtrack(pos + 1, remaining - v, depth + 1, chosen_vals, new_mask)
                    chosen_vals.pop()
            
            backtrack(0, n, 0, [], 0)
            if best[0] is not None:
                return best[0]
        
        return None
    
    for n in range(1, 129):
        if time.time() > DEADLINE:
            break
        answers[n - 1] = find_antichain(n)
    
    # Verify and fill any missing
    for i in range(128):
        if answers[i] is None:
            n = i + 1
            cand = candidates[n]
            found = False
            for k in range(1, min(31, len(cand) + 1)):
                vals = [v for _, v in cand]
                idxs = [i for i, _ in cand]
                for combo in combinations(range(len(vals)), k):
                    if sum(vals[j] for j in combo) == n:
                        valid = True
                        for a, b in combinations(combo, 2):
                            if incompatible_mask[idxs[a]] & (1 << idxs[b]):
                                valid = False
                                break
                        if valid:
                            answers[i] = [vals[j] for j in combo]
                            found = True
                            break
                if found:
                    break
    
    return answers
# EVOLVE-BLOCK-END
